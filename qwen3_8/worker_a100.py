"""Motor Qwen3.8 ejecutado exclusivamente con qwen3_8/.venv/bin/python."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import struct
import threading
import time
import traceback
import uuid
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# El wheel de FlashInfer intenta compilar el sampler top-k/top-p durante el
# warmup. Los ML Runtimes de Cloudera no garantizan ninja ni un toolkit NVCC
# visible. El sampler nativo evita esa compilacion JIT; la atencion sigue
# usando TRITON_ATTN y los pesos FP8 siguen usando Marlin.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Estas variables son configuración de este wrapper, no variables oficiales de
# vLLM. Se capturan antes de importarlo y se eliminan del entorno para que vLLM
# 0.29 no emita "Unknown vLLM environment variable". Los nombres QWEN_* son los
# canónicos; los VLLM_* anteriores se mantienen como alias retrocompatibles.
_LEGACY_CONFIG = {
    name: os.environ.pop(name, None)
    for name in (
        "VLLM_TENSOR_PARALLEL_SIZE",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_KV_CACHE_DTYPE",
        "VLLM_ENFORCE_EAGER",
        "VLLM_CPU_OFFLOAD_GB",
    )
}

from vllm import LLM, SamplingParams


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")
SERVED_MODEL_NAME = os.getenv("QWEN_SERVED_MODEL_NAME", "qwen3.8-27b-fp8")
_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 128 * 1024 * 1024
_GENERATION_LOCK = threading.Lock()


def _config_value(name: str, legacy_name: str, default: Any) -> str:
    value = os.getenv(name)
    if value is None:
        value = _LEGACY_CONFIG.get(legacy_name)
    return str(default if value is None else value)


def _env_int(
    name: str, legacy_name: str, default: int, minimum: int, maximum: int
) -> int:
    value = int(_config_value(name, legacy_name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


def _env_float(
    name: str, legacy_name: str, default: float, minimum: float, maximum: float
) -> float:
    value = float(_config_value(name, legacy_name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


def _env_bool(name: str, legacy_name: str, default: bool) -> bool:
    value = _config_value(
        name, legacy_name, "true" if default else "false"
    ).strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} debe ser booleano")
    return value in {"1", "true", "yes", "on"}


TENSOR_PARALLEL_SIZE = _env_int(
    "QWEN_TENSOR_PARALLEL_SIZE", "VLLM_TENSOR_PARALLEL_SIZE", 1, 1, 8
)
MAX_MODEL_LEN = _env_int(
    "QWEN_MAX_MODEL_LEN", "VLLM_MAX_MODEL_LEN", 262144, 2048, 262144
)
GPU_MEMORY_UTILIZATION = _env_float(
    "QWEN_GPU_MEMORY_UTILIZATION",
    "VLLM_GPU_MEMORY_UTILIZATION",
    0.90,
    0.50,
    0.95,
)
MAX_NUM_SEQS = _env_int(
    "QWEN_MAX_NUM_SEQS", "VLLM_MAX_NUM_SEQS", 1, 1, 64
)
MAX_NUM_BATCHED_TOKENS = _env_int(
    "QWEN_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    8192,
    2048,
    131072,
)
MAX_OUTPUT_TOKENS = _env_int(
    "QWEN_MAX_OUTPUT_TOKENS", "QWEN_MAX_OUTPUT_TOKENS", 512, 1, 8192
)
CPU_OFFLOAD_GB = _env_float(
    "QWEN_CPU_OFFLOAD_GB", "VLLM_CPU_OFFLOAD_GB", 0.0, 0.0, 128.0
)
KV_CACHE_DTYPE = _config_value(
    "QWEN_KV_CACHE_DTYPE", "VLLM_KV_CACHE_DTYPE", "bfloat16"
)
ENFORCE_EAGER = _env_bool(
    "QWEN_ENFORCE_EAGER", "VLLM_ENFORCE_EAGER", True
)
ATTENTION_BACKEND = os.getenv("QWEN_ATTENTION_BACKEND", "TRITON_ATTN")
MAX_IMAGES = _env_int(
    "QWEN_MAX_IMAGES_PER_PROMPT", "QWEN_MAX_IMAGES_PER_PROMPT", 1, 0, 4
)
_allowed_domains = [
    domain.strip()
    for domain in os.getenv("VLLM_ALLOWED_MEDIA_DOMAINS", "").split(",")
    if domain.strip()
]


def _load_engine() -> LLM:
    try:
        shm = os.statvfs("/dev/shm")
        print(
            "SHM_DIAGNOSTIC "
            f"total={shm.f_blocks * shm.f_frsize / 1024**2:.0f} MiB "
            f"free={shm.f_bavail * shm.f_frsize / 1024**2:.0f} MiB",
            flush=True,
        )
    except OSError as exc:
        print(f"SHM_DIAGNOSTIC unavailable={exc!r}", flush=True)

    import numpy
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8 requiere una GPU NVIDIA visible en el deployment")
    gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        "GPU_DIAGNOSTIC "
        f"name={torch.cuda.get_device_name(0)} memory={gpu_total_gib:.1f} GiB "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"numpy={numpy.__version__}",
        flush=True,
    )
    if gpu_total_gib < 70:
        raise RuntimeError(
            "Este perfil requiere una A100 de 80 GB completa. Para una A100 de "
            "40 GB hace falta un perfil de contexto reducido validado aparte."
        )

    effective_kv_cache_dtype = KV_CACHE_DTYPE
    compute_capability = torch.cuda.get_device_capability(0)
    if KV_CACHE_DTYPE.lower().startswith("fp8") and compute_capability < (8, 9):
        effective_kv_cache_dtype = "bfloat16"
        print(
            "QWEN_CONFIG_ADJUSTMENT requested_kv_cache_dtype="
            f"{KV_CACHE_DTYPE} effective_kv_cache_dtype=bfloat16 reason="
            f"compute_capability_{compute_capability[0]}{compute_capability[1]}_lt_89",
            flush=True,
        )

    print(
        "VLLM_CONFIG "
        f"model={MODEL_ID} tp={TENSOR_PARALLEL_SIZE} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS} "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION} "
        f"kv_cache_dtype={effective_kv_cache_dtype} "
        f"attention_backend={ATTENTION_BACKEND} "
        f"enforce_eager={ENFORCE_EAGER}",
        flush=True,
    )

    return LLM(
        model=MODEL_ID,
        served_model_name=SERVED_MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        kv_cache_dtype=effective_kv_cache_dtype,
        attention_backend=ATTENTION_BACKEND,
        cpu_offload_gb=CPU_OFFLOAD_GB,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=False,
        hf_token=os.getenv("HF_TOKEN") or None,
        allowed_media_domains=_allowed_domains or None,
        limit_mm_per_prompt={"image": MAX_IMAGES, "video": 0},
    )


def _messages(args: dict[str, Any]) -> list[dict[str, Any]]:
    messages = args.get("messages")
    if messages is None:
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Indique un prompt no vacío o un array messages")
        messages = [{"role": "user", "content": prompt}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages debe ser un array no vacío")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Cada mensaje debe ser un objeto JSON")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Cada mensaje necesita un role válido")
        if not isinstance(message.get("content"), (str, list)):
            raise ValueError("content debe ser texto o una lista multimodal")
    return messages


def _number(
    args: dict[str, Any], name: str, default: float, minimum: float, maximum: float
) -> float:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser numérico")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return result


def _split_reasoning(text: str) -> tuple[str | None, str]:
    if "</think>" not in text:
        return None, text.strip()
    reasoning, answer = text.split("</think>", 1)
    reasoning = re.sub(r"^\s*<think>\s*", "", reasoning).strip()
    return reasoning or None, answer.strip()


def _predict(engine: LLM, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("La entrada debe ser un objeto JSON")
    messages = _messages(args)
    thinking = bool(args.get("enable_thinking", False))
    reasoning_effort = str(args.get("reasoning_effort", "low"))
    if reasoning_effort not in {"low", "medium", "xhigh"}:
        raise ValueError("reasoning_effort debe ser low, medium o xhigh")

    default_temperature = 1.0 if thinking else 0.7
    default_top_p = 0.95 if thinking else 0.80
    default_presence_penalty = 0.0 if thinking else 1.5
    sampling = SamplingParams(
        max_tokens=int(_number(args, "max_tokens", 128, 1, MAX_OUTPUT_TOKENS)),
        temperature=_number(args, "temperature", default_temperature, 0.0, 2.0),
        top_p=_number(args, "top_p", default_top_p, 0.01, 1.0),
        top_k=int(_number(args, "top_k", 20, -1, 1000)),
        min_p=_number(args, "min_p", 0.0, 0.0, 1.0),
        presence_penalty=_number(
            args, "presence_penalty", default_presence_penalty, -2.0, 2.0
        ),
        repetition_penalty=_number(
            args, "repetition_penalty", 1.0, 0.01, 2.0
        ),
        seed=int(_number(args, "seed", 0, 0, 2**31 - 1)),
        stop=args.get("stop"),
    )

    with _GENERATION_LOCK:
        result = engine.chat(
            messages=messages,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={
                "enable_thinking": thinking,
                "preserve_thinking": bool(args.get("preserve_thinking", True)),
                "reasoning_effort": reasoning_effort,
            },
        )[0]

    generated = result.outputs[0]
    reasoning, answer = _split_reasoning(generated.text)
    prompt_tokens = len(result.prompt_token_ids or [])
    completion_tokens = len(generated.token_ids or [])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "reasoning_content": reasoning,
                },
                "finish_reason": generated.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("El proxy Cloudera cerró la conexión IPC")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    (size,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if size > _MAX_MESSAGE_BYTES:
        raise RuntimeError(f"Petición IPC demasiado grande: {size} bytes")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Petición IPC inválida")
    return value


def _send_message(sock: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Respuesta IPC demasiado grande: {len(payload)} bytes")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _serve(ipc_fd: int) -> None:
    sock = socket.socket(fileno=ipc_fd)
    try:
        try:
            engine = _load_engine()
            _send_message(sock, {"ok": True, "status": "ready"})
        except BaseException:
            error = traceback.format_exc()
            _send_message(sock, {"ok": False, "error": error})
            raise

        while True:
            try:
                request = _recv_message(sock)
            except EOFError:
                return
            try:
                if request.get("command") != "predict":
                    raise ValueError("Comando IPC no soportado")
                result = _predict(engine, request.get("args"))
                _send_message(sock, {"ok": True, "result": result})
            except BaseException:
                _send_message(sock, {"ok": False, "error": traceback.format_exc()})
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-fd", required=True, type=int)
    args = parser.parse_args()
    _serve(args.ipc_fd)


if __name__ == "__main__":
    main()
