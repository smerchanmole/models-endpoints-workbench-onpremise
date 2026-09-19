"""Qwen3.8-27B-FP8 en una A100 80 GB como Cloudera Workbench Model."""

from __future__ import annotations

import os
import re
import site
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# El build instala Qwen/vLLM en un venv aislado. Workbench ejecuta el contenido
# del fichero como una celda Jupyter (sin definir __file__), mientras que Python
# normal sí lo importa como módulo. Se admiten ambos modos y también un Model
# Root Directory situado directamente en qwen3_8.
_model_file = globals().get("__file__")
_venv_candidates = []
if _model_file:
    _venv_candidates.append(Path(str(_model_file)).resolve().parent / ".venv")
_venv_candidates.extend(
    [
        Path.cwd() / "qwen3_8" / ".venv",
        Path.cwd() / ".venv",
    ]
)
_VENV_DIR = next(
    (candidate for candidate in _venv_candidates if candidate.is_dir()),
    _venv_candidates[0],
)
_VENV_SITE_PACKAGES = (
    _VENV_DIR
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if not _VENV_SITE_PACKAGES.is_dir():
    raise RuntimeError(
        f"No existe el entorno virtual Qwen esperado en {_VENV_SITE_PACKAGES}. "
        f"Rutas comprobadas: {', '.join(map(str, _venv_candidates))}. "
        "Ejecute cdsw-build.sh con MODEL_FAMILY=qwen3_8 y GPU_TYPE=a100."
    )
site.addsitedir(str(_VENV_SITE_PACKAGES))
sys.path.remove(str(_VENV_SITE_PACKAGES))
sys.path.insert(0, str(_VENV_SITE_PACKAGES))
os.environ.setdefault("VIRTUAL_ENV", str(_VENV_DIR))
os.environ["PATH"] = f"{_VENV_DIR / 'bin'}:{os.environ.get('PATH', '')}"

# Deben establecerse antes de importar vLLM.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.registry import ModelRegistry

# El registro lazy de vLLM inspecciona la arquitectura en un subprocess. PBJ
# oculta el stderr de ese proceso y acaba devolviendo únicamente el mensaje
# genérico "failed to be inspected". La clase ya se puede importar desde el
# venv, así que se registra directamente: vLLM obtiene sus capacidades en este
# proceso y no necesita ejecutar el inspector externo.
ModelRegistry.register_model(
    "Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration
)

try:
    import cml.models_v1 as models
except ImportError:  # Permite validación local sin un Runtime de Cloudera.
    class _Models:
        @staticmethod
        def cml_model(function):
            return function

    models = _Models()


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")
SERVED_MODEL_NAME = os.getenv("QWEN_SERVED_MODEL_NAME", "qwen3.8-27b-fp8")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} debe ser booleano")
    return value in {"1", "true", "yes", "on"}


TENSOR_PARALLEL_SIZE = _env_int("VLLM_TENSOR_PARALLEL_SIZE", 1, 1, 8)
MAX_MODEL_LEN = _env_int("VLLM_MAX_MODEL_LEN", 262144, 2048, 262144)
GPU_MEMORY_UTILIZATION = _env_float(
    "VLLM_GPU_MEMORY_UTILIZATION", 0.90, 0.50, 0.95
)
MAX_NUM_SEQS = _env_int("VLLM_MAX_NUM_SEQS", 1, 1, 64)
MAX_NUM_BATCHED_TOKENS = _env_int(
    "VLLM_MAX_NUM_BATCHED_TOKENS", 8192, 2048, 131072
)
MAX_OUTPUT_TOKENS = _env_int("QWEN_MAX_OUTPUT_TOKENS", 512, 1, 8192)
CPU_OFFLOAD_GB = _env_float("VLLM_CPU_OFFLOAD_GB", 0.0, 0.0, 128.0)
KV_CACHE_DTYPE = os.getenv("VLLM_KV_CACHE_DTYPE", "fp8")
ENFORCE_EAGER = _env_bool("VLLM_ENFORCE_EAGER", True)
ATTENTION_BACKEND = os.getenv("QWEN_ATTENTION_BACKEND", "TRITON_ATTN")
MAX_IMAGES = _env_int("QWEN_MAX_IMAGES_PER_PROMPT", 1, 0, 4)

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
            f"free={shm.f_bavail * shm.f_frsize / 1024**2:.0f} MiB"
        )
    except OSError as exc:
        print(f"SHM_DIAGNOSTIC unavailable={exc!r}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8 requiere una GPU NVIDIA visible en el deployment")
    gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        "GPU_DIAGNOSTIC "
        f"name={torch.cuda.get_device_name(0)} memory={gpu_total_gib:.1f} GiB "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda}"
    )
    if gpu_total_gib < 70:
        raise RuntimeError(
            "Este perfil requiere una A100 de 80 GB completa. Para una A100 de "
            "40 GB hace falta un perfil de contexto reducido validado aparte."
        )

    print(
        "VLLM_CONFIG "
        f"model={MODEL_ID} tp={TENSOR_PARALLEL_SIZE} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS} "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION} "
        f"kv_cache_dtype={KV_CACHE_DTYPE} attention_backend={ATTENTION_BACKEND} "
        f"enforce_eager={ENFORCE_EAGER}"
    )

    return LLM(
        model=MODEL_ID,
        served_model_name=SERVED_MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        kv_cache_dtype=KV_CACHE_DTYPE,
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


# La réplica no queda Ready hasta que pesos y kernels están cargados.
ENGINE = _load_engine()
_GENERATION_LOCK = threading.Lock()


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


@models.cml_model
def predict(args: dict[str, Any]) -> dict[str, Any]:
    """Genera una respuesta chat no streaming desde un objeto JSON."""
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
        max_tokens=int(
            _number(args, "max_tokens", 128, 1, MAX_OUTPUT_TOKENS)
        ),
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
        result = ENGINE.chat(
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


api_wrapper = predict
