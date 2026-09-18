"""Cloudera AI Workbench Model para Nemotron 3 Nano BF16 en 1 x H100 80 GB."""

from __future__ import annotations

import os
import re
import threading
from typing import Any

from vllm import LLM, SamplingParams

try:
    import cml.models_v1 as models
except ImportError:  # Permite pruebas fuera de Cloudera.
    class _Models:
        @staticmethod
        def cml_model(function):
            return function

    models = _Models()


MODEL_ID = os.getenv(
    "LLM_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)
MAX_MODEL_LEN = int(os.getenv("LLM_MAX_MODEL_LEN", "262144"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))

_engine = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype=os.getenv("LLM_DTYPE", "bfloat16"),
    kv_cache_dtype=os.getenv("LLM_KV_CACHE_DTYPE", "auto"),
    gpu_memory_utilization=float(os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.92")),
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=int(os.getenv("LLM_MAX_NUM_SEQS", "1")),
    max_num_batched_tokens=int(os.getenv("LLM_MAX_NUM_BATCHED_TOKENS", "8192")),
    enable_chunked_prefill=True,
    mamba_ssm_cache_dtype=os.getenv("LLM_MAMBA_SSM_CACHE_DTYPE", "float32"),
    enforce_eager=os.getenv("LLM_ENFORCE_EAGER", "false").lower() == "true",
)
_tokenizer = _engine.get_tokenizer()
_lock = threading.Lock()


def _messages(args: dict[str, Any]) -> list[dict[str, str]]:
    supplied = args.get("messages")
    if supplied is not None:
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("messages debe ser una lista no vacía")
        result = []
        for item in supplied:
            if not isinstance(item, dict) or item.get("role") not in {
                "system", "user", "assistant"
            } or not isinstance(item.get("content"), str):
                raise ValueError("Cada mensaje necesita role y content de texto")
            result.append({"role": item["role"], "content": item["content"]})
        return result

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Indique prompt o messages")
    result = []
    system_prompt = args.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        result.append({"role": "system", "content": system_prompt})
    result.append({"role": "user", "content": prompt})
    return result


def _split_reasoning(text: str) -> tuple[str | None, str]:
    if "</think>" not in text:
        return None, text.strip()
    reasoning, answer = text.split("</think>", 1)
    reasoning = re.sub(r"^\s*<think>\s*", "", reasoning).strip()
    return reasoning or None, answer.strip()


@models.cml_model
def api_wrapper(args: dict[str, Any]) -> dict[str, Any]:
    """Acepta prompt o messages y devuelve texto, reasoning y uso de tokens."""
    if not isinstance(args, dict):
        raise ValueError("La entrada debe ser un objeto JSON")

    messages = _messages(args)
    thinking = bool(args.get("thinking", True))
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    prompt_token_ids = _tokenizer.encode(prompt, add_special_tokens=False)
    available_tokens = MAX_MODEL_LEN - len(prompt_token_ids)
    if available_tokens < 1:
        raise ValueError(
            f"La conversación ocupa {len(prompt_token_ids)} tokens y supera "
            f"LLM_MAX_MODEL_LEN={MAX_MODEL_LEN}"
        )
    max_tokens = max(
        1,
        min(int(args.get("max_tokens", 1024)), MAX_OUTPUT_TOKENS, available_tokens),
    )
    sampling = SamplingParams(
        temperature=max(0.0, float(args.get("temperature", 0.6))),
        top_p=min(1.0, max(0.01, float(args.get("top_p", 0.95)))),
        max_tokens=max_tokens,
        repetition_penalty=max(1.0, float(args.get("repetition_penalty", 1.0))),
        seed=int(args.get("seed", 0)),
    )
    with _lock:
        result = _engine.generate([prompt], sampling, use_tqdm=False)[0]

    candidate = result.outputs[0]
    reasoning, answer = _split_reasoning(candidate.text)
    return {
        "model": MODEL_ID,
        "context_window": MAX_MODEL_LEN,
        "text": answer,
        "reasoning": reasoning,
        "finish_reason": candidate.finish_reason,
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(candidate.token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(candidate.token_ids),
            "remaining_context_tokens": max(
                0,
                MAX_MODEL_LEN - len(result.prompt_token_ids) - len(candidate.token_ids),
            ),
        },
    }
