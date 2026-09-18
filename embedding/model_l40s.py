"""Endpoint de embeddings E5 multilingüe optimizado para L40S."""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

try:
    import cml.models_v1 as models
except ImportError:
    class _Models:
        @staticmethod
        def cml_model(function):
            return function

    models = _Models()


MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
MAX_BATCH_SIZE = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "32"))
_model = SentenceTransformer(
    MODEL_ID,
    device=os.getenv("EMBEDDING_DEVICE", "cuda"),
    model_kwargs={"torch_dtype": torch.float16},
)
_model.max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "8192"))
_lock = threading.Lock()


def _texts(args: dict[str, Any]) -> list[str]:
    value = args.get("input", args.get("texts"))
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ValueError("input debe ser texto o una lista de textos")
    if not values or len(values) > MAX_BATCH_SIZE:
        raise ValueError(f"El lote debe contener entre 1 y {MAX_BATCH_SIZE} textos")

    input_type = str(args.get("input_type", "query")).lower()
    if input_type not in {"query", "passage", "raw"}:
        raise ValueError("input_type debe ser query, passage o raw")
    # BGE-M3 no necesita prefijos distintos para consulta y documento.
    return values


@models.cml_model
def api_wrapper(args: dict[str, Any]) -> dict[str, Any]:
    """Devuelve embeddings normalizados con formato similar a OpenAI."""
    if not isinstance(args, dict):
        raise ValueError("La entrada debe ser un objeto JSON")
    texts = _texts(args)
    normalize = bool(args.get("normalize", True))
    batch_size = min(len(texts), int(args.get("batch_size", 8)), MAX_BATCH_SIZE)
    with _lock:
        vectors = _model.encode(
            texts,
            batch_size=max(1, batch_size),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return {
        "object": "list",
        "model": MODEL_ID,
        "dimension": int(vectors.shape[1]),
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
    }
