# backend/app/core/embeddings.py
"""
Gemini embeddings + tiny helpers.
- Uses GoogleGenerativeAIEmbeddings ("models/text-embedding-004")
- Exposes: embedding_model, cosine_sim, embed_texts, embed_query
- Mirrors Cell 1 utilities from the notebook
"""

from __future__ import annotations

import math
import time
from typing import List, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import get_gemini_api_key

# ---- Model handle -----------------------------------------------------------

_embedding_key = get_gemini_api_key()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=_embedding_key,
)

# ---- Helpers (match Cell 1) ------------------------------------------------

def cosine_sim(u: List[float], v: List[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    if not u or not v:
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def _batch(iterable: List[Any], size: int) -> List[List[Any]]:
    """Yield successive batches of given size from a list."""
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def embed_texts(texts: List[str], batch_size: int = 64, sleep_between: float = 0.0) -> List[List[float]]:
    """
    Embed a list of texts using Gemini embeddings with optional batching.
    Identical behavior to the notebook's Cell 1 function.
    """
    if not texts:
        return []
    vecs: List[List[float]] = []
    for chunk in _batch(texts, batch_size):
        chunk_vecs = embedding_model.embed_documents(chunk)
        vecs.extend(chunk_vecs)
        if sleep_between:
            time.sleep(sleep_between)
    return vecs


def embed_query(text: str) -> List[float]:
    """Single-text embedding (used for queries like JD vectors)."""
    return embedding_model.embed_query(text or "")


__all__ = [
    "embedding_model",
    "cosine_sim",
    "embed_texts",
    "embed_query",
]
