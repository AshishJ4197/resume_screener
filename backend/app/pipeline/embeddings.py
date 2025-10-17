# app/pipeline/embeddings.py

from __future__ import annotations
import os, time, math
from typing import List

# LangChain / Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ------------ API key wiring ------------
# Accept any of these env vars:
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_APIKEY")
)

if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    raise RuntimeError(
        "Missing Gemini key. Set GEMINI_API_KEY (or GOOGLE_API_KEY / GEMINI_APIKEY) "
        "in your environment before starting the server."
    )

# Make sure downstream libs see the key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
# Avoid ADC confusion
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# ------------ Construct LLM + Embeddings ------------
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=GEMINI_API_KEY,
)

_embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)

# ------------ Simple wrappers used by pipeline ------------
def embed_texts(texts: List[str], batch_size: int = 64, sleep_between: float = 0.0) -> List[List[float]]:
    """Embed a list of texts; batched to reduce API calls."""
    if not texts:
        return []
    vecs: List[List[float]] = []
    for i in range(0, len(texts), max(1, batch_size)):
        chunk = texts[i : i + batch_size]
        vecs.extend(_embedding.embed_documents(chunk))
        if sleep_between:
            time.sleep(sleep_between)
    return vecs

def embed_query(text: str) -> List[float]:
    return _embedding.embed_query(text or "")

# Public handle bag consumed by ingest()
HANDLES = {
    "llm": _llm,                   # used via LangChain .invoke() inside prompts
    "embed_texts": embed_texts,    # callable
    "embed_query": embed_query,    # callable
}
