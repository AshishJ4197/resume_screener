# backend/app/core/llm.py
"""
Gemini LLM handle (LangChain wrapper).
- Uses ChatGoogleGenerativeAI with model "gemini-2.0-flash", temperature=0
- Reads API key via core.config.get_gemini_api_key()
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from .config import get_gemini_api_key

# Lazily constructed singleton-ish LLM (module import creates it once)
_GEMINI_KEY = get_gemini_api_key()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=_GEMINI_KEY,
)

__all__ = ["llm"]
