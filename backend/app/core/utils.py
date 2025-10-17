# backend/app/core/utils.py
"""
Generic helpers used across the pipeline.

Includes:
- lightweight file readers (pdf/docx/txt) and text cleanup
- section detection/splitting for resumes
- safe JSON extraction
- date parsing / time math
- lexical tokenizer and misc string utils

These mirror the behavior used in the notebook cells so the pipeline code
(extract_align / ingest / orchestrator / score / service) can import them.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# -------- File readers (pdf/docx/txt or raw text) ---------------------------

def read_pdf_pymupdf(path: str) -> str:
    """Fast+robust PDF text via LangChain's PyMuPDFLoader."""
    from langchain.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    return "\n".join((d.page_content or "") for d in docs)

def read_docx_quick(path: str) -> str:
    """Tiny docx reader (no external deps) that extracts paragraph text."""
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: List[str] = []
    for p in root.findall(".//w:p", ns):
        txt = "".join((t.text or "") for t in p.findall(".//w:t", ns)).strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines)

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_any(path_or_text: Optional[str]) -> str:
    """Accept either a filesystem path or raw text; return text."""
    if not path_or_text:
        return ""
    p = Path(path_or_text)
    if p.exists():
        ext = p.suffix.lower()
        if ext == ".pdf":
            return read_pdf_pymupdf(str(p))
        if ext == ".docx":
            return read_docx_quick(str(p))
        return read_text_file(str(p))
    return str(path_or_text)

# -------- Text cleanup + sectioning -----------------------------------------

def drop_common_headers_footers(text: str) -> str:
    """Remove lines repeated very frequently (likely headers/footers)."""
    lines = [ln.strip() for ln in text.splitlines()]
    if not lines:
        return text
    freq: Dict[str, int] = {}
    for ln in lines:
        if not ln:
            continue
        freq[ln] = freq.get(ln, 0) + 1
    # Heuristic threshold that scales with total lines & page count hints
    threshold = max(3, int(0.5 * (len(lines) / max(1, (text.count("\f") or 1)))))
    cleaned = [ln for ln in lines if not ln or freq.get(ln, 0) < threshold]
    return "\n".join(cleaned)

SECTION_PAT = re.compile(
    r"^\s*(summary|profile|objective|experience|work experience|employment|projects|internships?|education|skills|certifications?|awards?|publications?|achievements?|volunteer|activities|extracurricular|research|patents?)\s*$",
    re.I,
)

def smart_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text into (section_title, section_text) pairs using common resume headings.
    Fallback: one unlabeled section with full text.
    """
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "unlabeled"
    bucket: List[str] = []
    for ln in lines:
        if SECTION_PAT.match(ln.strip()):
            if bucket:
                sections.append((current_title.lower(), bucket))
                bucket = []
            current_title = ln.strip().lower()
        else:
            bucket.append(ln)
    if bucket:
        sections.append((current_title.lower(), bucket))
    out: List[Tuple[str, str]] = []
    for t, b in sections:
        body = "\n".join(b).strip()
        if body:
            out.append((t, body))
    return out

def approx_token_len(s: str) -> int:
    """Rough proxy: 1 token ≈ 4 characters."""
    return max(1, math.ceil(len(s) / 4))

# -------- Lexical tokenizer --------------------------------------------------

def lex_keywords(s: str) -> List[str]:
    """
    Lightweight lexical tokenizer:
    - lowercase
    - keep alnum and a few job-critical symbols (+, #, ., /, -) for tokens like c++, c#, .net
    - drop stop-words implicitly by min length
    """
    s = re.sub(r"[^A-Za-z0-9+#.\-/ ]+", " ", s.lower())
    toks = [t for t in s.split() if len(t) >= 2]
    seen: set[str] = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# -------- JSON + strings -----------------------------------------------------

def json_loose(s: str) -> Any:
    """
    Parse a possibly noisy LLM response and return the first valid JSON object/array.
    """
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

def clip(s: Optional[str], n: int = 1200) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n]

# -------- Time + dates -------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def parse_date_soft(s: Optional[str]) -> Optional[datetime]:
    """Best-effort parse for resume dates; supports 'Present/Current'."""
    if not s:
        return None
    tl = str(s).strip().lower()
    if any(k in tl for k in ["present", "current", "now"]):
        return now_utc()
    # Prefer python-dateutil if present
    try:
        from dateutil import parser as _dp  # type: ignore
        dt = _dp.parse(tl, default=datetime(2000, 1, 1), fuzzy=True)
        if 1900 <= dt.year <= 2100:
            return datetime(dt.year, dt.month if dt.month else 1, 1)
    except Exception:
        pass
    m = re.search(r"(20\d{2}|19\d{2})", tl)
    if m:
        return datetime(int(m.group(1)), 1, 1)
    return None

def months_between(a: Optional[datetime], b: Optional[datetime]) -> int:
    if not a or not b:
        return 0
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


__all__ = [
    # readers
    "read_any", "read_pdf_pymupdf", "read_docx_quick", "read_text_file",
    # cleanup/sections
    "drop_common_headers_footers", "smart_sections", "SECTION_PAT", "approx_token_len",
    # lexical
    "lex_keywords",
    # json/string utils
    "json_loose", "clip",
    # time/dates
    "now_utc", "parse_date_soft", "months_between",
]
