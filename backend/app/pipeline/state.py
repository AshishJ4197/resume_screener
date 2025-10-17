# backend/app/pipeline/state.py
"""
Cells 1 & 2 → refactor:
- Gemini LLM + Embeddings setup (no outbound call on import)
- Options, prompts, and STATE constructor

This module is imported by other pipeline steps and by the orchestrator.
"""

from __future__ import annotations

# ==== Cell 1: Imports + Setup (Gemini-only: LLM + Embeddings) ====
# Std libs
import os, math, uuid, json, time
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict

# LangChain utilities (kept for parity, some used in later steps)
# NOTE: use community namespaces to avoid deprecation warnings
from langchain_community.vectorstores import FAISS  # noqa: F401  (not used directly but retained)
from langchain_community.document_loaders import PyMuPDFLoader  # noqa: F401
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: F401

# Gemini via LangChain (LLM + Embeddings)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ---------- API key (Gemini) ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_APIKEY")
if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    raise RuntimeError("Missing Gemini key. Set GEMINI_API_KEY in your environment before running.")

# Make sure downstream libs find the key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
# Avoid ADC fallback confusion
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# ---------- Model handles ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=GEMINI_API_KEY,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)

# ---------- Utilities (no outbound calls here) ----------
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
    Keep this function for later cells; do NOT call it at import time.
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

print("[ok] Gemini LLM ready (gemini-2.0-flash)")
print("[ok] Gemini embeddings ready (models/text-embedding-004)")
print("[ok] Cell 1 complete — using Gemini for both reasoning and semantic analysis (no local models).")

# ==== Cell 2: State + Options + Prompts ====
import re, json, uuid  # noqa: E402

# ---------- Configuration / thresholds ----------
class RunOptions(TypedDict):
    # chunking + retrieval
    chunk_tokens: int
    chunk_overlap: float
    faiss_topk: int
    extract_max_chunks: int
    hybrid_alpha: float
    mmr_lambda: float

    # API budgets
    llm_budget_calls: int
    embed_batch_size: int

    # semantic thresholds (embeddings)
    strong_sim: float
    partial_sim: float
    canon_threshold: float

    # complexity / transferability
    max_projects_for_complexity: int
    complexity_weight_cap: float

    # scoring weights (0–100 total)
    weights: Dict[str, float]
    gate_hard_cap: int

    # eligibility + UX
    eligibility_threshold: int
    sections_to_emit: List[str]

    # misc
    recency_months_ideal: int
    language: str
    keep_artifacts: bool

DEFAULT_OPTIONS: RunOptions = {
    # retrieval
    "chunk_tokens": 900,
    "chunk_overlap": 0.15,
    "faiss_topk": 8,
    "extract_max_chunks": 4,
    "hybrid_alpha": 0.6,
    "mmr_lambda": 0.35,

    # budgets
    "llm_budget_calls": 7,
    "embed_batch_size": 64,

    # semantic thresholds
    "strong_sim": 0.78,
    "partial_sim": 0.65,
    "canon_threshold": 0.82,

    # complexity / transferability
    "max_projects_for_complexity": 4,
    "complexity_weight_cap": 0.12,

    # scoring weights (sum to ~100)
    "weights": {
        "must_have_coverage": 22.0,
        "required_coverage": 16.0,
        "preferred_coverage": 6.0,
        "role_alignment": 16.0,
        "project_alignment": 10.0,
        "evidence_depth": 6.0,
        "seniority_fit": 8.0,
        "responsibility_overlap": 6.0,
        "transferability_bonus": 10.0,
    },
    "gate_hard_cap": 59,

    # eligibility + UX
    "eligibility_threshold": 50,
    "sections_to_emit": [
        "present_against_jd",
        "missing_against_jd",
        "extra_strengths"
    ],

    # misc
    "recency_months_ideal": 6,
    "language": "en",
    "keep_artifacts": True,
}

# ---------- Pipeline STATE ----------
class PipelineState(TypedDict, total=False):
    run_id: str
    options: RunOptions
    inputs: Dict[str, Optional[str]]
    raw: Dict[str, Optional[str]]
    provenance: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    faiss: Dict[str, Any]
    contacts: Dict[str, Any]
    high_level: Dict[str, Any]
    education: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    certs: List[Dict[str, Any]]
    awards: List[Dict[str, Any]]
    locations: List[str]
    jd_snapshot: Dict[str, Any]
    canon: Dict[str, Any]
    jd_alignment: Dict[str, Any]
    coverage: Dict[str, Any]
    gates: Dict[str, Any]
    complexity: Dict[str, Any]
    final: Dict[str, Any]
    artifacts: Dict[str, Any]
    audit: List[str]
    _llm_calls: int
    flags: Dict[str, Any]

def new_state(
    resume_file: Optional[str],
    resume_text: Optional[str],
    jd_file: Optional[str],
    jd_text: Optional[str],
    options: RunOptions = DEFAULT_OPTIONS,
) -> PipelineState:
    run_id = uuid.uuid4().hex[:12]
    base_dir = Path(f"./tmp/{run_id}")
    paths = {
        "base_dir": str(base_dir),
        "chunks_json": str(base_dir / "chunks.json"),
        "entities_by_chunk_json": str(base_dir / "entities_by_chunk.json"),
        "final_json": str(base_dir / "final.json"),
        "faiss_dir": str(base_dir / "faiss"),
        "jd_snapshot_json": str(base_dir / "jd_snapshot.json"),
        "complexity_json": str(base_dir / "complexity.json"),
    }
    st: PipelineState = {
        "run_id": run_id,
        "options": options,
        "inputs": {
            "resume_file": resume_file, "resume_text": resume_text,
            "jd_file": jd_file, "jd_text": jd_text,
        },
        "raw": {"resume_text": None, "jd_text": None},
        "provenance": {"chunks": [], "hybrid_retrieval": [], "mmr_selected": []},
        "chunks": [],
        "faiss": {"index_path": None, "topk_ids": []},
        "contacts": {"name": None, "email": None, "phone": None,
                     "links": {"linkedin": None, "github": None, "portfolio": None, "website": None}},
        "high_level": {"summary": None, "location": None, "years_experience": None},
        "education": [], "timeline": [], "projects": [],
        "skills": [], "certs": [], "awards": [], "locations": [],
        "jd_snapshot": {
            "title": None,
            "must_haves": [],
            "required": [],
            "preferred": [],
            "responsibilities": [],
            "hard_gates": {
                "degree_required": False,
                "min_years": None,
                "license": [],
                "work_auth": None,
                "clearance": None,
                "location_mode": None,
                "onsite_city": None,
                "shift": None,
                "travel": None,
            },
            "evidence": {"must": {}, "req": {}, "pref": {}, "resp": {}},
            "conf": {"must": {}, "req": {}, "pref": {}, "resp": {}},
        },
        "canon": {
            "skill_alias": {},
            "normalized_skills": [],
            "normalized_required": [],
            "normalized_preferred": [],
        },
        "jd_alignment": {
            "must_have": [],
            "required": [],
            "preferred": [],
            "responsibilities": {"coverage": 0.0, "count": 0}
        },
        "coverage": {},
        "gates": {"failed": [], "notes": []},
        "complexity": {"scored": [], "bonus": 0.0},
        "final": {},
        "artifacts": {"base_dir": paths["base_dir"], "paths": paths},
        "audit": [],
        "_llm_calls": 0,
        "flags": {"is_resume": False},
    }
    if options.get("keep_artifacts", True):
        base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ok] STATE initialized — run_id={st['run_id']} → artifacts: {paths['base_dir']}")
    return st

# ---------- Prompt helpers ----------
def _escape_braces_keep_vars(template: str, keep_vars: List[str]) -> str:
    esc = template.replace("{", "{{").replace("}", "}}")
    for v in keep_vars:
        esc = esc.replace("{{" + v + "}}", "{" + v + "}")
    return esc

PROMPTS: Dict[str, str] = {}

# 1) JD snapshot + hard gates
PROMPTS["jd_snapshot_and_gates"] = _escape_braces_keep_vars(r"""
You are a **strict but fair** parser for an Applicant Tracking System. Read the Job Description (any domain)
and convert it into a compact, **evidence-bound** JSON snapshot used for automated matching.

General rules:
- Tokens must be **atomic capabilities/credentials**, in **lowercase** (e.g., "object oriented programming", "excel", "b2b sales", "sterile technique", "lean six sigma").
- Include items that are **explicit** or **clearly implied** by phrasing; do **NOT** list examples not present.
- Do **NOT** expand families (no listing "gcc/clang" if only "compilers" is mentioned).
- For every token/phrase/gate, supply a short **evidence** snippet (≤160 chars). If unclear, use "".
- Include a **confidence** score (0.0–1.0) reflecting how certain the token/phrase was required or implied.

Return STRICT JSON only:
{
  "title": "<short role title>",
  "must_haves": [{"token": "", "evidence": "", "conf": 0.0}],
  "required": [{"token": "", "evidence": "", "conf": 0.0}],
  "preferred": [{"token": "", "evidence": "", "conf": 0.0}],
  "responsibilities": [{"phrase": "", "evidence": "", "conf": 0.0}],
  "hard_gates": {
    "degree_required": { "value": true|false, "evidence": "" },
    "min_years": { "value": null | 0, "evidence": "" },
    "license": [{ "token": "", "evidence": "" }],
    "work_auth": { "value": null | "us citizen|eu work permit|...", "evidence": "" },
    "clearance": { "value": null | "active secret|public trust|...", "evidence": "" },
    "location_mode": { "value": null | "onsite|hybrid|remote", "evidence": "" },
    "onsite_city": { "value": null | "<city or region>", "evidence": "" },
    "shift": { "value": null | "night|rotational|weekend|...", "evidence": "" },
    "travel": { "value": null | "<% or phrase>", "evidence": "" }
  }
}

Guidance:
- Prefer role-agnostic phrasing; avoid technology bias. Examples across domains: "inventory optimization", "sterile technique", "crm", "lead generation", "risk analysis", "six sigma", "autocad", "wound care".
- Keep lists **minimal & atomic**; never merge two different tokens into one.

JD:
---
{jd_text}
---
""", ["jd_text"])

# 2) Chunk extractor
PROMPTS["extract_all_from_chunk"] = _escape_braces_keep_vars(r"""
Extract **only** what appears in THIS chunk of a resume. Do not infer content from outside this chunk.

Return STRICT JSON only:
{
  "contacts": { "name": null, "email": null, "phone": null, "links": ["urls"] },
  "education": [
    { "degree": "", "field": "", "institution": "", "start": null, "end": null, "location": null, "evidence": "" }
  ],
  "experience": [
    { "title": "", "company": "", "location": null, "start": null, "end": null,
      "highlights": ["impact/achievements, keep concise; include metrics if any"], "evidence": "" }
  ],
  "projects": [
    { "name": "", "tech": ["as written tokens"], "impact": null, "duration": null, "role": null, "links": ["urls"], "evidence": "" }
  ],
  "skills": ["as written tokens"],
  "certifications": ["as written short tokens"],
  "awards": ["as written"],
  "locations": ["city/state/country names mentioned"]
}

Chunk ID: {chunk_id}
Chunk:
---
{chunk_text}
---
""", ["chunk_id","chunk_text"])

# 3) Consolidator
PROMPTS["consolidate_resume_json"] = _escape_braces_keep_vars(r"""
You are merging multiple chunk-level JSON extractions of one resume.

Rules:
- **Deduplicate** across chunks.
- **Unify obvious synonyms** ONLY if each synonym has some evidence (e.g., "o.o.p", "oop", "object oriented programming").
  Use a single **canonical** string; keep all synonyms in **aliases**.
- **Keep evidence arrays** for each item (gather from input snippets).
- Normalize dates to "YYYY-MM" when possible; if ambiguous, retain the raw string.
- Preserve short, high-signal highlights (metrics, scale, scope).

Return STRICT JSON only:
{
  "contacts": { "name": null, "email": null, "phone": null, "links": ["urls"] },
  "education": [ { "degree": "", "field": "", "institution": "", "start": null, "end": null, "location": null, "evidence": ["..."] } ],
  "experience": [ { "title": "", "company": "", "location": null, "start": null, "end": null, "highlights": ["..."], "evidence": ["..."] } ],
  "projects": [ { "name": "", "tech": ["tokens"], "impact": null, "duration": null, "role": null, "links": ["urls"], "evidence": ["..."] } ],
  "skills": [ { "canonical": "", "aliases": ["..."], "evidence": ["..."] } ],
  "certifications": [ { "canonical": "", "aliases": ["..."], "evidence": ["..."] } ],
  "awards": [ { "canonical": "", "evidence": ["..."] } ],
  "locations": [ { "canonical": "", "evidence": ["..."] } ]
}

Inputs:
---
{chunk_json}
---
""", ["chunk_json"])

# 4) Project/Internship complexity & transferability scorer
PROMPTS["score_project_complexity_batch"] = _escape_braces_keep_vars(r"""
Evaluate each PROJECT/INTERNSHIP against the Job Description **in principle** (domain-agnostic).
Score two axes from 0.0–1.0, and provide a concise rationale:

- "complexity": problem difficulty, scale (#users, data/transactions), technical or domain depth,
  constraints (latency, safety, compliance), duration/tenure, ownership, novelty, integration breadth.
- "transferability": how well the **capabilities and patterns** demonstrated can apply to the JD role, even if the domain differs
  (e.g., experimentation, stakeholder comms, process automation, safety standards, optimization, data handling,
   customer workflows, regulations, reliability, cost control, quality metrics, design-to-constraints).

Important:
- Do NOT penalize domain mismatch if fundamentals/approaches are transferable.
- Use the project text only (no invention). Prefer metrics when present.

Return STRICT JSON only:
{
  "items": [
    { "name": "", "complexity": 0.0, "transferability": 0.0, "rationale": "1–2 lines grounded in the text" }
  ]
}

Context:
JD_title: {jd_title}
JD_tokens: {jd_tokens}

PROJECTS_JSON:
---
{projects_json}
---
""", ["jd_title","jd_tokens","projects_json"])

print("[ok] Prompts ready (detailed, domain-agnostic): jd_snapshot_and_gates, extract_all_from_chunk, consolidate_resume_json, score_project_complexity_batch")
print("[ok] Cell 2 updated — context-aware, not strict, supports project complexity/transferability, eligibility ≥ 50.")
