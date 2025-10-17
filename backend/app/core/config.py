# backend/app/core/config.py
"""
Central config & environment helpers.
- Loads env (.env) early
- Exposes GEMINI_API_KEY and convenience flags
- Holds DEFAULT_OPTIONS used by the pipeline (mirrors notebook Cell 2)
"""

from __future__ import annotations

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env once for the whole app
load_dotenv(override=False)

# --- API keys ---------------------------------------------------------------

def get_gemini_api_key() -> str:
    """
    Returns the Gemini API key (same resolution order used in notebook Cell 1).
    Raises RuntimeError if missing.
    """
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_APIKEY")
        or ""
    ).strip()

    if not key:
        raise RuntimeError(
            "Missing Gemini key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment."
        )

    # Ensure downstream libs see the same key
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GEMINI_API_KEY"] = key
    # Avoid ADC confusion in server envs
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    return key


# --- Options (keep in sync with pipeline/state defaults) --------------------

DEFAULT_OPTIONS: Dict[str, Any] = {
    # retrieval
    "chunk_tokens": 900,
    "chunk_overlap": 0.15,
    "faiss_topk": 8,
    "extract_max_chunks": 4,
    "hybrid_alpha": 0.6,   # 60% semantic, 40% lexical
    "mmr_lambda": 0.35,

    # budgets
    "llm_budget_calls": 7,  # JD snapshot (1) + ≤4 chunk extracts + consolidation (1) + complexity batch (1)
    "embed_batch_size": 64,

    # semantic thresholds
    "strong_sim": 0.78,
    "partial_sim": 0.65,
    "canon_threshold": 0.82,

    # complexity / transferability
    "max_projects_for_complexity": 4,
    "complexity_weight_cap": 0.12,   # max +12 points in scoring

    # scoring weights (~100 total)
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
