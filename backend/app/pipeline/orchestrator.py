# backend/app/pipeline/orchestrator.py
"""
Glue for the ATS pipeline.

Stages:
  1) ingest.run_ingest       (Cell 3 refactor) → load, chunk, embed, hybrid+MMR, resume sanity
  2) extract_align.run_extract_align  (Cell 4 refactor)
  3) score.run_score         (Cell 5 refactor)

Public entry:
  run_full_pipeline(resume_source, jd_source, options=None) -> (run_id, state)

`resume_source` / `jd_source` must be dicts in one of these shapes:
  {"file_path": "/abs/or/rel/path/to/file.pdf"}    # pdf/docx/txt supported
  {"text": "raw text ..."}                         # raw string

This module *only* orchestrates; each stage persists its own artifacts under ./tmp/<run_id>.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

from .state import new_state, DEFAULT_OPTIONS, PipelineState
from .ingest import run_ingest
from .extract_align import run_extract_align
from .score import run_score


def _merge_options(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Overlay user options on top of DEFAULT_OPTIONS (shallow)."""
    if not user:
        return dict(DEFAULT_OPTIONS)
    base = dict(DEFAULT_OPTIONS)
    for k, v in user.items():
        # allow nested weights override
        if k == "weights" and isinstance(v, dict):
            w = dict(base.get("weights", {}))
            w.update(v)
            base["weights"] = w
        else:
            base[k] = v
    return base


def run_full_pipeline(
    resume_source: Dict[str, Any],
    jd_source: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, PipelineState]:
    """
    Main orchestrator used by FastAPI.

    Returns:
        (run_id, state) — where `state["final"]` contains score, eligibility, narratives, etc.
    """
    # Normalize inputs for state bootstrap using the Cell-2-style new_state signature
    resume_file = (resume_source.get("file_path") if isinstance(resume_source, dict) else None)
    resume_text = (resume_source.get("text") if isinstance(resume_source, dict) else None)
    jd_file     = (jd_source.get("file_path") if isinstance(jd_source, dict) else None)
    jd_text     = (jd_source.get("text") if isinstance(jd_source, dict) else None)

    merged_opts = _merge_options(options)

    state = new_state(
        resume_file=resume_file,
        resume_text=resume_text,
        jd_file=jd_file,
        jd_text=jd_text,
        options=merged_opts,
    )

    # Stage 1: ingest (load → chunk → embeddings → hybrid+MMR → resume sanity)
    state = run_ingest(state)

    # If ingest classified the doc as not-a-resume, it already wrote a final JSON.
    if state.get("final") and not state.get("flags", {}).get("is_resume", True):
        return state["run_id"], state

    # Stage 2: extraction + alignment + complexity bonus
    state = run_extract_align(state)

    # Stage 3: scoring + narratives + persist
    state = run_score(state)

    return state["run_id"], state
