# app/pipeline/ingest.py

from __future__ import annotations
from typing import Dict, Any
from app.pipeline.embeddings import HANDLES  # provides: llm, embed_texts, embed_query

def ingest(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal ingest: register model/embedding handles in STATE.

    Why both keys?
      - Some modules read STATE["handles"]["..."]
      - Others (older code) read STATE["_handles"]["..."]
    We set both to the same dict so you don't have to refactor everywhere.
    """
    state.setdefault("handles", {})
    state.setdefault("_handles", {})

    to_register = {
        "llm": HANDLES.get("llm"),
        "embed_texts": HANDLES.get("embed_texts"),
        "embed_query": HANDLES.get("embed_query"),
    }

    # Validate to avoid 'NoneType is not callable'
    for k, v in to_register.items():
        if v is None:
            raise RuntimeError(f"ingest: handle '{k}' is None. Check app/pipeline/embeddings.py setup.")

    state["handles"].update(to_register)
    state["_handles"].update(to_register)
    return state

# Alias expected by main/service/orchestrator imports
def run_ingest(state: Dict[str, Any]) -> Dict[str, Any]:
    return ingest(state)
