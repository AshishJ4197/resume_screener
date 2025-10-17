# backend/app/db/crud.py
"""
Tiny CRUD helpers for CandidateMatch.
Usage (with context manager):
    from .session import session_scope, ensure_tables
    ensure_tables()
    with session_scope() as s:
        save_candidate_match(s, payload)

Or with FastAPI dependency `get_session()` if you wire it in a router.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from .models import CandidateMatch
from .session import ensure_tables


# ----------------- Inserts / Upserts -----------------

def save_candidate_match(session: Session, payload: Dict[str, Any]) -> CandidateMatch:
    """
    Insert a minimal candidate match row.
    Not a strict upsert; if you want to dedupe by (run_id), call get_match_by_run_id first.
    Expected keys in payload (all optional except run_id):
      run_id, name, email, phone, linkedin, score, eligible
    """
    ensure_tables()

    row = CandidateMatch(
        run_id=str(payload.get("run_id") or ""),
        name=payload.get("name"),
        email=payload.get("email"),
        phone=payload.get("phone"),
        linkedin=payload.get("linkedin"),
        score=int(payload.get("score") or 0),
        eligible=bool(payload.get("eligible")),
    )
    session.add(row)
    # commit is handled by caller (session_scope) or FastAPI dep
    return row


# ----------------- Queries -----------------

def get_match_by_run_id(session: Session, run_id: str) -> Optional[CandidateMatch]:
    q = select(CandidateMatch).where(CandidateMatch.run_id == run_id).limit(1)
    return session.execute(q).scalars().first()


def list_recent_matches(session: Session, limit: int = 50) -> Iterable[CandidateMatch]:
    q = (
        select(CandidateMatch)
        .order_by(desc(CandidateMatch.created_at))
        .limit(max(1, min(limit, 500)))
    )
    return session.execute(q).scalars().all()
