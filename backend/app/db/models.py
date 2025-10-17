# backend/app/db/models.py
"""
SQLAlchemy ORM models.
Currently only one table is needed for the ATS MVP:
- CandidateMatch: minimal, idempotent record per run with top-line fields that
  the frontend needs to hydrate dashboards/lists quickly.
"""

from __future__ import annotations

from sqlalchemy import String, Integer, Boolean, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


class CandidateMatch(Base):
    __tablename__ = "candidate_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # run/artifacts
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False, unique=False)

    # minimal PII (nullable to keep fail-open ingestion)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    email: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    linkedin: Mapped[str | None] = mapped_column(Text, nullable=True)

    # outcomes
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # server timestamp
    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<CandidateMatch id={self.id} run_id={self.run_id} score={self.score} eligible={self.eligible}>"
