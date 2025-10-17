# backend/app/db/session.py
"""
SQLAlchemy session/engine bootstrap.
- Reads DATABASE_URL from env (optional).
- Exposes: engine, SessionLocal, get_session(), ensure_tables().
- Keeps the app fail-open: if DATABASE_URL is missing or invalid, DB is disabled
  but the API continues to work.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

# --- config from env ---------------------------------------------------------

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
DB_ECHO = (os.getenv("DB_ECHO") or "false").lower() == "true"

# --- SQLAlchemy base ---------------------------------------------------------

class Base(DeclarativeBase):
    pass

# --- engine & session --------------------------------------------------------

engine = None
SessionLocal: Optional[sessionmaker[Session]] = None
DB_ENABLED = False

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, echo=DB_ECHO, pool_pre_ping=True, future=True)
        SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False, expire_on_commit=False)
        DB_ENABLED = True
    except Exception as e:
        # Fail-open: API should still run without DB
        print(f"[db] WARNING: could not initialize engine: {e}")
        engine = None
        SessionLocal = None
        DB_ENABLED = False
else:
    print("[db] DATABASE_URL not set; DB layer disabled.")

# --- helpers ----------------------------------------------------------------

@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Context manager for a DB session (if DB is enabled).
    Example:
        with session_scope() as s:
            s.add(obj)
            s.commit()
    """
    if not DB_ENABLED or SessionLocal is None:
        # Provide a clear error if someone tries to use the DB when disabled
        raise RuntimeError("Database is not enabled (missing DATABASE_URL)")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_session() -> Iterator[Session]:
    """
    FastAPI dependency style generator.
    Usage:
        @router.get(...)
        def handler(db: Session = Depends(get_session)):
            ...
    """
    if not DB_ENABLED or SessionLocal is None:
        # If DB disabled, raise is clearer than returning None to handlers
        raise RuntimeError("Database is not enabled (missing DATABASE_URL)")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def ensure_tables() -> None:
    """
    Create tables if needed. Import models lazily to avoid circulars.
    Call this once at startup or before first insert.
    """
    if not DB_ENABLED or engine is None:
        return
    # local import to prevent circular import during module import
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "DB_ENABLED",
    "session_scope",
    "get_session",
    "ensure_tables",
]
