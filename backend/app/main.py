# backend/main.py
"""
FastAPI entrypoint for Resume ↔ JD analyzer.
- Local friendly (uvicorn) and Vercel-ready (exports `app`).
- Accepts PDF/DOCX/TXT uploads or raw text for both Resume and JD.
- Calls the modular pipeline (Cells 1–5 refactored) via app.pipeline.service.run_full_pipeline
- Returns: score, eligibility, 3 narratives, breakdown, and minimal PII (name/email/phone/linkedin).
- Optional Postgres insert when DATABASE_URL is set (idempotent table bootstrap).

Run locally:
  uvicorn main:app --reload --port 8000

Env (.env):
  GEMINI_API_KEY=...
  DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname   # optional
  ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000     # optional
"""

from __future__ import annotations
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- load env early ---
load_dotenv(override=False)

# --- CORS ---
_default_origins = {
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
}
extra_origins = set(
    [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "").split(",") if o.strip()]
)
origins = list(_default_origins | extra_origins)

app = FastAPI(title="ATS Analyzer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB (optional) -----------------------------------------------------------
DB_ENABLED = False
_engine = None
_table_ready = False
try:
    DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
    if DATABASE_URL:
        from sqlalchemy import create_engine, text
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        DB_ENABLED = True

        def _ensure_table():
            global _table_ready
            if _table_ready:
                return
            with _engine.begin() as conn:
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS candidate_matches (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(64) NOT NULL,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    linkedin TEXT,
                    score INTEGER NOT NULL,
                    eligible BOOLEAN NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """))
            _table_ready = True
except Exception as _e:
    # Fail-open (API still works without DB)
    DB_ENABLED = False
    _engine = None

def _db_insert_minimal(row: Dict[str, Any]) -> None:
    if not DB_ENABLED or _engine is None:
        return
    _ensure_table()
    try:
        from sqlalchemy import text
        with _engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO candidate_matches
                        (run_id, name, email, phone, linkedin, score, eligible)
                    VALUES
                        (:run_id, :name, :email, :phone, :linkedin, :score, :eligible)
                """),
                {
                    "run_id": row.get("run_id"),
                    "name": row.get("name"),
                    "email": row.get("email"),
                    "phone": row.get("phone"),
                    "linkedin": row.get("linkedin"),
                    "score": int(row.get("score") or 0),
                    "eligible": bool(row.get("eligible")),
                },
            )
    except Exception as e:
        # Don't break the request if DB insert fails
        print(f"[warn] DB insert skipped: {e}")

# --- Pipeline bridge ---------------------------------------------------------
# Expecting app/pipeline/service.py with: run_full_pipeline(resume_source, jd_source, options) -> (run_id, state_dict)
#   - resume_source: dict like {"file_path": "..."} OR {"text": "..."}
#   - jd_source: same
#   - options: dict or None
try:
    from app.pipeline.service import run_full_pipeline
except Exception as e:
    raise RuntimeError(
        "Missing pipeline modules. Ensure backend/app/pipeline/service.py "
        "and its siblings exist. Error: {}".format(e)
    )

# --- Models for request/response --------------------------------------------
class AnalyzeResponse(BaseModel):
    run_id: str
    score: int
    eligible: bool
    contacts: Dict[str, Optional[str]]
    narratives: Dict[str, str]
    breakdown: Dict[str, Any]
    strong_matches: list[str]
    skill_gaps: list[str]
    risk_flags: list[str]
    artifacts_path: Optional[str] = None

# --- Utils ------------------------------------------------------------------
def _save_upload(tmpdir: Path, file: UploadFile, fname: Optional[str] = None) -> Path:
    """Persist an UploadFile to disk inside tmpdir and return full path."""
    dst = tmpdir / (fname or file.filename or "upload.bin")
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dst

def _build_sources(
    run_dir: Path,
    resume_file: Optional[UploadFile],
    resume_text: Optional[str],
    jd_file: Optional[UploadFile],
    jd_text: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create normalized source dicts for pipeline."""
    # Resume
    if resume_file is not None and getattr(resume_file, "filename", None):
        rp = _save_upload(run_dir, resume_file)
        resume_source = {"file_path": str(rp)}
    elif (resume_text or "").strip():
        resume_source = {"text": resume_text.strip()}
    else:
        raise HTTPException(status_code=400, detail="Provide either resume_file or resume_text")

    # JD
    if jd_file is not None and getattr(jd_file, "filename", None):
        jp = _save_upload(run_dir, jd_file)
        jd_source = {"file_path": str(jp)}
    elif (jd_text or "").strip():
        jd_source = {"text": jd_text.strip()}
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_file or jd_text")

    return resume_source, jd_source

# --- Routes -----------------------------------------------------------------
@app.get("/api/v1/health")
def health():
    return {
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "db_enabled": DB_ENABLED,
        "origins": origins,
        "vercel_hint": "Exporting FastAPI app as `app` is supported.",
    }

@app.get("/api/v1/config")
def config_view():
    return {
        "db_enabled": DB_ENABLED,
        "artifacts_root": "./tmp",
        "eligibility_threshold": 50,
        "vercel_note": "On Vercel, keep heavy files in ephemeral /tmp and persist to external storage if needed.",
    }

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    resume_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    options_json: str | None = Form(default=None),
):
    """
    Analyze one Resume against one JD.
    Accepts either file uploads or raw text for each. Options JSON is optional.
    """
    # Options (safe parse)
    options: Optional[Dict[str, Any]] = None
    if options_json:
        try:
            options = json.loads(options_json)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid options_json (must be JSON)")

    # Per-run working dir under ./tmp
    tmp_root = Path("./tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    # The pipeline itself creates its own run_id directory; we just keep an ingest dir here:
    ingest_dir = tmp_root / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)

    # Normalize sources
    resume_source, jd_source = _build_sources(ingest_dir, resume_file, resume_text, jd_file, jd_text)

    # Run the pipeline (Cells 1–5 orchestrated inside service.py)
    try:
        run_id, state = run_full_pipeline(resume_source=resume_source, jd_source=jd_source, options=options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    final = state.get("final", {}) if isinstance(state, dict) else {}
    contacts = state.get("contacts", {}) if isinstance(state, dict) else {}

    # Minimal PII for DB
    name = (contacts.get("name") or None) if isinstance(contacts, dict) else None
    email = (contacts.get("email") or None)
    phone = (contacts.get("phone") or None)
    linkedin = None
    if isinstance(contacts, dict):
        ln = contacts.get("links", {}).get("linkedin") if isinstance(contacts.get("links"), dict) else None
        linkedin = ln or None

    score = int(final.get("score_100") or 0)
    eligible = bool(final.get("selected", False))
    narratives = final.get("narrative", {}) or {}
    breakdown = final.get("breakdown", {}) or {}
    strong_matches = final.get("strong_matches", []) or []
    skill_gaps = final.get("skill_gaps", []) or []
    risk_flags = final.get("risk_flags", []) or []
    artifacts_path = state.get("artifacts", {}).get("base_dir") if isinstance(state.get("artifacts"), dict) else None

    # Optional DB persist
    _db_insert_minimal({
        "run_id": run_id,
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "score": score,
        "eligible": eligible,
    })

    return AnalyzeResponse(
        run_id=run_id,
        score=score,
        eligible=eligible,
        contacts={
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
        },
        narratives={
            "present_summary": narratives.get("present_summary", ""),
            "gaps_summary": narratives.get("gaps_summary", ""),
            "bonus_summary": narratives.get("bonus_summary", ""),
        },
        breakdown=breakdown,
        strong_matches=strong_matches,
        skill_gaps=skill_gaps,
        risk_flags=risk_flags,
        artifacts_path=artifacts_path,
    )

@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str):
    """
    Fetch the final JSON artifact for a run (handy for dev / frontend hydration).
    """
    run_dir = Path(f"./tmp/{run_id}")
    final_path = run_dir / "final.json"
    if not final_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        data = json.loads(final_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read run: {e}")

# For local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
