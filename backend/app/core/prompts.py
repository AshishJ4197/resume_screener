# backend/app/core/prompts.py
"""
Prompt templates used by the pipeline.
- Keys mirror notebook Cell 2: 
  jd_snapshot_and_gates, extract_all_from_chunk, consolidate_resume_json, score_project_complexity_batch
- These are LangChain-friendly templates (use with ChatPromptTemplate.from_template)
"""

from __future__ import annotations

from typing import Dict, List

def _escape_braces_keep_vars(template: str, keep_vars: List[str]) -> str:
    esc = template.replace("{", "{{").replace("}", "}}")
    for v in keep_vars:
        esc = esc.replace("{{" + v + "}}", "{" + v + "}")
    return esc

PROMPTS: Dict[str, str] = {}

# 1) JD snapshot + hard gates (STRICT, evidence-bound)
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

JD:
---
{jd_text}
---
""", ["jd_text"])

# 2) Per-chunk resume extractor (STRICT)
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
""", ["chunk_id", "chunk_text"])

# 3) Consolidator across chunks (evidence required for each item)
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

# 4) Project complexity & transferability (batch)
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
""", ["jd_title", "jd_tokens", "projects_json"])

__all__ = ["PROMPTS"]
