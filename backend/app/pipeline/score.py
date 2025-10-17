# backend/app/pipeline/score.py
"""
Cell 5 → refactor:
Scoring → Gates → Eligibility → 3 Narratives → Persist

Entry:
    run_score(state) -> state  (mutates & returns state)
"""

from __future__ import annotations
import json, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from .state import PipelineState, llm

# --------- helpers ----------
def _now():
    return datetime.utcnow()

def _budget_ok(state: PipelineState, n: int = 1) -> bool:
    return state["_llm_calls"] + n <= state["options"]["llm_budget_calls"]

def _bump(state: PipelineState, n: int = 1):
    state["_llm_calls"] += n

def _escape_braces_keep_vars(template: str, keep_vars: List[str]) -> str:
    esc = template.replace("{", "{{").replace("}", "}}")
    for v in keep_vars:
        esc = esc.replace("{{" + v + "}}", "{" + v + "}")
    return esc


def run_score(state: PipelineState) -> PipelineState:
    opts = state.get("options", {})
    weights = opts.get("weights", {
        "must_have_coverage": 30.0,
        "required_coverage": 18.0,
        "preferred_coverage": 8.0,
        "role_alignment": 18.0,
        "project_alignment": 8.0,
        "evidence_depth": 6.0,
        "seniority_fit": 8.0,
        "responsibility_overlap": 4.0,
    })
    gate_cap = opts.get("gate_hard_cap", 59)

    # ---------- 1) Evidence depth ----------
    highlights: List[str] = []
    for r in state.get("timeline", []):
        highlights.extend(r.get("highlights") or [])
    for p in state.get("projects", []):
        if p.get("impact"): highlights.append(p["impact"])

    metric_pat = re.compile(r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?(?:k|m|b)\b|\b\d{2,}\b|\$\s?\d[\d,]*(?:\.\d+)?", re.I)
    metric_count = sum(len(metric_pat.findall(h or "")) for h in highlights)
    evidence_depth = min(1.0, metric_count / 8.0)

    # ---------- 2) Seniority fit ----------
    jd = state.get("jd_snapshot", {})
    req_years = None
    try:
        req_years = int(jd.get("hard_gates", {}).get("min_years") or 0) or None
    except Exception:
        req_years = None

    yrs = state.get("high_level", {}).get("years_experience")
    seniority_fit = 1.0
    if yrs is not None and req_years is not None:
        denom = max(1.0, 0.7 * req_years)  # 70% near-fit rule
        seniority_fit = max(0.0, min(1.0, float(yrs) / denom))

    # ---------- 3) Responsibilities overlap ----------
    resp_cover = float(state.get("jd_alignment", {}).get("responsibilities", {}).get("coverage", 0.0))

    # ---------- 4) Role & Project alignment ----------
    def _parse_date_soft(s: Optional[str]) -> Optional[datetime]:
        if not s: return None
        tl = str(s).strip().lower()
        if any(k in tl for k in ["present","current","now"]): return _now()
        from dateutil import parser
        try:
            dt = parser.parse(tl, default=datetime(2000,1,1), fuzzy=True)
            if 1900 <= dt.year <= 2100:
                return datetime(dt.year, dt.month if dt.month else 1, 1)
        except Exception:
            pass
        m = re.search(r"(20\d{2}|19\d{2})", tl)
        if m: return datetime(int(m.group(1)), 1, 1)
        return None

    def _months_between(a: Optional[datetime], b: Optional[datetime]) -> int:
        if not a or not b: return 0
        return max(0, (b.year - a.year) * 12 + (b.month - a.month))

    def _recency_weight(months: Optional[int]) -> float:
        if months is None: return 0.7
        if months <= 6: return 1.0
        if months <= 24: return 0.5 + 0.5 * (24 - months) / 18.0
        if months <= 60: return 0.3 + 0.2 * (60 - months) / 36.0
        return 0.3

    now = _now()
    role_scores = []
    for r in state.get("timeline", []):
        st = _parse_date_soft(r.get("start"))
        en = _parse_date_soft(r.get("end")) or now
        tenure_mo = _months_between(st, en) if st else 0
        r["tenure_months"] = tenure_mo
        rec_w = _recency_weight(r.get("recency_months", _months_between(en, now)))
        ten_w = min(1.0, (tenure_mo or 0) / 9.0)  # 9+ months ~ full credit
        rel = float(r.get("jd_relevance") or 0.0)
        role_scores.append(rel * rec_w * ten_w)

    role_alignment = sum(role_scores) / max(1, len(role_scores)) if role_scores else 0.0
    proj_scores = [float(p.get("jd_relevance") or 0.0) for p in state.get("projects", [])]
    project_alignment = sum(proj_scores) / max(1, len(proj_scores)) if proj_scores else 0.0

    # ---------- 5) Coverage ----------
    def _coverage(items: List[Dict[str, Any]]) -> float:
        if not items: return 0.0
        total = len(items)
        strong = sum(1 for x in items if x.get("status") == "present_strong")
        partial = sum(1 for x in items if x.get("status") == "present_partial")
        return (strong + 0.6 * partial) / max(1, total)

    must_cov = _coverage(state.get("jd_alignment", {}).get("must_have", []))
    req_cov  = _coverage(state.get("jd_alignment", {}).get("required", []))
    pref_cov = _coverage(state.get("jd_alignment", {}).get("preferred", []))

    if role_alignment >= 0.6: req_cov = min(1.0, req_cov + 0.05)
    if project_alignment >= 0.6: pref_cov = min(1.0, pref_cov + 0.04)

    # ---------- 6) Hard gates ----------
    failed_gates: List[str] = []
    gate_notes: List[str] = []
    hg = jd.get("hard_gates", {}) or {}

    degree_required = bool(hg.get("degree_required"))
    has_degree = any((e.get("degree") or e.get("field")) for e in state.get("education", []))
    if degree_required and not has_degree:
        failed_gates.append("degree_required")
        gate_notes.append("JD requires a degree; none detected in education section")

    if req_years is not None and (yrs is None or yrs + 0.01 < 0.6 * req_years):
        failed_gates.append("min_years_experience")
        gate_notes.append(f"Requires ~{req_years}y; inferred ~{yrs or 0}y")

    licenses_needed = [str(x).lower() for x in (hg.get("license") or [])]
    if licenses_needed:
        resume_lics = [str(c.get("name","")).lower() for c in state.get("certs", [])]
        lic_missing = [ln for ln in licenses_needed if all(ln not in rl for rl in resume_lics)]
        if lic_missing:
            failed_gates.append("license_required")
            gate_notes.append(f"Missing license(s): {', '.join(lic_missing)}")

    if hg.get("work_auth"):
        wanted = str(hg.get("work_auth")).lower()
        if wanted not in (state.get("raw", {}).get("resume_text","").lower()):
            failed_gates.append("work_authorization")
            gate_notes.append(f"Work authorization required: {wanted}")

    if hg.get("clearance"):
        need = str(hg.get("clearance")).lower()
        if need not in (state.get("raw", {}).get("resume_text","").lower()):
            failed_gates.append("security_clearance")
            gate_notes.append(f"Security clearance required: {need}")

    location_soft_penalty = 0
    if hg.get("location_mode") == "onsite":
        city = (hg.get("onsite_city") or "").lower()
        resume_locs = [str(x).lower() for x in state.get("locations") or []]
        if city and all(city not in l for l in resume_locs):
            location_soft_penalty = 5
            gate_notes.append(f"Onsite location preference '{city}' not evidenced in resume")

    state.setdefault("gates", {})
    state["gates"]["failed"] = failed_gates
    state["gates"]["notes"] = gate_notes

    # ---------- 7) Score computation (+ project complexity bonus) ----------
    score = (
        weights["must_have_coverage"]      * must_cov +
        weights["required_coverage"]       * req_cov +
        weights["preferred_coverage"]      * pref_cov +
        weights["role_alignment"]          * role_alignment +
        weights["project_alignment"]       * project_alignment +
        weights["evidence_depth"]          * evidence_depth +
        weights["seniority_fit"]           * seniority_fit +
        weights["responsibility_overlap"]  * resp_cover
    )

    complexity_bonus = float(state.get("complexity", {}).get("bonus", 0.0))
    score += 100.0 * complexity_bonus  # cap handled when computing bonus

    missing_must = [x["name"] for x in state.get("jd_alignment", {}).get("must_have", []) if x["status"] == "missing"]
    if len(missing_must) > 4:
        score -= 5
    if degree_required and not has_degree:
        score -= 3
    if metric_count == 0:
        score -= 3
    score -= location_soft_penalty

    if failed_gates:
        score = min(score, float(gate_cap))

    score_100 = int(round(max(0.0, min(100.0, score))))
    selected = (score_100 >= 50) and (len(failed_gates) == 0)

    # ---------- 8) Narratives ----------
    def _top_names(items: List[Dict[str, Any]], status: str, limit: int = 12) -> List[str]:
        xs_sorted = sorted([x for x in items if x.get("status")==status],
                           key=lambda y: float(y.get("similarity", 0.0)), reverse=True)
        return [x["name"] for x in xs_sorted][:limit]

    strong_must = _top_names(state.get("jd_alignment", {}).get("must_have", []), "present_strong")
    strong_req  = _top_names(state.get("jd_alignment", {}).get("required", []), "present_strong")
    part_req    = _top_names(state.get("jd_alignment", {}).get("required", []), "present_partial")
    pref_hits   = _top_names(state.get("jd_alignment", {}).get("preferred", []), "present_strong")
    gaps        = sorted(set(
                    [x["name"] for x in state.get("jd_alignment", {}).get("must_have", []) if x["status"]=="missing"] +
                    [x["name"] for x in state.get("jd_alignment", {}).get("required", [])  if x["status"]=="missing"]
                 ))[:15]

    top_projects = sorted(
        state.get("projects", []),
        key=lambda p: (float(p.get("complexity",0.0))*0.6 + float(p.get("transferability",0.0))*0.4),
        reverse=True
    )[:3]
    proj_blurbs = [f"{p.get('name','')} (complexity {round(100*float(p.get('complexity',0.0)))}%, transferability {round(100*float(p.get('transferability',0.0)))}%)"
                   for p in top_projects if p.get("name")]

    FINAL_NARRATIVE_PROMPT = r"""
You are writing 3 short paragraphs for an ATS report.
Audience: recruiter/hiring manager. Tone: concise, neutral, useful. Avoid hype words. No bullet points.

Inputs (JSON):
---
{facts_json}
---

Write exactly 3 paragraphs in this order, separated by a blank line:

1) "What aligns well" — Summarize the strongest matches vs JD (must-haves & required), the most relevant responsibilities covered,
   and any role/project alignment signals, referencing projects/roles when helpful.
2) "Gaps & must-haves to address" — Clearly list the most important missing or partially-met items and any hard-gate risks;
   be practical about what to add/clarify in resume.
3) "Transferable strengths" — Highlight complex or substantial projects/internships and explain briefly why they transfer to the role,
   even if not a perfect JD match. Keep to 3–5 crisp sentences total.

Keep each paragraph ≤ 4 sentences. Do not restate the JSON verbatim; synthesize it.
"""

    align = state.get("jd_alignment", {})
    facts = {
        "jd_title": state.get("jd_snapshot", {}).get("title"),
        "must_have_strong": strong_must,
        "required_strong": strong_req,
        "required_partial": part_req,
        "preferred_strong": pref_hits,
        "gaps": gaps,
        "resp_coverage": float(align.get("responsibilities", {}).get("coverage", 0.0)),
        "role_alignment": float(role_alignment),
        "project_alignment": float(project_alignment),
        "years_experience": state.get("high_level", {}).get("years_experience"),
        "req_years": req_years,
        "metric_count": metric_count,
        "failed_gates": state.get("gates", {}).get("failed", []),
        "gate_notes": state.get("gates", {}).get("notes", []),
        "project_summaries": proj_blurbs,
        "complexity_bonus_points": round(100.0*complexity_bonus, 2)
    }
    facts_json = json.dumps(facts, ensure_ascii=False)

    narrative = {}
    if _budget_ok(state, 1):
        try:
            prompt = _escape_braces_keep_vars(FINAL_NARRATIVE_PROMPT, ["facts_json"])
            chain = ChatPromptTemplate.from_template(prompt) | llm
            out = chain.invoke({"facts_json": facts_json})
            _bump(state, 1)
            text = getattr(out, "content", str(out)).strip()
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            while len(parts) < 3: parts.append("")
            narrative = {
                "present_summary": parts[0],
                "gaps_summary": parts[1],
                "bonus_summary": parts[2],
            }
        except Exception:
            narrative = {}
    if not narrative:
        # fallback synthesis
        p1_bits = []
        if strong_must or strong_req:
            p1_bits.append("Strong alignment on: " + ", ".join((strong_must + strong_req)[:8]))
        if part_req:
            p1_bits.append("Partially met: " + ", ".join(part_req[:8]))
        if pref_hits:
            p1_bits.append("Preferred present: " + ", ".join(pref_hits[:8]))
        p1_bits.append(f"Responsibilities overlap ≈ {int(round(100*float(align.get('responsibilities',{}).get('coverage',0.0))))}%.")
        p1 = " ".join(p1_bits)

        p2_bits = []
        if gaps: p2_bits.append("Key gaps: " + ", ".join(gaps))
        if state.get("gates", {}).get("failed"): p2_bits.append("Hard-gate risks: " + ", ".join(state["gates"]["failed"]))
        p2_bits.append("Consider evidencing missing skills/credentials and add measurable outcomes.")
        p2 = " ".join(p2_bits)

        p3_bits = []
        if proj_blurbs: p3_bits.append("Notable transferable work: " + "; ".join(proj_blurbs))
        p3_bits.append("These experiences indicate practical problem-solving and readiness for the role.")
        p3 = " ".join(p3_bits)

        narrative = {
            "present_summary": p1,
            "gaps_summary": p2,
            "bonus_summary": p3
        }

    # ---------- 9) Reasons / Strong matches / Risks ----------
    def _names_with_status(items: List[Dict[str, Any]], status: str) -> List[str]:
        return [x["name"] for x in items if x.get("status") == status]

    strong_from_must = _names_with_status(state.get("jd_alignment", {}).get("must_have", []), "present_strong")
    strong_from_req  = _names_with_status(state.get("jd_alignment", {}).get("required", []), "present_strong")
    strong_matches = sorted(set(strong_from_must + strong_from_req))[:20]

    reasons: List[str] = []
    reasons.append(f"Must-have coverage: {int(round(100*must_cov))}% ; Required: {int(round(100*req_cov))}% ; Preferred: {int(round(100*pref_cov))}%")
    if yrs is not None:
        if req_years is not None:
            reasons.append(f"Experience: {yrs} yrs vs JD ~{req_years} yrs (fit≈{int(round(100*seniority_fit))}%).")
        else:
            reasons.append(f"Experience: {yrs} yrs (JD years not specified).")
    reasons.append(f"Responsibilities overlap≈{int(round(100*resp_cover))}% ; Evidence signals={metric_count}.")
    reasons.append(f"Role alignment≈{int(round(100*role_alignment))}% ; Project alignment≈{int(round(100*project_alignment))}%")
    if complexity_bonus > 0:
        reasons.append(f"Portfolio bonus: +{round(100.0*complexity_bonus, 1)} pts from complex/transferable projects.")

    risk_flags: List[str] = []
    if state.get("gates", {}).get("failed"):
        risk_flags.append("Hard gates failed: " + ", ".join(state["gates"]["failed"]))
    if not state.get("contacts", {}).get("email"):
        risk_flags.append("No email detected")
    if yrs is None:
        risk_flags.append("Years of experience could not be inferred")
    if metric_count == 0:
        risk_flags.append("No quantifiable achievements detected")
    if gate_notes:
        risk_flags.extend(gate_notes)

    # ---------- 10) Persist final ----------
    state["final"] = {
        "score_100": score_100,
        "selected": bool(selected),
        "breakdown": {
            "must_have_coverage": round(weights["must_have_coverage"] * must_cov, 2),
            "required_coverage": round(weights["required_coverage"] * req_cov, 2),
            "preferred_coverage": round(weights["preferred_coverage"] * pref_cov, 2),
            "role_alignment": round(weights["role_alignment"] * role_alignment, 2),
            "project_alignment": round(weights["project_alignment"] * project_alignment, 2),
            "evidence_depth": round(weights["evidence_depth"] * evidence_depth, 2),
            "seniority_fit": round(weights["seniority_fit"] * seniority_fit, 2),
            "responsibility_overlap": round(weights["responsibility_overlap"] * resp_cover, 2),
            "complexity_bonus_points": round(100.0 * complexity_bonus, 2),
            "soft_penalties": {
                "missing_many_must_haves": int(len(missing_must) > 4) * 5,
                "degree_missing_penalty": (3 if (degree_required and not has_degree) else 0),
                "zero_metrics_penalty": (3 if metric_count == 0 else 0),
                "location_penalty": location_soft_penalty,
            },
            "hard_gate_cap": (gate_cap if failed_gates else None),
        },
        "reasons": reasons,
        "strong_matches": strong_matches,
        "skill_gaps": gaps,
        "risk_flags": risk_flags,
        "narrative": {
            "present_summary": narrative["present_summary"],
            "gaps_summary": narrative["gaps_summary"],
            "bonus_summary": narrative["bonus_summary"],
        },
    }

    Path(state["artifacts"]["base_dir"]).mkdir(parents=True, exist_ok=True)
    with open(state["artifacts"]["paths"]["final_json"], "w", encoding="utf-8") as f:
        json.dump(state["final"], f, ensure_ascii=False, indent=2)

    print(f"[ok] Final score: {state['final']['score_100']}/100  | eligible={state['final']['selected']}")
    print("[ok] Breakdown:", json.dumps(state["final"]["breakdown"], indent=2))
    print("\n[ok] Paragraphs prepared for UI.")
    print(f"[ok] Saved → {state['artifacts']['paths']['final_json']}")

    return state
