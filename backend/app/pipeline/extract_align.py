# app/pipeline/extract_align.py

from __future__ import annotations
import json, math, re
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

__all__ = ["run_extract_align"]

# ---------------- small utils ----------------
def _now() -> datetime: return datetime.utcnow()
def _cos(u: List[float], v: List[float]) -> float:
    if not u or not v: return 0.0
    num = sum(a*b for a,b in zip(u,v))
    den = (sum(a*a for a in u) ** 0.5) * (sum(b*b for b in v) ** 0.5) + 1e-9
    return num/den
def _escape_braces_keep_vars(template: str, keep_vars: List[str]) -> str:
    esc = template.replace("{", "{{").replace("}", "}}")
    for v in keep_vars: esc = esc.replace("{{"+v+"}}", "{"+v+"}")
    return esc
def _uniq_preserve(xs):
    seen=set(); out=[]
    for x in xs or []:
        k = json.dumps(x, sort_keys=True) if isinstance(x,(dict,list)) else str(x).lower()
        if k not in seen: seen.add(k); out.append(x)
    return out
def _clip(s: Optional[str], n: int = 1200) -> str:
    if not s: return ""
    s = str(s); return s if len(s) <= n else s[:n]

_JSON_OBJECT_RE = re.compile(r"\{.*\}|\[.*\]", re.S)
def _json_loose(s: str) -> Any:
    if not s: return {}
    text = s.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    m = _JSON_OBJECT_RE.search(text)
    if m: text = m.group(0)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    def _q(mo: re.Match) -> str: return f'"{mo.group(1)}":'
    text = re.sub(r'(?m)(?<!")\b([A-Za-z_][A-Za-z0-9_\- ]*)\b\s*:', _q, text)
    text = text.replace("'", '"')
    text = re.sub(r'""', '"', text)
    try:
        return json.loads(text)
    except Exception:
        m2 = _JSON_OBJECT_RE.search(text)
        if m2:
            frag = re.sub(r",(\s*[}\]])", r"\1", m2.group(0)).replace("'", '"')
            return json.loads(frag)
        raise

# ---------------- prompts (same shape as your notebook) ----------------
JD_PROMPT_STRICT = _escape_braces_keep_vars(r"""
You are assisting an Applicant Tracking System.

TASK: Convert the Job Description into a STRICT, EVIDENCE-BOUND JSON snapshot.
RULES:
- Only include items that appear EXPLICITLY in the JD text (no invention).
- Family terms stay as-is (no enumerating examples).
- Tokens are atomic/lowercase. Provide ≤160-char evidence for everything.

Return STRICT JSON ONLY:
{
  "title": "<short>",
  "must_haves": [{"token": "", "evidence": "", "conf": 0.0}],
  "required":    [{"token": "", "evidence": "", "conf": 0.0}],
  "preferred":   [{"token": "", "evidence": "", "conf": 0.0}],
  "responsibilities": [{"phrase": "", "evidence": "", "conf": 0.0}],
  "hard_gates": {
    "degree_required": { "value": true|false, "evidence": "" },
    "min_years": { "value": null | 0, "evidence": "" },
    "license": [{ "token": "", "evidence": "" }],
    "work_auth": { "value": null | "us citizen|eu work permit|...", "evidence": "" },
    "clearance": { "value": null | "active secret|...", "evidence": "" },
    "location_mode": { "value": null | "onsite|hybrid|remote", "evidence": "" },
    "onsite_city": { "value": null | "<city>", "evidence": "" },
    "shift": { "value": null | "night|rotational|...", "evidence": "" },
    "travel": { "value": null | "<% or phrase>", "evidence": "" }
  }
}

JD:
---
{jd_text}
---
""", ["jd_text"])

EXTRACT_PROMPT_STRICT = _escape_braces_keep_vars(r"""
Extract structured resume data that appears in the CHUNK ONLY. Do NOT infer beyond this chunk.

Return STRICT JSON only:
{
  "contacts": { "name": null, "email": null, "phone": null, "links": ["urls"], "evidence": {"name":"", "email":"", "phone":"", "links":["..."]} },
  "education": [ { "degree": "", "field": "", "institution": "", "start": null, "end": null, "location": null, "evidence": "" } ],
  "experience": [ { "title": "", "company": "", "location": null, "start": null, "end": null, "highlights": ["..."], "evidence": "" } ],
  "projects": [ { "name": "", "tech": ["tokens"], "impact": null, "links": ["urls"], "role": null, "duration": null, "highlights": ["..."], "evidence": "" } ],
  "skills": [ { "token": "", "evidence": "" } ],
  "certifications": [ { "token": "", "evidence": "" } ],
  "awards": [ { "token": "", "evidence": "" } ],
  "locations": [ { "token": "", "evidence": "" } ]
}

Chunk ID: {chunk_id}
Chunk:
---
{chunk_text}
---
""", ["chunk_id","chunk_text"])

CONSOLIDATE_PROMPT = _escape_braces_keep_vars(r"""
You are merging multiple chunk-level JSON extractions of a single resume.

Rules:
- Deduplicate across chunks.
- Unify obvious synonyms ONLY if each synonym has evidence somewhere in the inputs.
- If there is no evidence for an item, DROP it.
- Keep arrays of evidence in the final.

Return STRICT JSON only:
{
  "contacts": { "name": null, "email": null, "phone": null, "links": ["urls"] },
  "education": [ { "degree": "", "field": "", "institution": "", "start": null, "end": null, "location": null, "evidence": ["..."] } ],
  "experience": [ { "title": "", "company": "", "location": null, "start": null, "end": null, "highlights": ["..."], "evidence": ["..."] } ],
  "projects": [ { "name": "", "tech": ["tokens"], "impact": null, "links": ["urls"], "role": null, "duration": null, "highlights": ["..."], "evidence": ["..."] } ],
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

PROJECT_COMPLEXITY_PROMPT = _escape_braces_keep_vars(r"""
Score candidate PROJECTS on:
- "complexity" (0..1) and
- "transferability" (0..1) vs JD tokens.

Return STRICT JSON only:
{
  "items": [
    { "name": "<exact>", "complexity": 0.0, "transferability": 0.0, "rationale": "<<=220 chars>" }
  ]
}

JD title: {jd_title}
JD tokens (context): {jd_tokens}

PROJECTS (JSON array):
---
{projects_json}
---
""", ["jd_title","jd_tokens","projects_json"])

# ---------------- safe handle getter ----------------
def _get_handles(state: Dict[str, Any]):
    h = state.get("_handles") or state.get("handles")
    if h and all(k in h and h[k] is not None for k in ("llm","embed_texts","embed_query")):
        return h
    # lazy register from embeddings if ingest wasn’t called for some reason
    try:
        from app.pipeline.embeddings import HANDLES as H
    except Exception as e:
        raise RuntimeError("Model handles not found; check embeddings/ingest.") from e
    to_register = {k: H[k] for k in ("llm","embed_texts","embed_query")}
    state.setdefault("_handles", {}); state.setdefault("handles", {})
    state["_handles"].update(to_register); state["handles"].update(to_register)
    return state["_handles"]

# ---------------- main entry ----------------
def run_extract_align(STATE: Dict[str, Any]) -> None:
    handles = _get_handles(STATE)
    llm = handles["llm"]
    embed_texts = handles["embed_texts"]
    embed_query = handles["embed_query"]

    opts  = STATE["options"]
    paths = STATE["artifacts"]["paths"]

    def _budget_ok(n: int = 1) -> bool: return STATE["_llm_calls"] + n <= opts["llm_budget_calls"]
    def _bump(n: int = 1): STATE.setdefault("_llm_calls", 0); STATE["_llm_calls"] += n

    # ----- 1) JD snapshot -----
    if not STATE["jd_snapshot"].get("required") and _budget_ok(1):
        chain = ChatPromptTemplate.from_template(JD_PROMPT_STRICT) | llm
        raw = chain.invoke({"jd_text": STATE["raw"]["jd_text"]})
        _bump(1)
        obj = _json_loose(getattr(raw, "content", str(raw)) or "{}")

        js = STATE["jd_snapshot"]
        js["title"] = obj.get("title") or js.get("title")

        def _pull(items, key="token", evkey="evidence", cfkey="conf"):
            toks, evid, conf = [], {}, {}
            for it in items or []:
                tok = (it.get(key) or "").strip().lower()
                ev  = (it.get(evkey) or "").strip()
                try: cf = float(it.get(cfkey) or 0.0)
                except Exception: cf = 0.0
                if tok:
                    toks.append(tok); evid[tok] = ev; conf[tok] = cf
            uniq = list(dict.fromkeys(toks))
            return uniq, evid, conf

        must, must_ev, must_cf = _pull(obj.get("must_haves") or [])
        req,  req_ev,  req_cf  = _pull(obj.get("required") or [])
        pref, pref_ev, pref_cf = _pull(obj.get("preferred") or [])
        resp, resp_ev, resp_cf = _pull(obj.get("responsibilities") or [], key="phrase")

        js["must_haves"] = must
        js["required"]   = req
        js["preferred"]  = pref
        js["responsibilities"] = resp
        js["evidence"] = {"must": must_ev, "req": req_ev, "pref": pref_ev, "resp": resp_ev}
        js["conf"]     = {"must": must_cf, "req": req_cf, "pref": pref_cf, "resp": resp_cf}

        hg = obj.get("hard_gates") or {}
        def _gval(k): g = hg.get(k) or {}; return g.get("value") if isinstance(g, dict) else None
        def _gev(k):  g = hg.get(k) or {}; return g.get("evidence") if isinstance(g, dict) else ""
        js["hard_gates"] = {
            "degree_required": _gval("degree_required") if _gval("degree_required") is not None else False,
            "min_years": _gval("min_years"),
            "license": [x.get("token","").lower() for x in (hg.get("license") or []) if isinstance(x, dict) and x.get("token")],
            "work_auth": _gval("work_auth"),
            "clearance": _gval("clearance"),
            "location_mode": _gval("location_mode"),
            "onsite_city": _gval("onsite_city"),
            "shift": _gval("shift"),
            "travel": _gval("travel"),
        }
        js["hard_gate_evidence"] = {k: _gev(k) for k in [
            "degree_required","min_years","license","work_auth","clearance","location_mode","onsite_city","shift","travel"
        ]}
        if opts.get("keep_artifacts", True):
            Path(paths["jd_snapshot_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths["jd_snapshot_json"]).write_text(json.dumps(STATE["jd_snapshot"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] JD snapshot ready (strict; evidence-bound).")

    # ----- 2) Select chunks (budget-aware) -----
    sel_ids = list(STATE.get("faiss", {}).get("topk_ids") or [])
    if not sel_ids:
        sel_ids = [c["id"] for c in STATE["chunks"][:opts["extract_max_chunks"]]]
    sel_ids = sel_ids[:opts["extract_max_chunks"]]
    id2chunk = {c["id"]: c for c in STATE["chunks"]}
    sel_chunks = [id2chunk[i] for i in sel_ids if i in id2chunk]

    remaining_llm = max(0, opts["llm_budget_calls"] - STATE.get("_llm_calls", 0))
    reserve = 2 if remaining_llm > 2 else 0
    allowed = max(0, min(len(sel_chunks), remaining_llm - reserve))
    if allowed < len(sel_chunks): sel_chunks = sel_chunks[:allowed]
    print(f"[ok] Planning extraction for {len(sel_chunks)} chunk(s) within budget (used={STATE.get('_llm_calls',0)}/{opts['llm_budget_calls']}).")

    # ----- 3) Per-chunk extraction -----
    entities_by_chunk: Dict[str, Dict[str, Any]] = {}
    if sel_chunks:
        chain = ChatPromptTemplate.from_template(EXTRACT_PROMPT_STRICT) | llm
        for ch in sel_chunks:
            if not _budget_ok(1): break
            out = chain.invoke({"chunk_id": ch["id"], "chunk_text": ch["text"]})
            _bump(1)
            raw = getattr(out, "content", str(out))
            try:
                obj = _json_loose(raw)
            except Exception:
                if opts.get("keep_artifacts", True):
                    (Path(paths["base_dir"]) / f"bad_json_chunk_{ch['id']}.txt").write_text(raw or "", encoding="utf-8")
                obj = {}
            entities_by_chunk[ch["id"]] = obj

    if opts.get("keep_artifacts", True):
        (Path(paths["base_dir"]) / "extract_pass_raw.json").write_text(
            json.dumps(entities_by_chunk, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----- 4) Consolidation (try LLM, fallback local) -----
    merged: Dict[str, Any] = {}
    if entities_by_chunk and _budget_ok(1):
        try:
            cons_chain = ChatPromptTemplate.from_template(CONSOLIDATE_PROMPT) | llm
            payload = json.dumps(entities_by_chunk, ensure_ascii=False)
            cons_raw = cons_chain.invoke({"chunk_json": payload})
            _bump(1)
            merged = _json_loose(getattr(cons_raw, "content", str(cons_raw)))
        except Exception:
            merged = {}

    if not merged:
        merged = {
            "contacts": {"name": None, "email": None, "phone": None, "links": []},
            "education": [], "experience": [], "projects": [],
            "skills": [], "certifications": [], "awards": [], "locations": []
        }
        for obj in entities_by_chunk.values():
            if not isinstance(obj, dict): continue
            c = obj.get("contacts") or {}
            if isinstance(c, dict):
                if c.get("name") and not merged["contacts"]["name"]: merged["contacts"]["name"] = c.get("name")
                if c.get("email") and not merged["contacts"]["email"]: merged["contacts"]["email"] = c.get("email")
                if c.get("phone") and not merged["contacts"]["phone"]: merged["contacts"]["phone"] = c.get("phone")
                if isinstance(c.get("links"), list): merged["contacts"]["links"].extend([str(u) for u in c.get("links") if u])

            for ed in obj.get("education") or []:
                if isinstance(ed, dict): merged["education"].append(ed)

            exp_candidates = []
            for k in ("experience","work_experience","employment","roles","jobs"):
                if isinstance(obj.get(k), list): exp_candidates.extend(obj.get(k))
            for ex in exp_candidates:
                if isinstance(ex, dict): merged["experience"].append(ex)

            for pr in obj.get("projects") or []:
                if isinstance(pr, dict): merged["projects"].append(pr)

            for s in obj.get("skills") or []:
                if isinstance(s, str):
                    merged["skills"].append({"canonical": s.lower(), "aliases": [], "evidence": []})
                elif isinstance(s, dict):
                    tok = (s.get("canonical") or s.get("token") or "").lower()
                    if tok:
                        merged["skills"].append({"canonical": tok, "aliases": [], "evidence": [s.get("evidence","")] if s.get("evidence") else []})

            for ce in obj.get("certifications") or []:
                if isinstance(ce, dict):
                    nm = (ce.get("canonical") or ce.get("token") or "").lower()
                    if nm: merged["certifications"].append({"canonical": nm, "aliases": [], "evidence": [ce.get("evidence","")] if ce.get("evidence") else []})
                elif isinstance(ce, str):
                    merged["certifications"].append({"canonical": ce.lower(), "aliases": [], "evidence": []})

            for aw in obj.get("awards") or []:
                if isinstance(aw, dict):
                    nm = (aw.get("canonical") or aw.get("token") or "").lower()
                    if nm: merged["awards"].append({"canonical": nm, "evidence": [aw.get("evidence","")] if aw.get("evidence") else []})
                elif isinstance(aw, str):
                    merged["awards"].append({"canonical": aw.lower(), "evidence": []})

            for loc in obj.get("locations") or []:
                if isinstance(loc, dict):
                    nm = (loc.get("canonical") or loc.get("token") or "").lower()
                    if nm: merged["locations"].append({"canonical": nm, "evidence": [loc.get("evidence","")] if loc.get("evidence") else []})
                elif isinstance(loc, str):
                    merged["locations"].append({"canonical": loc.lower(), "evidence": []})

        for k in ("education","experience","projects","skills","certifications","awards","locations"):
            merged[k] = _uniq_preserve(merged[k])

    # ----- 5) Move into STATE -----
    STATE.setdefault("contacts", {"name": None, "email": None, "phone": None,
                                  "links": {"linkedin":None,"github":None,"portfolio":None,"website":None}})
    c = merged.get("contacts") or {}
    if isinstance(c, dict):
        if c.get("name"):  STATE["contacts"]["name"]  = STATE["contacts"].get("name")  or c.get("name")
        if c.get("email"): STATE["contacts"]["email"] = STATE["contacts"].get("email") or c.get("email")
        if c.get("phone"): STATE["contacts"]["phone"] = STATE["contacts"].get("phone") or c.get("phone")
        for u in (c.get("links") or []):
            u = str(u)
            if "linkedin.com" in u and not STATE["contacts"]["links"].get("linkedin"): STATE["contacts"]["links"]["linkedin"] = u
            elif "github.com" in u and not STATE["contacts"]["links"].get("github"):   STATE["contacts"]["links"]["github"] = u
            elif not STATE["contacts"]["links"].get("portfolio"): STATE["contacts"]["links"]["portfolio"] = u

    for key in ["education","timeline","projects","skills","certs","awards","locations"]:
        STATE[key] = []

    STATE["education"].extend(merged.get("education") or [])
    exp_list = merged.get("experience") or []
    if not exp_list and merged.get("projects"):
        for pr in merged["projects"]:
            STATE["timeline"].append({
                "title": pr.get("role") or "project",
                "company": None, "location": None, "start": None, "end": None,
                "highlights": pr.get("highlights") or ([pr.get("impact")] if pr.get("impact") else []),
                "evidence": pr.get("evidence") or [],
            })
    else:
        STATE["timeline"].extend(exp_list)

    STATE["projects"].extend(merged.get("projects") or [])

    for s in merged.get("skills") or []:
        if isinstance(s, dict):
            nm = (s.get("canonical") or s.get("token") or "").lower()
            if nm:
                STATE["skills"].append({"name": nm, "aliases": [a.lower() for a in (s.get("aliases") or [])], "evidence": s.get("evidence") or [], "chunk_ids": []})
        elif isinstance(s, str):
            STATE["skills"].append({"name": s.lower(), "aliases": [], "evidence": [], "chunk_ids": []})

    for ce in merged.get("certifications") or []:
        if isinstance(ce, dict):
            nm = (ce.get("canonical") or ce.get("token") or "").lower(); ev = ce.get("evidence") or []
        else:
            nm = str(ce).lower(); ev = []
        if nm: STATE["certs"].append({"name": nm, "evidence": ev})

    for aw in merged.get("awards") or []:
        if isinstance(aw, dict):
            nm = (aw.get("canonical") or aw.get("token") or "").lower(); ev = aw.get("evidence") or []
        else:
            nm = str(aw).lower(); ev = []
        if nm: STATE["awards"].append({"name": nm, "evidence": ev})

    for loc in merged.get("locations") or []:
        if isinstance(loc, dict): nm = (loc.get("canonical") or loc.get("token") or "").lower()
        else: nm = str(loc).lower()
        if nm: STATE["locations"].append(nm)

    for k in ("education","timeline","projects","skills","certs","awards","locations"):
        STATE[k] = _uniq_preserve(STATE[k])

    # ----- 6) Canonicalize observed terms -----
    raw_terms: List[str] = []
    for s in STATE["skills"]:
        if s.get("name"): raw_terms.append(s["name"])
        for a in s.get("aliases") or []: raw_terms.append(a)
    for p in STATE["projects"]:
        for t in (p.get("tech") or []): raw_terms.append(str(t).lower())
    for r in STATE["timeline"]:
        if r.get("title"): raw_terms.append(str(r["title"]).lower())
        for h in (r.get("highlights") or []):
            if h: raw_terms.append(str(h).lower())
    for c in STATE["certs"]:
        if c.get("name"): raw_terms.append(str(c["name"]).lower())

    unique_terms = list(dict.fromkeys([t for t in (str(x).strip().lower() for x in raw_terms) if t]))
    term_vecs = embed_texts(unique_terms, batch_size=opts.get("embed_batch_size", 64)) if unique_terms else []
    THRESH_CANON = float(opts.get("canon_threshold", 0.82))

    canons: List[Dict[str, Any]] = []
    for term, vec in zip(unique_terms, term_vecs):
        if not canons:
            canons.append({"name": term, "vec": vec, "members": {term}}); continue
        best_i, best_sc = None, -1.0
        for i, c in enumerate(canons):
            sc = _cos(vec, c["vec"])
            if sc > best_sc: best_sc, best_i = sc, i
        if best_sc >= THRESH_CANON: canons[best_i]["members"].add(term)
        else:                       canons.append({"name": term, "vec": vec, "members": {term}})

    term_to_canon: Dict[str, str] = {}
    for c in canons:
        label = min(c["members"], key=len)
        for m in c["members"]: term_to_canon[m] = label

    STATE.setdefault("canon", {})
    STATE["canon"]["normalized_skills"] = sorted(set(term_to_canon.get(t, t) for t in unique_terms))

    # ----- 7) JD alignment -----
    jd = STATE["jd_snapshot"]
    JD_MUST = jd.get("must_haves", []) or []
    JD_REQ  = jd.get("required", []) or []
    JD_PREF = jd.get("preferred", []) or []
    JD_RESP = jd.get("responsibilities", []) or []

    def _best_match(token: str, candidate_terms: List[str]) -> Dict[str, Any]:
        if not token or not candidate_terms: return {"match": None, "score": 0.0}
        vecs = embed_texts([token] + candidate_terms, batch_size=opts.get("embed_batch_size", 64))
        tv = vecs[0]; cvs = vecs[1:]
        best_sc = -1.0; best_term = None
        for tm, vv in zip(candidate_terms, cvs):
            sc = _cos(tv, vv)
            if sc > best_sc: best_sc, best_term = sc, tm
        return {"match": best_term, "score": float(best_sc)}

    def _align_list(tokens: List[str]) -> List[Dict[str, Any]]:
        STRONG = float(opts.get("strong_sim", 0.78)); PART = float(opts.get("partial_sim", 0.65))
        out = []; canon_terms = STATE["canon"]["normalized_skills"]
        for t in tokens:
            bm = _best_match(t, canon_terms)
            status = "missing"
            if bm["score"] >= STRONG: status = "present_strong"
            elif bm["score"] >= PART: status = "present_partial"
            out.append({
                "name": t, "status": status, "evidence": bm["match"], "similarity": round(bm["score"],3),
                "jd_evidence": jd.get("evidence",{}).get("req",{}).get(t)
                               or jd.get("evidence",{}).get("must",{}).get(t)
                               or jd.get("evidence",{}).get("pref",{}).get(t) or ""
            })
        return out

    STATE.setdefault("jd_alignment", {})
    STATE["jd_alignment"]["must_have"] = _align_list(JD_MUST)
    STATE["jd_alignment"]["required"]  = _align_list(JD_REQ)
    STATE["jd_alignment"]["preferred"] = _align_list(JD_PREF)

    # Responsibilities coverage
    resp_cover = 0.0
    highlights: List[str] = []
    for r in STATE["timeline"]:
        for h in (r.get("highlights") or []):
            if h: highlights.append(h)
    for p in STATE["projects"]:
        if p.get("impact"): highlights.append(p.get("impact"))

    if JD_RESP and highlights:
        rv = embed_texts(JD_RESP, batch_size=opts.get("embed_batch_size", 64))
        hv = embed_texts(highlights[:60], batch_size=opts.get("embed_batch_size", 64))
        PART = float(opts.get("partial_sim", 0.65))
        hits = 0
        for rvec in rv:
            best = max(_cos(rvec, hvec) for hvec in hv) if hv else 0.0
            hits += 1 if best >= PART else 0
        resp_cover = hits / max(1, len(JD_RESP))
    STATE["jd_alignment"]["responsibilities"] = {"coverage": round(resp_cover, 3), "count": len(JD_RESP)}

    # ----- 8) Relevance vs JD vector -----
    jd_vec = embed_query(" | ".join((JD_REQ or []) + (JD_RESP or []))[:4000])
    def _recency_months(end_str: Optional[str]) -> int:
        if not end_str or str(end_str).strip().lower() in {"present","current","now"}: return 0
        m = re.search(r"(20\d{2}|19\d{2})", str(end_str)); y = int(m.group(1)) if m else _now().year
        return max(0, (_now().year - y) * 12)

    for r in STATE["timeline"]:
        blob = " ".join([_clip(r.get("title")), _clip(r.get("company"))] + [_clip(h) for h in (r.get("highlights") or [])])
        rv = embed_query(blob)
        r["jd_relevance"] = float(_cos(rv, jd_vec))
        r["recency_months"] = _recency_months(r.get("end"))

    for p in STATE["projects"]:
        blob = " ".join([_clip(p.get("name"))] + [_clip(t) for t in (p.get("tech") or [])] + ([_clip(p.get("impact"))] if p.get("impact") else []))
        pv = embed_query(blob)
        p["jd_relevance"] = float(_cos(pv, jd_vec))

    # ----- 9) Project complexity / transferability -----
    STATE.setdefault("complexity", {"scored": [], "bonus": 0.0})
    proj_items = []
    for pr in STATE["projects"]:
        ev_len = len(" ".join(pr.get("evidence") or [])) if isinstance(pr.get("evidence"), list) else len(str(pr.get("evidence") or ""))
        score = 0.6 * float(pr.get("jd_relevance") or 0.0) + 0.4 * min(1.0, ev_len / 400.0)
        proj_items.append((score, pr))
    proj_items.sort(key=lambda x: x[0], reverse=True)
    topN = [x[1] for x in proj_items[: int(opts.get("max_projects_for_complexity", 4))]]

    STATE["complexity"]["scored"] = []; STATE["complexity"]["bonus"] = 0.0
    if topN and _budget_ok(1):
        pcs = [{
            "name": pr.get("name") or "",
            "tech": pr.get("tech") or [],
            "impact": pr.get("impact") or "",
            "role": pr.get("role"),
            "duration": pr.get("duration"),
            "highlights": pr.get("highlights") or [],
        } for pr in topN]
        payload = json.dumps(pcs, ensure_ascii=False)
        comp_chain = ChatPromptTemplate.from_template(PROJECT_COMPLEXITY_PROMPT) | llm
        comp_raw = comp_chain.invoke({
            "jd_title": STATE["jd_snapshot"].get("title") or "",
            "jd_tokens": ", ".join((JD_MUST or []) + (JD_REQ or []) + (JD_RESP or []))[:1500],
            "projects_json": payload,
        })
        _bump(1)
        try:
            comp = _json_loose(getattr(comp_raw, "content", str(comp_raw)))
        except Exception:
            comp = {}
        items = comp.get("items") if isinstance(comp, dict) else None
        if isinstance(items, list):
            name2ct = { (it.get("name") or ""): {
                "complexity": float(it.get("complexity") or 0.0),
                "transferability": float(it.get("transferability") or 0.0),
                "rationale": it.get("rationale") or "" } for it in items }
            for pr in STATE["projects"]:
                nm = pr.get("name") or ""
                if nm in name2ct:
                    pr["complexity"] = name2ct[nm]["complexity"]
                    pr["transferability"] = name2ct[nm]["transferability"]
                    pr["ct_rationale"] = name2ct[nm]["rationale"]
            STATE["complexity"]["scored"] = items

            best = 0.0; cap = float(opts.get("complexity_weight_cap", 0.12))
            for it in items:
                v = 0.5*float(it.get("complexity") or 0.0) + 0.5*float(it.get("transferability") or 0.0)
                if v > best: best = v
            STATE["complexity"]["bonus"] = float(min(cap, best * cap))

    if opts.get("keep_artifacts", True):
        Path(paths["complexity_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(paths["complexity_json"]).write_text(json.dumps(STATE["complexity"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] Extraction + consolidation + canonicalization + semantic alignment complete.")
    print(f"  - roles: {len(STATE['timeline'])}, projects: {len(STATE['projects'])}, skills (observed canonical): {len(STATE['canon']['normalized_skills'])}")
    print(f"  - alignment: must={len(STATE['jd_alignment']['must_have'])}, req={len(STATE['jd_alignment']['required'])}, pref={len(STATE['jd_alignment']['preferred'])}")
    print(f"  - responsibilities coverage≈{int(STATE['jd_alignment']['responsibilities']['coverage']*100)}%")
    print(f"  - complexity bonus≈{round(STATE['complexity']['bonus']*100, 1)} pts potential")
