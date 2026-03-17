from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

YEAR_RE = re.compile(r"\b(1[0-9]{3}|[8-9][0-9]{2})\b")
PLACE_RX = re.compile(r"\b(?:a|in)\s+([A-ZÀ-Ü][\wÀ-ÿ'\-]*(?:\s+[A-ZÀ-Ü][\wÀ-ÿ'\-]*)*)")
SPLIT_RX = re.compile(r"\s+(?:e|ed)\s+|[;,]\s*")

INTENT_PID_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(nato|born)\b", re.I), "P19"),
    (re.compile(r"\b(morto|died)\b", re.I), "P20"),
    (re.compile(r"\b(studiato|ha\s+studiato|educated|formato)\b", re.I), "P69"),
    (re.compile(r"\b(cittadinanza|citizenship)\b", re.I), "P27"),
    (re.compile(r"\b(moglie|marito|coniuge|spouse)\b", re.I), "P26"),
    (re.compile(r"\b(padre)\b", re.I), "P22"),
    (re.compile(r"\b(madre)\b", re.I), "P25"),
    (re.compile(r"\b(occupazione|occupazioni|professione|lavoro)\b", re.I), "P106"),
    (re.compile(r"\b(carica|cariche|ruolo|incarico)\b", re.I), "P39"),
    (re.compile(r"\b(si trova|ubicato|localizzato)\b", re.I), "P276"),
]

@dataclass
class HitView:
    rank: int
    score: float
    fact_id: str
    pid: str
    src: str
    dst: str
    text: str
    src_label: str
    dst_label: str
    matched_year: Optional[str]

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def effective_k(k_max: int, n: int) -> int:
    return int(min(int(k_max), int(n))) if n is not None else int(k_max)

def extract_year(q: str) -> Optional[str]:
    m = YEAR_RE.search(q or "")
    return m.group(1) if m else None

def extract_place(q: str) -> Optional[str]:
    m = PLACE_RX.search(q or "")
    return m.group(1) if m else None

def infer_pid(q: str) -> Optional[str]:
    for rx, pid in INTENT_PID_RULES:
        if rx.search(q or ""):
            return pid
    return None

def split_into_clauses(q: str) -> List[str]:
    parts = [p.strip() for p in SPLIT_RX.split(q or "") if p.strip()]
    return parts if parts else [q.strip()]

def ont_time_fields(pid: str) -> List[str]:
    if pid == "P19": return ["birth", "start"]
    if pid == "P20": return ["death", "end"]
    return ["start", "end"]

def payload_text(p: Dict[str, Any]) -> str:
    return (p.get("text_labeled") or p.get("text") or "").strip()

def payload_year(p: Dict[str, Any], pid: str) -> Optional[str]:
    for f in ont_time_fields(pid):
        v = p.get(f)
        if v is None: continue
        m = YEAR_RE.search(str(v))
        if m: return m.group(1)
    return None

def match_fact(p: Dict[str, Any], pid: Optional[str], place: Optional[str], year: Optional[str]) -> bool:
    if pid and p.get("pid") != pid: return False
    if place:
        dl = (p.get("dst_label") or "")
        if not dl or place.lower() not in dl.lower(): return False
    if year:
        if pid: return payload_year(p, pid) == year
        for f in ["birth", "death", "start", "end"]:
            v = p.get(f)
            if v:
                m = YEAR_RE.search(str(v))
                if m and m.group(1) == year: return True
        return False
    return True

def qdrant_query(
    client: QdrantClient, collection: str, query_vector: List[float], topk: int,
    pid_any: Optional[List[str]] = None, src_any: Optional[List[str]] = None,
) -> List[Any]:
    must = []
    if pid_any: must.append(FieldCondition(key="pid", match=MatchAny(any=pid_any)))
    if src_any: must.append(FieldCondition(key="src", match=MatchAny(any=src_any)))
    flt = Filter(must=must) if must else None
    res = client.query_points(
        collection_name=collection, query=query_vector, limit=topk, with_payload=True, query_filter=flt,
    )
    return list(res.points)

def hit_key(h: Any) -> Tuple:
    p = getattr(h, "payload", None) or {}
    fid = p.get("fact_id")
    if fid: return ("fact_id", str(fid))
    return (str(p.get("src") or ""), str(p.get("pid") or ""), str(p.get("dst") or ""), str(p.get("text_labeled") or p.get("text") or ""))

def sort_and_dedup(hits: List[Any]) -> List[Any]:
    hits_sorted = sorted(hits, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)
    seen, out = set(), []
    for h in hits_sorted:
        k = hit_key(h)
        if k in seen: continue
        seen.add(k)
        out.append(h)
    return out

def summarize_hit(h: Any, rank: int, pid_for_year: Optional[str]) -> HitView:
    p = getattr(h, "payload", None) or {}
    pid = str(p.get("pid") or "")
    matched_year = payload_year(p, pid) if pid else (payload_year(p, pid_for_year) if pid_for_year else None)
    
    return HitView(
        rank, float(getattr(h, "score", 0.0)), str(p.get("fact_id") or ""), pid, 
        str(p.get("src") or ""), str(p.get("dst") or ""), payload_text(p), 
        str(p.get("src_label") or ""), str(p.get("dst_label") or ""), matched_year
    )

def eval_at_k(candidates: List[Any], k: int, pid_eval: Optional[str], place: Optional[str], year: Optional[str]) -> Dict[str, Any]:
    topk = candidates[:k]
    valid_topk = sum(1 for h in topk if match_fact((getattr(h, "payload", None) or {}), pid_eval, place, year))

    rank_first_valid = None
    for i, h in enumerate(candidates, 1):
        if match_fact((getattr(h, "payload", None) or {}), pid_eval, place, year):
            rank_first_valid = i
            break

    return {
        "valid_at_k": valid_topk,
        "noise_at_k": k - valid_topk,
        "success_at_k": 1 if valid_topk > 0 else 0,
        "rank_first_valid": rank_first_valid,
    }

# -------------------
# PURE - SVR
# -------------------
def run_pure(
    client: QdrantClient, embedder: SentenceTransformer, collection: str,
    q: str, topk: int, max_evidence: int,
) -> Dict[str, Any]:
    clauses = split_into_clauses(q)
    is_multi = len(clauses) > 1

    vec = embedder.encode([q], normalize_embeddings=True)[0].tolist()
    hits = qdrant_query(client, collection, vec, topk)

    candidates = sort_and_dedup(hits)
    year, place, pid_eval = extract_year(q), extract_place(q), infer_pid(q)

    k_used = effective_k(max_evidence, len(candidates))
    
    if is_multi:
        metrics = {"success_at_k": None, "rank_first_valid": None, "valid_at_k": None, "noise_at_k": None}
    else:
        metrics = eval_at_k(candidates, k_used, pid_eval, place, year)

    top_hits = [summarize_hit(h, i, pid_eval) for i, h in enumerate(candidates[:k_used], 1)]
    first_valid = summarize_hit(candidates[metrics["rank_first_valid"] - 1], metrics["rank_first_valid"], pid_eval) if metrics.get("rank_first_valid") else None
    candidates_unique_src = len({(getattr(h, "payload", None) or {}).get("src") for h in candidates if (getattr(h, "payload", None) or {}).get("src")})

    return {
        "q": q, "mode": "PURE", "topk": topk, "join_topk": None, "k_max": max_evidence,
        "k_used": k_used, "retrieved_hits": len(hits), "candidates_len": len(candidates),
        "candidates_unique_src": candidates_unique_src, "pid_eval": pid_eval if not is_multi else None,
        "raw_success_at_k": metrics["success_at_k"], "raw_rank_first_valid": metrics["rank_first_valid"],
        "raw_valid_at_k": metrics["valid_at_k"], "raw_noise_at_k": metrics["noise_at_k"],
        "success_at_k": metrics["success_at_k"], "rank_first_valid": metrics["rank_first_valid"],
        "valid_at_k": metrics["valid_at_k"], "noise_at_k": metrics["noise_at_k"],
        "num_clauses": None, "hits_raw_total": None, "hits_meta_total": None, 
        "unique_src_meta_total": None, "join_success": None, "join_common_src": None,
        "join_rate": None, "join_answers_shown": None, "first_joined_src": None,
        "top_hits": top_hits, "first_valid": first_valid 
    }

# -------------------
# GRAPH - OCR
# -------------------
def run_graph(
    client: QdrantClient, embedder: SentenceTransformer, collection: str,
    q: str, topk: int, join_topk: int, max_evidence: int,
) -> Dict[str, Any]:
    clauses = split_into_clauses(q)
    pid_eval, year, place = infer_pid(q), extract_year(q), extract_place(q)

    per_clause, all_hits_for_llm, all_hits_raw = [], [], []
    for c in clauses:
        pid = infer_pid(c)
        if not pid: continue
        vec = embedder.encode([c], normalize_embeddings=True)[0].tolist()
        hits_raw = qdrant_query(client, collection, vec, join_topk, pid_any=[pid])

        year_c, place_c = extract_year(c), extract_place(c)
        if pid in {"P69", "P27"} and not place_c:
            m = re.search(r"(cittadinanza|citizenship|studiato\s+a|ha\s+studiato\s+a|educated_at)\s+(.+)$", c, re.I)
            if m: 
                place_c = m.group(2).strip().strip('"').strip("'").rstrip('?.!,;')

        hits_meta = [h for h in hits_raw if match_fact((getattr(h, "payload", None) or {}), pid, place_c, year_c)]
        srcset = { (getattr(h, "payload", None) or {}).get("src") for h in hits_meta if (getattr(h, "payload", None) or {}).get("src") }
        srcset = {x for x in srcset if x}

        per_clause.append({
            "clause": c, "pid": pid, "hits_raw": len(hits_raw), "hits_meta": len(hits_meta),
            "year": year_c, "place": place_c, "unique_src_meta": len(srcset), "srcset": srcset,
        })
        all_hits_for_llm.extend(hits_meta)
        all_hits_raw.extend(hits_raw)

    candidates = sort_and_dedup(all_hits_for_llm)
    k_used = effective_k(max_evidence, len(candidates))

    raw_success_at_k = raw_rank_first_valid = raw_valid_at_k = raw_noise_at_k = None
    if len(per_clause) == 1:
        raw_candidates = sort_and_dedup(all_hits_raw)
        k_raw = effective_k(max_evidence, len(raw_candidates))
        raw_m = eval_at_k(raw_candidates, k_raw, pid_eval, place, year)
        raw_success_at_k = raw_m["success_at_k"]
        raw_rank_first_valid = raw_m["rank_first_valid"]
        raw_valid_at_k = raw_m["valid_at_k"]
        raw_noise_at_k = raw_m["noise_at_k"]

    num_clauses = len(per_clause)
    hits_raw_total = sum(item["hits_raw"] for item in per_clause)
    hits_meta_total = sum(item["hits_meta"] for item in per_clause)
    all_src = set()
    for item in per_clause: all_src |= item["srcset"]
    unique_src_meta_total = len(all_src)

    join_metrics, joined_ranked, first_joined_src = {}, [], None

    if num_clauses == 1:
        metrics = eval_at_k(candidates, k_used, pid_eval, place, year)
    elif num_clauses >= 2:
        common, union = None, set()
        for item in per_clause:
            s = set(item["srcset"])
            union |= s
            common = s if common is None else (common & s)
        common = {x for x in (common or set()) if x}

        per_clause_best: List[Dict[str, float]] = []
        for item in per_clause:
            pid_c, place_c, year_c = item["pid"], item.get("place"), item.get("year")
            best: Dict[str, float] = {}
            for h in all_hits_for_llm:
                p = getattr(h, "payload", None) or {}
                src = p.get("src")
                if not src or src not in item["srcset"] or p.get("pid") != pid_c: continue
                if not match_fact(p, pid_c, place_c, year_c): continue
                score = float(getattr(h, "score", 0.0))
                if (src not in best) or (score > best[src]): best[src] = score
            per_clause_best.append(best)

        for src in common:
            sum_score = sum(float(best.get(src, 0.0)) for best in per_clause_best)
            joined_ranked.append({"src": src, "sum_score": sum_score})

        joined_ranked.sort(key=lambda x: float(x["sum_score"]), reverse=True)
        joined_ranked = joined_ranked[:max_evidence]
        k_used = len(joined_ranked)

        if joined_ranked:
            first_joined_src = str(joined_ranked[0].get("src") or "") or None

        join_metrics = {
            "join_common_src": len(common),
            "join_rate": safe_div(len(common), len(union)) if union else 0.0,
            "join_answers_shown": k_used,
            "join_success": 1 if len(common) > 0 else 0,
        }
        
        metrics = {"valid_at_k": None, "noise_at_k": None, "success_at_k": None, "rank_first_valid": None}
    else:
        metrics = {"valid_at_k": 0, "noise_at_k": 0, "success_at_k": 0, "rank_first_valid": None}

    top_hits = [summarize_hit(h, i, pid_eval) for i, h in enumerate(candidates[:k_used], 1)] if num_clauses < 2 else []
    first_valid = summarize_hit(candidates[metrics["rank_first_valid"] - 1], metrics["rank_first_valid"], pid_eval) if metrics.get("rank_first_valid") else None
    candidates_unique_src = len({(getattr(h, "payload", None) or {}).get("src") for h in candidates if (getattr(h, "payload", None) or {}).get("src")})

    return {
        "q": q, "mode": "GRAPH", "topk": topk, "join_topk": join_topk, "k_max": max_evidence,
        "k_used": k_used, "retrieved_hits": hits_raw_total, "candidates_len": len(candidates),
        "candidates_unique_src": candidates_unique_src, "pid_eval": pid_eval,
        "raw_success_at_k": raw_success_at_k, "raw_rank_first_valid": raw_rank_first_valid,
        "raw_valid_at_k": raw_valid_at_k, "raw_noise_at_k": raw_noise_at_k,
        "success_at_k": metrics["success_at_k"], "rank_first_valid": metrics["rank_first_valid"],
        "valid_at_k": metrics["valid_at_k"], "noise_at_k": metrics["noise_at_k"],
        "num_clauses": num_clauses, "hits_raw_total": hits_raw_total, "hits_meta_total": hits_meta_total,
        "unique_src_meta_total": unique_src_meta_total,
        "join_success": join_metrics.get("join_success"), "join_common_src": join_metrics.get("join_common_src"),
        "join_rate": join_metrics.get("join_rate"), "join_answers_shown": join_metrics.get("join_answers_shown"),
        "first_joined_src": first_joined_src,
        "top_hits": top_hits, "first_valid": first_valid, "joined_ranked": joined_ranked if num_clauses >= 2 else []
    }