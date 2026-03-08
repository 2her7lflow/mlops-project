from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict

_lock = threading.Lock()
_request_counts: Dict[str, int] = defaultdict(int)
_latency_ms_sum: Dict[str, float] = defaultdict(float)
_latency_ms_max: Dict[str, float] = defaultdict(float)

_rag_totals: Dict[str, float] = defaultdict(float)
_rag_counts: Dict[str, int] = defaultdict(int)


def record(path: str, method: str, latency_ms: float) -> None:
    key = f"{method} {path}"
    with _lock:
        _request_counts[key] += 1
        _latency_ms_sum[key] += latency_ms
        if latency_ms > _latency_ms_max[key]:
            _latency_ms_max[key] = latency_ms


def record_rag(route_type: str, latency_ms: float, meta: dict | None = None, sources_count: int = 0) -> None:
    meta = meta or {}
    with _lock:
        _rag_totals["requests_total"] += 1
        _rag_totals["latency_total_ms_sum"] += float(latency_ms)

        if route_type == "rag":
            _rag_totals["rag_requests_total"] += 1
            _rag_counts["rag_requests_total"] += 1
            _rag_totals["avg_contexts_used_sum"] += float(meta.get("num_contexts") or 0.0)
            _rag_totals["avg_best_relevance_sum"] += float(meta.get("best_relevance") or 0.0)
            _rag_totals["avg_unique_sources_sum"] += float(meta.get("unique_sources") or sources_count or 0.0)
            _rag_totals["retrieval_latency_ms_sum"] += float(meta.get("retrieval_ms") or 0.0)
            llm_ms = float(meta.get("draft_llm_ms") or 0.0) + float(meta.get("safety_llm_ms") or 0.0)
            _rag_totals["llm_latency_ms_sum"] += llm_ms

            if meta.get("guardrail_no_context") or meta.get("guardrail_no_relevant_kb") or meta.get("guardrail_not_indexed"):
                _rag_totals["no_context_total"] += 1
            if meta.get("safety_review_run"):
                _rag_totals["safety_review_total"] += 1
            if meta.get("page_index_hit"):
                _rag_totals["page_index_hit_total"] += 1
        else:
            _rag_totals["calc_requests_total"] += 1
            _rag_counts["calc_requests_total"] += 1


def record_feedback(page: str, rating: int | None) -> None:
    with _lock:
        if page == "advisor":
            if rating == 1:
                _rag_totals["feedback_positive_total"] += 1
            elif rating == -1:
                _rag_totals["feedback_negative_total"] += 1


def snapshot() -> dict:
    with _lock:
        rows = []
        for key, count in sorted(_request_counts.items()):
            avg = (_latency_ms_sum[key] / count) if count else 0.0
            rows.append(
                {
                    "endpoint": key,
                    "count": count,
                    "avg_latency_ms": round(avg, 2),
                    "max_latency_ms": round(_latency_ms_max[key], 2),
                }
            )

        rag_requests = int(_rag_totals.get("rag_requests_total", 0.0))
        all_requests = int(_rag_totals.get("requests_total", 0.0))

        rag_summary = {
            "requests_total": all_requests,
            "rag_requests_total": rag_requests,
            "calc_requests_total": int(_rag_totals.get("calc_requests_total", 0.0)),
            "no_context_total": int(_rag_totals.get("no_context_total", 0.0)),
            "safety_review_total": int(_rag_totals.get("safety_review_total", 0.0)),
            "page_index_hit_total": int(_rag_totals.get("page_index_hit_total", 0.0)),
            "feedback_positive_total": int(_rag_totals.get("feedback_positive_total", 0.0)),
            "feedback_negative_total": int(_rag_totals.get("feedback_negative_total", 0.0)),
            "latency_ms_avg": round((_rag_totals.get("latency_total_ms_sum", 0.0) / all_requests), 2) if all_requests else 0.0,
            "avg_contexts_used": round((_rag_totals.get("avg_contexts_used_sum", 0.0) / rag_requests), 2) if rag_requests else 0.0,
            "avg_best_relevance": round((_rag_totals.get("avg_best_relevance_sum", 0.0) / rag_requests), 3) if rag_requests else 0.0,
            "avg_unique_sources": round((_rag_totals.get("avg_unique_sources_sum", 0.0) / rag_requests), 2) if rag_requests else 0.0,
            "retrieval_latency_ms_avg": round((_rag_totals.get("retrieval_latency_ms_sum", 0.0) / rag_requests), 2) if rag_requests else 0.0,
            "llm_latency_ms_avg": round((_rag_totals.get("llm_latency_ms_sum", 0.0) / rag_requests), 2) if rag_requests else 0.0,
        }

        return {
            "generated_at": time.time(),
            "endpoints": rows,
            "rag": rag_summary,
        }
