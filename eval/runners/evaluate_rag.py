"""Offline-ish RAG evaluation.

- Retrieval: hit@k based on expected_source_contains and expected keywords.
- Latency: avg/p95.
- Optional: LLM groundedness judge if GOOGLE_API_KEY is set.

Run:
  python scripts/evaluate_rag.py --eval scripts/eval_set.json --k 4

Note: Requires that you have already indexed the KB (POST /admin/setup-rag).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from app.rag_engine import get_rag


def _try_import_mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def _try_ragas_scores(rows: list[dict]) -> dict:
    """Optional RAGAS evaluation (medium-low): faithfulness + answer relevancy.

    Returns {} if ragas is not installed or API key missing.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except Exception:
        return {}

    # RAGAS needs an LLM+embeddings configured via environment (LangChain integrations).
    # We'll only run when GOOGLE_API_KEY exists.
    if not os.getenv("GOOGLE_API_KEY"):
        return {}

    # RAGAS expects contexts used to answer. We approximate with source snippets.
    consult_rows = [r for r in rows if (r.get("mode") or "consult") == "consult"]
    if not consult_rows:
        return {}

    data = {
        "question": [r["question"] for r in consult_rows],
        "answer": [r.get("answer", "") for r in consult_rows],
        "contexts": [[s.get("snippet", "") for s in (r.get("top_sources") or [])] for r in consult_rows],
    }
    ds = Dataset.from_dict(data)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])

    # result is a pandas-like object; convert carefully
    out = {}
    try:
        out["ragas_faithfulness"] = float(result["faithfulness"].mean())
        out["ragas_answer_relevancy"] = float(result["answer_relevancy"].mean())
    except Exception:
        # fallback best-effort
        pass
    return out


def _contains_any(haystack: str, needles: list[str]) -> bool:
    h = (haystack or "").lower()
    return any(n.lower() in h for n in needles)


def _contains_all(haystack: str, needles: list[str]) -> bool:
    h = (haystack or "").lower()
    return all(n.lower() in h for n in (needles or []))


def score_one(item: dict, k: int) -> dict:
    rag = get_rag()

    mode = (item.get("mode") or "consult").lower()
    pet_context = item.get("pet_context")

    t0 = time.perf_counter()
    out = rag.ask(item["question"], pet_context=pet_context)
    latency_ms = (time.perf_counter() - t0) * 1000

    sources = out.get("sources", [])[:k]
    joined_sources = "\n".join([str(s.get("source", "")) for s in sources])
    joined_snips = "\n".join([str(s.get("snippet", "")) for s in sources])

    expected_src = item.get("expected_source_contains", [])
    expected_kw = item.get("expected_keywords", [])
    expected_profile = item.get("expected_answer_contains", [])

    hit_src = _contains_any(joined_sources, expected_src) if expected_src else None
    hit_kw = _contains_any(out.get("answer", ""), expected_kw) if expected_kw else None
    hit_profile = _contains_all(out.get("answer", ""), expected_profile) if expected_profile else None

    # Combined hit logic:
    # - profile mode: evaluate by expected_answer_contains
    # - consult mode: evaluate by source match or answer keywords
    if mode == "profile":
        hit = bool(hit_profile) if hit_profile is not None else False
    else:
        hit = bool(hit_src) if hit_src is not None else False
        hit = hit or (bool(hit_kw) if hit_kw is not None else False)

    return {
        "id": item.get("id"),
        "mode": mode,
        "question": item.get("question"),
        "pet_context": pet_context,
        "latency_ms": round(latency_ms, 2),
        "hit": hit,
        "hit_source": hit_src,
        "hit_keywords": hit_kw,
        "hit_profile": hit_profile,
        "top_sources": sources,
        "answer": out.get("answer", ""),
        "snippets_for_judge": joined_snips,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=str, default=str(Path(__file__).with_name("eval_set.json")))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=str, default="scripts/eval_results.json")
    args = ap.parse_args()

    with open(args.eval, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = [score_one(it, args.k) for it in items]

    hits = [1 if r["hit"] else 0 for r in rows]
    lat = [r["latency_ms"] for r in rows]

    summary = {
        "k": args.k,
        "n": len(rows),
        "hit_rate": round(sum(hits) / len(hits), 3) if hits else 0.0,
        "latency_avg_ms": round(statistics.mean(lat), 2) if lat else 0.0,
        "latency_p95_ms": round(statistics.quantiles(lat, n=20)[18], 2) if len(lat) >= 20 else (round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else 0.0),
    }

    # Optional: profile-only hit rate (if dataset includes profile rows)
    prof = [r for r in rows if (r.get("mode") or "consult") == "profile"]
    if prof:
        summary["profile_hit_rate"] = round(sum(1 for r in prof if r.get("hit")) / len(prof), 3)

    payload = {"summary": summary, "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== RAG Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved: {args.out}")

    # -----------------------------
    # MLflow logging (optional)
    # -----------------------------
    # -----------------------------
    # MLflow logging (optional)
    # -----------------------------
    mlflow = _try_import_mlflow()
    if mlflow is not None and os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "pet-nutrition"))
        run_name = os.getenv("MLFLOW_RUN_NAME", f"rag-k{args.k}")
        with mlflow.start_run(run_name=run_name):
            rag = get_rag()
            # params
            mlflow.log_param("k", args.k)
            mlflow.log_param("prompt_version", getattr(rag, "prompt_version", "unknown"))
            mlflow.log_param("retrieval_strategy", getattr(rag, "retriever_mode", "unknown"))

            # summary metrics (same as before)
            mlflow.log_metric("hit_rate", summary["hit_rate"])
            mlflow.log_metric("latency_avg_ms", summary["latency_avg_ms"])
            mlflow.log_metric("latency_p95_ms", summary["latency_p95_ms"])
            if "profile_hit_rate" in summary:
                mlflow.log_metric("profile_hit_rate", summary["profile_hit_rate"])

            # optional RAGAS
            ragas_scores = _try_ragas_scores(rows)
            for mk, mv in ragas_scores.items():
                mlflow.log_metric(mk, mv)

            # artifacts
            mlflow.log_artifact(args.out)


if __name__ == "__main__":
    # Ensure env is set (DATABASE_URL needed for app import in some setups)
    os.environ.setdefault("DATABASE_URL", os.getenv("DATABASE_URL", "sqlite:///./eval.db"))
    main()
