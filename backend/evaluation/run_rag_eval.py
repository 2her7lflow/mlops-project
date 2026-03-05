"""Run a small RAG evaluation and log results to MLflow.

This script evaluates ONE retrieval strategy (the same one used by the API):
  - Page/row-first narrowing (PageIndex-inspired) + Hybrid fusion (Dense chunks + BM25 chunks)

Outputs:
  - Console summary
  - MLflow run with parameters + RAGAS metrics

Notes (college/medium level):
- We always compute RAGAS metrics that DON'T require a reference answer:
    faithfulness, answer_relevancy, context_precision
- context_recall is optional because it needs ground_truth/reference answers.
  If you later fill `ground_truth` in testset.jsonl, set:
      EVAL_INCLUDE_CONTEXT_RECALL=true

Run:
  python -m evaluation.run_rag_eval
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
import faulthandler
from pathlib import Path
from typing import Dict, List, Any, Tuple

from app.services.mlflow_tracker import mlflow_run, log_params, log_metrics, log_artifact, log_dict

def _mlflow_set_tag(key: str, value: str) -> None:
    """Best-effort MLflow tag setter (safe even if MLflow is unavailable)."""
    try:
        import mlflow  # type: ignore
        mlflow.set_tag(key, value)
    except Exception:
        return


# If the eval hangs (LLM call / vectorstore init), periodically dump stack traces to the console.
try:
    faulthandler.enable()
    _secs = int(os.getenv("EVAL_FAULTHANDLER_SECS", "120"))
    if _secs > 0:
        faulthandler.dump_traceback_later(_secs, repeat=True)
except Exception:
    pass



ROOT = Path(__file__).resolve().parent
DEFAULT_TESTSET_JSON = ROOT / "testset.json"
DEFAULT_TESTSET_JSONL = ROOT / "testset.jsonl"


def _load_testset(path: Path) -> List[Dict[str, Any]]:
    """Load either JSON (list) or JSONL (one object per line).

    Supports blank lines and comment lines starting with '#'.
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # JSON list file
    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("testset JSON must be a list")
        return [x for x in data if isinstance(x, dict)]

    # JSONL file
    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _build_dataset(samples: List[Dict[str, Any]]):
    """Return a HF datasets.Dataset for ragas."""
    try:
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError("datasets is required. Install backend/requirements.txt") from e

    return Dataset.from_list(samples)


def _contains_all(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return all((n or "").lower() in t for n in (needles or []) if str(n).strip())


def _run_once(questions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Run retrieval+generation for the current RAG implementation.

    Returns:
      - samples_for_ragas: list[dict]
      - retr_stats: dict
      - profile_stats: dict
      - detailed_rows: list[dict] (all rows)
    """
    # Late import so env vars are applied
    print("[eval] stage=init_rag", flush=True)
    _mlflow_set_tag("stage", "init_rag")
    log_metrics({"eval_heartbeat_ts": float(time.time())})
    from rag_engine import PetNutritionRAG

    rag = PetNutritionRAG()
    print("[eval] stage=load_vectorstore", flush=True)
    _mlflow_set_tag("stage", "load_vectorstore")
    log_metrics({"eval_heartbeat_ts": float(time.time())})
    rag.load_existing_vectorstore()
    print("[eval] stage=setup_qa_chain", flush=True)
    _mlflow_set_tag("stage", "setup_qa_chain")
    log_metrics({"eval_heartbeat_ts": float(time.time())})
    rag.setup_qa_chain()
    print("[eval] stage=qa_ready", flush=True)
    _mlflow_set_tag("stage", "qa_ready")
    log_metrics({"eval_heartbeat_ts": float(time.time())})

    # Retrieval-only stats (consult questions only)
    n = 0
    n_no_docs = 0
    n_weak = 0
    sum_best_rel = 0.0
    sum_ctx = 0.0
    sum_page_hit = 0.0

    # Profile-only stats
    n_profile = 0
    n_profile_hit = 0

    samples_for_ragas: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []

    # Progress logging (helps diagnose "stuck" MLflow runs in the UI)
    progress_every = int(os.getenv("EVAL_PROGRESS_EVERY", "1"))
    processed = 0
    errors = 0
    log_metrics({"eval_questions_done": 0.0, "eval_errors": 0.0})

    for row in questions:
        q = (row.get("question") or "").strip()
        if not q:
            continue

        pet_context = row.get("pet_context")
        mode = (row.get("mode") or "consult").lower()

        # -------------------------
        # Profile-fact mode (no retrieval)
        # -------------------------
        if mode == "profile":
            n_profile += 1
            _t0 = time.perf_counter()
            try:
                ans = rag.ask(q, pet_context=pet_context)
            except Exception as e:
                errors += 1
                processed += 1
                if progress_every > 0 and (processed % progress_every == 0):
                    log_metrics({"eval_questions_done": float(processed), "eval_errors": float(errors)}, step=processed)
                detailed_rows.append({"id": row.get("id"), "mode": "profile", "question": q, "error": repr(e)})
                continue
            _lat = time.perf_counter() - _t0
            processed += 1
            if progress_every > 0 and (processed % progress_every == 0):
                log_metrics({"eval_questions_done": float(processed), "eval_last_latency_sec": float(_lat), "eval_errors": float(errors)}, step=processed)
            answer_text = ans.get("answer", "")

            expected = row.get("expected_answer_contains") or []
            hit = _contains_all(answer_text, expected) if expected else None
            if hit is True:
                n_profile_hit += 1

            detailed_rows.append(
                {
                    "id": row.get("id"),
                    "mode": "profile",
                    "question": q,
                    "pet_context": pet_context,
                    "answer": answer_text,
                    "expected_answer_contains": expected,
                    "hit": hit,
                    "meta": ans.get("_meta", {}),
                }
            )
            continue

        # -------------------------
        # Consult mode (retrieval + generation)
        # -------------------------
        eval_retr = rag.retrieve_for_eval(q, pet_context=pet_context)
        contexts = eval_retr.get("contexts", [])
        best_rel = float(eval_retr.get("best_relevance", 0.0) or 0.0)
        meta = eval_retr.get("_meta", {}) or {}

        n += 1
        sum_best_rel += best_rel
        sum_ctx += float(len(contexts))
        sum_page_hit += float(meta.get("page_index_hit", 0.0) or 0.0)

        if not contexts:
            n_no_docs += 1
        if contexts and best_rel < float(rag.min_relevance):
            n_weak += 1

        _t0 = time.perf_counter()
        try:
            ans = rag.ask(q, pet_context=pet_context)
        except Exception as e:
            errors += 1
            processed += 1
            if progress_every > 0 and (processed % progress_every == 0):
                log_metrics({"eval_questions_done": float(processed), "eval_errors": float(errors)}, step=processed)
            detailed_rows.append({"id": row.get("id"), "mode": "consult", "question": q, "error": repr(e)})
            continue
        _lat = time.perf_counter() - _t0
        processed += 1
        if progress_every > 0 and (processed % progress_every == 0):
            log_metrics({"eval_questions_done": float(processed), "eval_last_latency_sec": float(_lat), "eval_errors": float(errors)}, step=processed)
        answer_text = ans.get("answer", "")

        sample = {
            "id": row.get("id"),
            "question": q,
            "answer": answer_text,
            "contexts": contexts,
            "ground_truth": row.get("ground_truth", ""),
        }
        samples_for_ragas.append(sample)

        detailed_rows.append(
            {
                "id": row.get("id"),
                "mode": "consult",
                "question": q,
                "pet_context": pet_context,
                "answer": answer_text,
                "n_contexts": len(contexts),
                "best_relevance": best_rel,
                "retrieval_meta": meta,
                "answer_meta": ans.get("_meta", {}),
            }
        )

    retr_stats = {
        "n": float(n),
        "avg_best_relevance": (sum_best_rel / n) if n else 0.0,
        "avg_n_contexts": (sum_ctx / n) if n else 0.0,
        "pct_no_context": (n_no_docs / n) if n else 0.0,
        "pct_weak_context": (n_weak / n) if n else 0.0,
        "avg_page_index_hit": (sum_page_hit / n) if n else 0.0,
        "min_relevance_threshold": float(rag.min_relevance),
    }

    profile_stats = {
        "n": float(n_profile),
        "hit_rate": (n_profile_hit / n_profile) if n_profile else 0.0,
    }

    return samples_for_ragas, retr_stats, profile_stats, detailed_rows


def _ragas_score(dataset, llm=None, embeddings=None, include_context_recall: bool = False) -> Dict[str, float]:
    """Compute a standard small set of RAGAS metrics."""
    # Keep metric keys stable even if ragas isn't installed.
    nan = float("nan")
    out: Dict[str, float] = {
        "faithfulness": nan,
        "answer_relevancy": nan,
        "context_precision": nan,
    }
    if include_context_recall:
        out["context_recall"] = nan

    # If we don't have a dataset (or it's empty), return NaNs to keep metric keys stable.
    try:
        if dataset is None:
            return out
        if hasattr(dataset, "__len__") and len(dataset) == 0:
            return out
    except Exception:
        return out

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except Exception:
        return out

    metrics = [faithfulness, answer_relevancy, context_precision]
    scores: Dict[str, float] = dict(out)

    # Always compute the no-reference metrics
    try:
        res = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        scores.update({k: float(v) for k, v in res.items()})
    except Exception:
        return scores

    # Optional: context_recall needs reference answers (ground_truth)
    if include_context_recall:
        try:
            subset = dataset.filter(lambda x: bool((x.get("ground_truth") or "").strip()))
            if len(subset) >= 3:
                res2 = evaluate(subset, metrics=[context_recall], llm=llm, embeddings=embeddings)
                scores.update({k: float(v) for k, v in res2.items()})
            else:
                scores["context_recall"] = float("nan")
        except Exception:
            scores["context_recall"] = float("nan")

    return scores


def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Run RAG evaluation + log to MLflow")
    ap.add_argument("--testset", type=str, default="", help="Path to testset.json or testset.jsonl")
    ap.add_argument("--max_n", type=int, default=int(os.getenv("EVAL_MAX_N", "0")), help="Optional cap to reduce cost (0 = no cap)")
    ap.add_argument("--report", type=str, default=str(ROOT / "reports" / "rag_eval_report.json"), help="Where to write the JSON report")
    args = ap.parse_args(argv)

    testset_path = Path(args.testset).resolve() if args.testset else (DEFAULT_TESTSET_JSON if DEFAULT_TESTSET_JSON.exists() else DEFAULT_TESTSET_JSONL)
    if not testset_path.exists():
        raise SystemExit(f"testset not found: {testset_path}")

    questions = _load_testset(testset_path)
    if not questions:
        raise SystemExit(f"testset is empty: {testset_path}")

    if args.max_n and args.max_n > 0:
        questions = questions[: args.max_n]

    include_recall = os.getenv("EVAL_INCLUDE_CONTEXT_RECALL", "false").lower() in {"1", "true", "yes"}

    # Use the same LLM/embeddings as your app
    from rag_engine import PetNutritionRAG

    tmp = PetNutritionRAG()
    rag = PetNutritionRAG()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow_run(run_name="rag_eval", tags={"component": "evaluation"}, log_system_metrics=os.getenv("EVAL_ENABLE_SYSTEM_METRICS", "true").lower() in {"1","true","yes"}):
# --- heartbeat / stage markers (so the MLflow UI shows activity even if later steps hang)
_mlflow_set_tag("stage", "run_started")
log_metrics({"eval_started": 1.0, "eval_heartbeat_ts": float(time.time())})
print("[eval] stage=run_started", flush=True)

        # Common params
        log_params(
            {
                "testset": str(testset_path.name),
                "testset_path": str(testset_path),
                "n_questions": len(questions),
                "kb_dir": os.getenv("KNOWLEDGE_BASE_DIR", ""),
                "prompt_version": os.getenv("PROMPT_VERSION", "v3"),
                "include_context_recall": include_recall,
                "retrieval_strategy": getattr(rag, "retriever_mode", "unknown"),
                # retrieval knobs (logged for reproducibility)
                "rag_k_vector": os.getenv("RAG_K_VECTOR", "4"),
                "rag_k_bm25": os.getenv("RAG_K_BM25", "6"),
                "rag_k_pages": os.getenv("RAG_K_PAGES", "4"),
                "rag_k_per_page": os.getenv("RAG_K_PER_PAGE", "2"),
                "rag_k_final": os.getenv("RAG_K_FINAL", "5"),
                "rag_min_relevance": os.getenv("RAG_MIN_RELEVANCE", "0.35"),
            }
        )

        samples, retr_stats, profile_stats, detailed_rows = _run_once(questions)

        scores: Dict[str, float]
        if samples:
            ds = _build_dataset(samples)
            scores = _ragas_score(ds, llm=tmp.llm, embeddings=tmp.embeddings, include_context_recall=include_recall)
        else:
            scores = _ragas_score(None, llm=None, embeddings=None, include_context_recall=include_recall)

        # Metrics (same names as before)
        log_metrics({f"ragas_{k}": v for k, v in scores.items()})
        log_metrics({f"retr_{k}": v for k, v in retr_stats.items()})
        log_metrics({f"profile_{k}": v for k, v in profile_stats.items()})

        report = {
            "summary": {
                "ragas": scores,
                "retrieval": retr_stats,
                "profile": profile_stats,
            },
            "rows": detailed_rows,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # Artifacts
        log_artifact(str(testset_path))
        log_artifact(str(report_path))
        log_dict(report.get("summary", {}), artifact_file="evaluation/summary.json")

    print("\n=== RAGAS ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}" if v == v else f"{k}: nan")

    print("\n=== Retrieval stats (consult) ===")
    for k, v in retr_stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Profile stats ===")
    for k, v in profile_stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
