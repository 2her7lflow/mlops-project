from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import json
import os
import time
from pathlib import Path

from rag_engine import get_rag

# Optional: MLflow eval tracking (works with separate MLflow server)
try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows



def _mlflow_log_eval(metrics: dict, artifact_paths: list[str], tags: dict) -> None:
    """Log eval metrics/artifacts to MLflow if configured.

    Behavior:
    - If MLFLOW_TRACKING_URI is not set or mlflow import fails, this is a no-op.
    - If MLFLOW_PARENT_RUN_ID is set, logs into a nested run under that parent.
    - Otherwise, creates a standalone 'eval' run.
    """
    if mlflow is None:
        return

    uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if not uri:
        return

    mlflow.set_tracking_uri(uri)
    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "pet-nutrition-rag")
    try:
        mlflow.set_experiment(exp)
    except Exception:
        # If server rejects experiment create, just proceed with default.
        pass

    parent_id = (os.getenv("MLFLOW_PARENT_RUN_ID") or "").strip()

    def _log_here():
        try:
            mlflow.set_tags(tags)
        except Exception:
            pass
        try:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception:
            pass
        for ap in artifact_paths:
            try:
                if ap and os.path.exists(ap):
                    mlflow.log_artifact(ap, artifact_path="eval")
            except Exception:
                pass

    if parent_id:
        # attach to parent run and create nested run
        try:
            with mlflow.start_run(run_id=parent_id):
                with mlflow.start_run(run_name="eval", nested=True):
                    _log_here()
            return
        except Exception:
            # fall back to standalone
            pass

    try:
        with mlflow.start_run(run_name="eval"):
            _log_here()
    except Exception:
        return
def main() -> None:
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "").strip()
    if not kb_dir:
        kb_dir = str(Path(__file__).resolve().parents[2] / "knowledge_base")
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    eval_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    rows = _load_jsonl(eval_path)
    if not rows:
        raise SystemExit(
            f"Missing eval set at {eval_path}. Create questions.jsonl (e.g. 20-50 lines)."
        )

    rag = get_rag()

    total = len(rows)
    hit = 0
    noctx = 0
    lat_ms: list[float] = []

    for r in rows:
        q = r.get("q") or r.get("question")
        if not q:
            continue
        t0 = time.perf_counter()
        out = rag.ask(str(q), pet_context=None)
        ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(ms)

        sources = out.get("sources") or []
        meta = out.get("_meta") or {}
        if sources:
            hit += 1
        if meta.get("guardrail_no_relevant_kb") == 1.0 or meta.get("guardrail_not_indexed") == 1.0:
            noctx += 1

    metrics = {
        "eval_total": total,
        "retrieval_hit_rate": round(hit / total, 3) if total else 0.0,
        "no_context_rate": round(noctx / total, 3) if total else 0.0,
        "latency_ms_avg": round(sum(lat_ms) / len(lat_ms), 1) if lat_ms else 0.0,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = Path(kb_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # MLflow logging (optional)
    artifact_paths = [
        str(out_dir / "eval_metrics.json"),
        str(out_dir / "manifest.json"),
    ]
    tags = {
        "component": "eval",
        "pipeline": os.getenv("MLFLOW_RUN_TAG_PIPELINE", "manual"),
        "kb_dir": kb_dir,
    }
    _mlflow_log_eval(metrics, artifact_paths, tags)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
