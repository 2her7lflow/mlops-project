from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

from .mlops_utils import (
    evaluate_thresholds,
    load_quality_thresholds,
    read_json,
    resolve_default_kb_dir,
)

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def _safe_log_artifact(path: Path) -> None:
    if mlflow is None:
        return
    try:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="pipeline")
    except Exception:
        pass


def _safe_log_eval_metrics(processed_dir: Path) -> None:
    if mlflow is None:
        return
    p = processed_dir / "eval_metrics.json"
    if not p.exists():
        return
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        metrics = {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        if metrics:
            mlflow.log_metrics(metrics)
    except Exception:
        pass


def _candidate_root(base_kb_dir: Path, run_id: str) -> Path:
    root = base_kb_dir / "candidates" / run_id
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    src_raw = base_kb_dir / "raw"
    dst_raw = root / "raw"
    try:
        os.symlink(src_raw, dst_raw, target_is_directory=True)
    except Exception:
        shutil.copytree(src_raw, dst_raw)
    return root


def _write_gate_report(candidate_processed: Path, failures: list[str], thresholds: dict[str, float]) -> None:
    metrics = read_json(candidate_processed / "eval_metrics.json")
    report = {
        "passed": not failures,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": thresholds,
        "failures": failures,
        "metrics": metrics,
    }
    (candidate_processed / "quality_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _promote_candidate(candidate_root: Path, base_kb_dir: Path) -> None:
    source_processed = candidate_root / "processed"
    target_processed = base_kb_dir / "processed"
    next_processed = base_kb_dir / "processed.__next__"
    backup_processed = base_kb_dir / "processed.previous"

    if not source_processed.exists():
        raise FileNotFoundError(f"Candidate processed directory not found: {source_processed}")

    if next_processed.exists():
        shutil.rmtree(next_processed, ignore_errors=True)
    shutil.copytree(source_processed, next_processed)

    if backup_processed.exists():
        shutil.rmtree(backup_processed, ignore_errors=True)
    if target_processed.exists():
        target_processed.rename(backup_processed)
    next_processed.rename(target_processed)


def _pipeline_once(env: dict[str, str], log_to_mlflow: bool) -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    repo_root = backend_dir.parent.parent
    scripts_dir = backend_dir / "scripts"

    build_cmd = [sys.executable, str(scripts_dir / "build_vectorstore.py")]
    eval_cmd = [sys.executable, str(scripts_dir / "eval_rag.py")]
    export_cmd = [sys.executable, str(scripts_dir / "export_feedback.py")]
    feedback_eval_cmd = [sys.executable, str(scripts_dir / "build_eval_from_feedback.py")]

    kb_dir = (env.get("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(resolve_default_kb_dir(__file__))
        env["KNOWLEDGE_BASE_DIR"] = kb_dir

    base_kb_dir = Path(kb_dir).resolve()
    feedback_dir = base_kb_dir / "feedback"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    candidate_kb_dir = _candidate_root(base_kb_dir, run_id)
    candidate_processed = candidate_kb_dir / "processed"

    base_env = env.copy()
    base_env["KNOWLEDGE_BASE_DIR"] = str(base_kb_dir)

    candidate_env = env.copy()
    candidate_env["KNOWLEDGE_BASE_DIR"] = str(candidate_kb_dir)
    candidate_env["KB_BUILD_MODE"] = "candidate"

    _run(export_cmd, env=base_env)
    _run(feedback_eval_cmd, env=base_env)
    _run(build_cmd, env=candidate_env)
    _run(eval_cmd, env=candidate_env)

    thresholds = load_quality_thresholds(repo_root)
    metrics = read_json(candidate_processed / "eval_metrics.json")
    failures = evaluate_thresholds(metrics, thresholds)
    _write_gate_report(candidate_processed, failures, thresholds)

    if log_to_mlflow:
        _safe_log_artifact(candidate_processed / "manifest.json")
        _safe_log_artifact(candidate_processed / "eval_metrics.json")
        _safe_log_artifact(candidate_processed / "eval_results.jsonl")
        _safe_log_artifact(candidate_processed / "quality_gate.json")
        _safe_log_eval_metrics(candidate_processed)
        _safe_log_artifact(feedback_dir / "export.jsonl")
        _safe_log_artifact(backend_dir / "data" / "eval" / "generated_from_feedback.jsonl")

    if failures:
        print("WARN: candidate KB failed quality gate. Current production KB is unchanged.", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return

    _promote_candidate(candidate_kb_dir, base_kb_dir)
    print(f"Promoted candidate KB -> production ({candidate_kb_dir.name})", flush=True)

    if log_to_mlflow:
        _safe_log_artifact(base_kb_dir / "processed" / "manifest.json")
        _safe_log_artifact(base_kb_dir / "processed" / "eval_metrics.json")

    if env.get("ENABLE_DVC_PUSH", "false").lower() in {"1", "true", "yes"}:
        try:
            _run(["dvc", "push"], env=env)
        except FileNotFoundError:
            print("WARN: dvc not installed in this container. Skipping dvc push.", flush=True)


def job() -> None:
    env = os.environ.copy()
    uri = (env.get("MLFLOW_TRACKING_URI") or "").strip()
    exp = (env.get("MLFLOW_EXPERIMENT_NAME") or "pet-nutrition-rag").strip()
    pipeline_tag = env.get("MLFLOW_RUN_TAG_PIPELINE", "scheduled")

    if mlflow is not None and uri:
        try:
            mlflow.set_tracking_uri(uri)
            try:
                mlflow.set_experiment(exp)
            except Exception:
                pass

            with mlflow.start_run(run_name="pipeline") as run:
                env["MLFLOW_PARENT_RUN_ID"] = run.info.run_id
                env["MLFLOW_RUN_TAG_PIPELINE"] = pipeline_tag
                try:
                    mlflow.set_tags(
                        {
                            "component": "scheduler",
                            "pipeline": pipeline_tag,
                            "kb_dir": env.get("KNOWLEDGE_BASE_DIR", ""),
                            "dvc_push_enabled": env.get("ENABLE_DVC_PUSH", "false"),
                        }
                    )
                except Exception:
                    pass

                _pipeline_once(env, log_to_mlflow=True)
            return
        except Exception as e:
            print(f"WARN: MLflow pipeline logging failed ({type(e).__name__}). Falling back to non-MLflow run.", flush=True)

    _pipeline_once(env, log_to_mlflow=False)


def main() -> None:
    days = int(os.getenv("REBUILD_EVERY_DAYS", "3"))
    run_once = os.getenv("SCHEDULER_RUN_ON_START", "true").lower() in {"1", "true", "yes"}

    sched = BlockingScheduler()
    sched.add_job(job, "interval", days=days)

    if run_once:
        job()

    print(f"Scheduler started (interval={days} days)", flush=True)
    sched.start()


if __name__ == "__main__":
    main()
