from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

# Optional: MLflow pipeline tracking (works with separate MLflow server)
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


def job() -> None:
    env = os.environ.copy()

    backend_dir = Path(__file__).resolve().parents[1]
    scripts_dir = backend_dir / "scripts"

    # Rebuild vectorstore
    build_cmd = [sys.executable, str(scripts_dir / "build_vectorstore.py")]
    # Evaluate
    eval_cmd = [sys.executable, str(scripts_dir / "eval_rag.py")]
    # Export feedback snapshot
    export_cmd = [sys.executable, str(scripts_dir / "export_feedback.py")]

    kb_dir = (env.get("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(backend_dir.parent / "knowledge_base")
        env["KNOWLEDGE_BASE_DIR"] = kb_dir

    processed_dir = Path(kb_dir) / "processed"
    feedback_dir = Path(kb_dir) / "feedback"

    uri = (env.get("MLFLOW_TRACKING_URI") or "").strip()
    exp = (env.get("MLFLOW_EXPERIMENT_NAME") or "pet-nutrition-rag").strip()
    pipeline_tag = env.get("MLFLOW_RUN_TAG_PIPELINE", "scheduled")

    # If MLflow is configured, create 1 parent run per pipeline execution
    if mlflow is not None and uri:
        try:
            mlflow.set_tracking_uri(uri)
            try:
                mlflow.set_experiment(exp)
            except Exception:
                pass

            with mlflow.start_run(run_name="pipeline") as run:
                # pass parent run ID to child scripts (eval will create a nested run)
                env["MLFLOW_PARENT_RUN_ID"] = run.info.run_id
                env["MLFLOW_RUN_TAG_PIPELINE"] = pipeline_tag

                try:
                    mlflow.set_tags(
                        {
                            "component": "scheduler",
                            "pipeline": pipeline_tag,
                            "kb_dir": kb_dir,
                            "dvc_push_enabled": env.get("ENABLE_DVC_PUSH", "false"),
                        }
                    )
                except Exception:
                    pass

                # 1) build
                _run(build_cmd, env=env)
                # 2) eval
                _run(eval_cmd, env=env)
                # 3) export feedback
                _run(export_cmd, env=env)

                # Log key artifacts/metrics to the parent run as well (convenience)
                _safe_log_artifact(processed_dir / "manifest.json")
                _safe_log_artifact(processed_dir / "eval_metrics.json")
                _safe_log_eval_metrics(processed_dir)
                _safe_log_artifact(feedback_dir / "export.jsonl")

                # Optional: push DVC artifacts (only if dvc is installed + enabled)
                if env.get("ENABLE_DVC_PUSH", "false").lower() in {"1", "true", "yes"}:
                    try:
                        _run(["dvc", "push"], env=env)
                    except FileNotFoundError:
                        print("WARN: dvc not installed in this container. Skipping dvc push.", flush=True)
            return
        except Exception as e:
            print(f"WARN: MLflow pipeline logging failed ({type(e).__name__}). Falling back to non-MLflow run.", flush=True)

    # Fallback: run without MLflow
    _run(build_cmd, env=env)
    _run(eval_cmd, env=env)
    _run(export_cmd, env=env)

    if env.get("ENABLE_DVC_PUSH", "false").lower() in {"1", "true", "yes"}:
        try:
            _run(["dvc", "push"], env=env)
        except FileNotFoundError:
            print("WARN: dvc not installed in this container. Skipping dvc push.", flush=True)


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
