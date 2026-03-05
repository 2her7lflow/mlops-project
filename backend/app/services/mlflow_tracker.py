"""Minimal MLflow utilities used by both evaluation scripts and (optionally) API requests.

Design goals (college / medium level):
- Zero impact if MLflow is not configured.
- Keep logging calls small and explicit.

Enable:
  export MLFLOW_TRACKING_URI=http://localhost:5000
  export MLFLOW_EXPERIMENT_NAME=pet-nutrition-rag
  export ENABLE_MLFLOW_CHAT_LOGGING=true   # optional, logs chat runs

NOTE: Do NOT log any secrets (API keys). Keep questions/answers optional.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


def _enabled() -> bool:
    return os.getenv("MLFLOW_TRACKING_URI") and os.getenv("DISABLE_MLFLOW", "false").lower() not in {"1", "true", "yes"}


@contextmanager
def mlflow_run(run_name: str, tags: Optional[Dict[str, str]] = None, log_system_metrics: Optional[bool] = None):
    """Context manager that becomes a no-op if MLflow isn't configured.

    System metrics (CPU/RAM/GPU) are optional and must be enabled explicitly.
    Priority: explicit arg > env var MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.
    """
    if not _enabled():
        yield None
        return

    import mlflow

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pet-nutrition")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(exp_name)

    if log_system_metrics is None:
        log_system_metrics = os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false").lower() in {"1", "true", "yes"}

    if log_system_metrics:
        try:
            # Some MLflow versions require this explicit enable call.
            import mlflow.system_metrics  # type: ignore
            mlflow.system_metrics.enable_system_metrics_logging()  # type: ignore
        except Exception:
            pass

    try:
        run_ctx = mlflow.start_run(run_name=run_name, log_system_metrics=bool(log_system_metrics))
    except TypeError:
        # Older MLflow versions don't support log_system_metrics.
        run_ctx = mlflow.start_run(run_name=run_name)

    with run_ctx:
        if tags:
            mlflow.set_tags(tags)
        yield mlflow


def log_params(params: Dict[str, Any]) -> None:
    if not _enabled():
        return
    import mlflow

    for k, v in (params or {}).items():
        if v is None:
            continue
        mlflow.log_param(k, v)


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    if not _enabled():
        return
    import mlflow

    for k, v in (metrics or {}).items():
        if v is None:
            continue
        mlflow.log_metric(k, float(v), step=step)


def log_artifact(path: str) -> None:
    """Log a file artifact to MLflow if configured."""
    if not _enabled():
        return
    import mlflow

    if path and Path(path).exists():
        mlflow.log_artifact(path)


def log_text(text: str, artifact_file: str) -> None:
    """Log a small text blob as an MLflow artifact.

    Falls back to writing a temp file if mlflow.log_text is unavailable.
    """
    if not _enabled():
        return
    import mlflow

    try:
        mlflow.log_text(text, artifact_file=artifact_file)  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    tmp_dir = Path(".mlflow_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / artifact_file
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(text or "", encoding="utf-8")
    mlflow.log_artifact(str(tmp_path))


def log_dict(data: Dict[str, Any], artifact_file: str) -> None:
    """Log a dict as a JSON artifact."""
    if not _enabled():
        return
    import json

    log_text(json.dumps(data or {}, ensure_ascii=False, indent=2), artifact_file=artifact_file)


def now_ms() -> float:
    return time.perf_counter() * 1000.0
