from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.mlops_utils import (
    evaluate_thresholds,
    load_quality_thresholds,
    read_json,
    resolve_default_kb_dir,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = (os.getenv("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(resolve_default_kb_dir(__file__))
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    metrics_path = Path(kb_dir).resolve() / "processed" / "eval_metrics.json"
    metrics = read_json(metrics_path)
    if not metrics:
        raise SystemExit(f"Missing eval metrics at {metrics_path}")

    thresholds = load_quality_thresholds(repo_root)
    failures = evaluate_thresholds(metrics, thresholds)

    gate_report = {
        "passed": not failures,
        "thresholds": thresholds,
        "failures": failures,
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(gate_report, ensure_ascii=False, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
