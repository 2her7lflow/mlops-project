from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def resolve_default_kb_dir(base_file: str | Path) -> Path:
    base_path = Path(base_file).resolve()
    repo_root = base_path.parents[2]
    return repo_root / "knowledge_base"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coerce_yaml_value(raw: str) -> Any:
    value = (raw or "").strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def load_params(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "params.yaml"
    if not path.exists():
        return {}

    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, data)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, remainder = line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if remainder == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent + 2, child))
        else:
            current[key] = _coerce_yaml_value(remainder)
    return data


def load_quality_thresholds(repo_root: Path) -> dict[str, float]:
    defaults: dict[str, float] = {
        "min_retrieval_hit_rate": 0.50,
        "min_source_match_rate": 0.30,
        "min_language_match_rate": 0.95,
        "min_expected_no_context_accuracy": 0.80,
        "max_no_context_rate": 0.80,
        "max_latency_ms_avg": 8000.0,
    }
    params = load_params(repo_root)
    mlops = params.get("mlops", {}) if isinstance(params, dict) else {}
    if isinstance(mlops, dict):
        for key in list(defaults):
            value = mlops.get(key)
            if isinstance(value, (int, float)):
                defaults[key] = float(value)
    return defaults


def evaluate_thresholds(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> list[str]:
    failures: list[str] = []

    def _num(name: str, default: float = 0.0) -> float:
        value = metrics.get(name, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    if _num("retrieval_hit_rate") < thresholds["min_retrieval_hit_rate"]:
        failures.append(
            f"retrieval_hit_rate={_num('retrieval_hit_rate'):.3f} < {thresholds['min_retrieval_hit_rate']:.3f}"
        )
    if _num("source_match_rate") < thresholds["min_source_match_rate"]:
        failures.append(
            f"source_match_rate={_num('source_match_rate'):.3f} < {thresholds['min_source_match_rate']:.3f}"
        )
    if _num("language_match_rate", 1.0) < thresholds["min_language_match_rate"]:
        failures.append(
            f"language_match_rate={_num('language_match_rate', 1.0):.3f} < {thresholds['min_language_match_rate']:.3f}"
        )
    if _num("expected_no_context_accuracy", 1.0) < thresholds["min_expected_no_context_accuracy"]:
        failures.append(
            "expected_no_context_accuracy="
            f"{_num('expected_no_context_accuracy', 1.0):.3f} < {thresholds['min_expected_no_context_accuracy']:.3f}"
        )
    if _num("no_context_rate") > thresholds["max_no_context_rate"]:
        failures.append(
            f"no_context_rate={_num('no_context_rate'):.3f} > {thresholds['max_no_context_rate']:.3f}"
        )
    if _num("latency_ms_avg") > thresholds["max_latency_ms_avg"]:
        failures.append(
            f"latency_ms_avg={_num('latency_ms_avg'):.1f} > {thresholds['max_latency_ms_avg']:.1f}"
        )
    return failures


def get_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        value = out.decode("utf-8", errors="ignore").strip()
        return value or None
    except Exception:
        return None
