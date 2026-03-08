from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.mlops_utils import resolve_default_kb_dir

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _normalize_text(text).replace("\n", " ").split(" ") if tok]


def _token_f1(reference: str, prediction: str) -> float:
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _detect_language(text: str) -> str:
    has_thai = any("\u0E00" <= ch <= "\u0E7F" for ch in (text or ""))
    has_ascii = any(("a" <= ch.lower() <= "z") for ch in (text or ""))
    if has_thai:
        return "th"
    if has_ascii:
        return "en"
    return "unknown"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _iter_eval_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary = data_dir / "questions.jsonl"
    generated = data_dir / "generated_from_feedback.jsonl"
    for src in [primary, generated]:
        for row in _load_jsonl(src):
            row["_eval_file"] = src.name
            rows.append(row)
    return rows


def _source_match(expected: Any, sources: list[dict[str, Any]]) -> bool | None:
    if expected is None:
        return None
    if isinstance(expected, str):
        expected_terms = [expected]
    else:
        expected_terms = [str(x) for x in expected if str(x).strip()]
    haystack = " ".join(
        _normalize_text(str(part))
        for src in sources
        for part in [src.get("source"), src.get("snippet")]
        if part is not None
    )
    return all(_normalize_text(term) in haystack for term in expected_terms)


def _expected_bool(row: dict[str, Any], *names: str) -> bool | None:
    for name in names:
        if name in row:
            value = row.get(name)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"true", "1", "yes"}:
                    return True
                if text in {"false", "0", "no"}:
                    return False
    return None


def _get_reference_answer(row: dict[str, Any]) -> str | None:
    for name in ["expected_answer", "reference_answer", "answer"]:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mlflow_log_eval(metrics: dict[str, Any], artifact_paths: list[str], tags: dict[str, str]) -> None:
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
        pass

    parent_id = (os.getenv("MLFLOW_PARENT_RUN_ID") or "").strip()

    def _log_here() -> None:
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
        try:
            with mlflow.start_run(run_id=parent_id):
                with mlflow.start_run(run_name="eval", nested=True):
                    _log_here()
            return
        except Exception:
            pass

    try:
        with mlflow.start_run(run_name="eval"):
            _log_here()
    except Exception:
        return


def main() -> None:
    kb_dir = (os.getenv("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(resolve_default_kb_dir(__file__))
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    kb_path = Path(kb_dir).resolve()
    eval_dir = Path(__file__).resolve().parents[1] / "data" / "eval"
    rows = _iter_eval_rows(eval_dir)
    if not rows:
        raise SystemExit(
            f"Missing eval set at {eval_dir}. Create questions.jsonl (and optionally generated_from_feedback.jsonl)."
        )

    from rag_engine import get_rag

    rag = get_rag()

    total = 0
    retrieval_hit = 0
    no_context = 0
    lat_ms: list[float] = []
    answer_lengths: list[int] = []

    source_checks = 0
    source_matches = 0
    no_context_checks = 0
    no_context_correct = 0
    language_checks = 0
    language_matches = 0
    exact_checks = 0
    exact_matches = 0
    token_f1_scores: list[float] = []
    per_case: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        q = row.get("q") or row.get("question")
        if not q:
            continue
        total += 1
        t0 = time.perf_counter()
        out = rag.ask(str(q), pet_context=None)
        ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(ms)

        answer = str(out.get("answer") or "")
        sources = out.get("sources") or []
        meta = out.get("_meta") or {}

        hit = bool(sources)
        if hit:
            retrieval_hit += 1

        is_no_context = bool(
            meta.get("guardrail_no_context")
            or meta.get("guardrail_no_relevant_kb")
            or meta.get("guardrail_not_indexed")
        )
        if is_no_context:
            no_context += 1

        source_match = _source_match(row.get("expected_source_contains"), sources)
        if source_match is not None:
            source_checks += 1
            if source_match:
                source_matches += 1

        expected_no_context = _expected_bool(row, "expected_no_context")
        if expected_no_context is not None:
            no_context_checks += 1
            if expected_no_context == is_no_context:
                no_context_correct += 1

        expected_language = row.get("expected_language")
        answer_language = _detect_language(answer)
        if isinstance(expected_language, str) and expected_language.strip():
            language_checks += 1
            if expected_language.strip().lower() == answer_language:
                language_matches += 1

        reference_answer = _get_reference_answer(row)
        exact_match = None
        token_f1 = None
        if reference_answer:
            exact_checks += 1
            exact_match = _normalize_text(reference_answer) == _normalize_text(answer)
            if exact_match:
                exact_matches += 1
            token_f1 = _token_f1(reference_answer, answer)
            token_f1_scores.append(token_f1)

        answer_lengths.append(len(answer))
        per_case.append(
            {
                "id": row.get("id") or f"eval_{idx}",
                "eval_file": row.get("_eval_file"),
                "question": q,
                "answer": answer,
                "latency_ms": round(ms, 1),
                "hit": hit,
                "source_match": source_match,
                "expected_no_context": expected_no_context,
                "actual_no_context": is_no_context,
                "expected_language": expected_language,
                "answer_language": answer_language,
                "exact_match": exact_match,
                "token_f1": round(token_f1, 4) if isinstance(token_f1, float) else None,
                "sources": [
                    {
                        "source": s.get("source", "unknown"),
                        "page": s.get("page"),
                    }
                    for s in sources
                ],
            }
        )

    metrics = {
        "eval_total": total,
        "retrieval_hit_rate": round(retrieval_hit / total, 3) if total else 0.0,
        "no_context_rate": round(no_context / total, 3) if total else 0.0,
        "source_match_rate": round(source_matches / source_checks, 3) if source_checks else 1.0,
        "expected_no_context_accuracy": round(no_context_correct / no_context_checks, 3) if no_context_checks else 1.0,
        "language_match_rate": round(language_matches / language_checks, 3) if language_checks else 1.0,
        "exact_match_rate": round(exact_matches / exact_checks, 3) if exact_checks else 0.0,
        "token_f1_avg": round(sum(token_f1_scores) / len(token_f1_scores), 3) if token_f1_scores else 0.0,
        "latency_ms_avg": round(sum(lat_ms) / len(lat_ms), 1) if lat_ms else 0.0,
        "latency_ms_p95": round(sorted(lat_ms)[max(0, int(len(lat_ms) * 0.95) - 1)], 1) if lat_ms else 0.0,
        "answer_length_avg_chars": round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else 0.0,
        "cases_with_source_expectation": source_checks,
        "cases_with_no_context_expectation": no_context_checks,
        "cases_with_language_expectation": language_checks,
        "cases_with_reference_answer": exact_checks,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = kb_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "eval_results.jsonl").open("w", encoding="utf-8") as f:
        for row in per_case:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    artifact_paths = [
        str(out_dir / "eval_metrics.json"),
        str(out_dir / "eval_results.jsonl"),
        str(out_dir / "manifest.json"),
    ]
    tags = {
        "component": "eval",
        "pipeline": os.getenv("MLFLOW_RUN_TAG_PIPELINE", "manual"),
        "kb_dir": str(kb_path),
    }
    _mlflow_log_eval(metrics, artifact_paths, tags)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
