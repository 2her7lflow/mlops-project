from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.mlops_utils import resolve_default_kb_dir


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    kb_dir = (os.getenv("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(resolve_default_kb_dir(__file__))
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    feedback_path = Path(kb_dir).resolve() / "feedback" / "export.jsonl"
    out_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "generated_from_feedback.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(feedback_path)
    generated: list[dict] = []
    seen_questions: set[str] = set()

    for row in rows:
        question = (row.get("question") or "").strip()
        corrected_answer = (row.get("corrected_answer") or "").strip()
        rating = row.get("rating")
        category = (row.get("category") or "").strip().lower()

        review_ready = bool(question and corrected_answer and (rating == -1 or category == "accuracy"))
        if not review_ready:
            continue

        normalized_question = " ".join(question.lower().split())
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)

        generated.append(
            {
                "id": f"feedback_{row.get('id')}",
                "question": question,
                "expected_answer": corrected_answer,
                "expected_language": "th" if any("\u0E00" <= ch <= "\u0E7F" for ch in corrected_answer) else "en",
                "source": "feedback_review",
                "feedback_id": row.get("id"),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in generated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"OK generated {len(generated)} eval rows from feedback -> {out_path}")


if __name__ == "__main__":
    main()
