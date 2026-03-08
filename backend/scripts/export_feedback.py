from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import json
import os
from pathlib import Path

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Feedback


def _kb_dir() -> Path:
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "").strip()
    if not kb_dir:
        kb_dir = str(Path(__file__).resolve().parents[2] / "knowledge_base")
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir
    return Path(kb_dir).resolve()

def main() -> None:
    kb = _kb_dir()
    out_dir = kb / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "export.jsonl"

    db: Session = SessionLocal()
    try:
        rows = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(5000).all()
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                question = getattr(r, "question", None)
                corrected_answer = getattr(r, "corrected_answer", None)
                review_ready = bool(question and corrected_answer and (r.rating == -1 or r.category == "accuracy"))
                f.write(json.dumps({
                    "id": r.id,
                    "user_email": r.user_email,
                    "pet_id": r.pet_id,
                    "page": r.page,
                    "category": r.category,
                    "rating": r.rating,
                    "message": r.message,
                    "question": question,
                    "answer": getattr(r, "answer", None),
                    "corrected_answer": corrected_answer,
                    "review_ready": review_ready,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }, ensure_ascii=False) + "\n")
        print(f"OK exported {len(rows)} feedback rows to {out_path}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
