from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import json
import os
import time
from pathlib import Path

from rag_engine import get_rag


def main() -> None:
    # Ensure the RAG system resolves KB_DIR consistently
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "").strip()
    if not kb_dir:
        kb_dir = str(Path(__file__).resolve().parents[2] / "knowledge_base")
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    t0 = time.time()
    rag = get_rag()
    chunks = rag.rebuild_vectorstore()

    processed = Path(kb_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_base_dir": kb_dir,
        "chunks_indexed": int(chunks),
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK rebuild_vectorstore chunks={chunks} sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
