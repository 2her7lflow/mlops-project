from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.mlops_utils import count_jsonl_rows, get_git_commit, resolve_default_kb_dir

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_source_file_metadata(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not raw_dir.exists():
        return rows
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(raw_dir)
        rows.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = (os.getenv("KNOWLEDGE_BASE_DIR") or "").strip()
    if not kb_dir:
        kb_dir = str(resolve_default_kb_dir(__file__))
        os.environ["KNOWLEDGE_BASE_DIR"] = kb_dir

    kb_path = Path(kb_dir).resolve()
    t0 = time.time()

    from rag_engine import get_rag

    rag = get_rag()
    chunks = rag.rebuild_vectorstore()

    processed = kb_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_duration_sec": round(time.time() - t0, 2),
        "knowledge_base_dir": str(kb_path),
        "build_mode": os.getenv("KB_BUILD_MODE", "default"),
        "chunks_indexed": int(chunks),
        "chunk_records": count_jsonl_rows(processed / "chunks.jsonl"),
        "page_records": count_jsonl_rows(processed / "pages.jsonl"),
        "embedding_model": "gemini-embedding-001",
        "llm_model": os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash"),
        "prompt_version": "rag_v3",
        "retrieval_strategy": getattr(rag, "retriever_mode", "unknown"),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "k_vector": getattr(rag, "k_vector", None),
        "k_bm25": getattr(rag, "k_bm25", None),
        "k_pages": getattr(rag, "k_pages", None),
        "k_per_page": getattr(rag, "k_per_page", None),
        "k_final": getattr(rag, "k_final", None),
        "min_relevance": getattr(rag, "min_relevance", None),
        "git_commit": get_git_commit(repo_root),
        "source_files": _collect_source_file_metadata(kb_path / "raw"),
    }
    manifest["source_file_count"] = len(manifest["source_files"])

    (processed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK rebuild_vectorstore chunks={chunks} sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
