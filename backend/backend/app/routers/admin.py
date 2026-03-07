from __future__ import annotations

from fastapi import APIRouter

from ..rag_engine import get_rag

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/setup-rag")
def setup_rag():
    rag = get_rag()
    chunks = rag.rebuild_vectorstore()
    return {"status": "ok", "chunks_indexed": chunks}
