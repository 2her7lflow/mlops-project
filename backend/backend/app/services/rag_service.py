from __future__ import annotations

import logging

from ..rag_engine import get_rag

logger = logging.getLogger(__name__)


def warmup_rag() -> None:
    """Best-effort: load existing vectorstore and prepare chain."""
    try:
        rag = get_rag()
        rag.load_existing_vectorstore()
        rag.setup_qa_chain()
        logger.info("RAG loaded")
    except Exception as e:
        logger.warning("RAG warmup failed (%s). You can run /admin/setup-rag", e)
