"""Thin wrapper so the FastAPI app uses the SAME RAG implementation everywhere.

The canonical implementation lives in backend/rag_engine.py (module name: rag_engine).
"""

from __future__ import annotations

from typing import Any

PetNutritionRAG = Any


def get_rag():
    from rag_engine import get_rag as _get_rag

    return _get_rag()


__all__ = ["PetNutritionRAG", "get_rag"]
