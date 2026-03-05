"""Thin wrapper so the FastAPI app and evaluation use the SAME RAG implementation.

The canonical implementation lives in backend/rag_engine.py (module name: rag_engine).

Why this file exists:
  - Your app code imports `app.rag_engine.get_rag()`.
  - Your evaluation imports `rag_engine.PetNutritionRAG`.
  - Keeping a single source-of-truth prevents "it works in eval but not in API" issues.
"""

from __future__ import annotations

from rag_engine import PetNutritionRAG, get_rag

__all__ = ["PetNutritionRAG", "get_rag"]
