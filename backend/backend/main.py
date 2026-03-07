"""Backward-compatible entrypoint.

Run:
  uvicorn main:app --reload

All implementation lives in app/main.py.
"""

from app.main import app  # noqa: F401
