from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from ..metrics import snapshot
from ..db import engine

router = APIRouter(tags=["system"])


@router.get("/")
def read_root():
    return {
        "message": "Pet Nutrition AI API",
        "version": "1.0.0",
        "endpoints": {"pets": "/api/pets", "nutrition": "/api/nutrition", "activity": "/api/activity"},
    }


@router.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics")
def metrics():
    return snapshot()


@router.get("/db")
def db_info():
    """Minimal DB diagnostics (no secrets).

    Use this to confirm the backend is connected to Supabase vs local SQLite.
    """
    try:
        # SQLAlchemy URL object already hides password in str(url)
        url = engine.url
        return {
            "driver": url.drivername,
            "host": getattr(url, "host", None),
            "database": getattr(url, "database", None),
        }
    except Exception as e:
        return {"error": type(e).__name__}
