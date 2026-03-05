from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from ..metrics import snapshot

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
