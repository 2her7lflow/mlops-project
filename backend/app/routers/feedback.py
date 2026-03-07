from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..models import Feedback, Pet, User
from ..schemas import FeedbackCreate, FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

_VALID_PAGES = {"general", "advisor", "pets", "activity", "feedback"}
_VALID_CATEGORIES = {"bug", "idea", "ui", "accuracy", "performance", "other"}


def _normalize_text(value: str, fallback: str) -> str:
    value = (value or "").strip().lower()
    return value or fallback


def _ensure_owned_pet(db: Session, pet_id: int, user: User) -> Pet:
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


@router.post("", response_model=FeedbackResponse)
def create_feedback(payload: FeedbackCreate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    page = _normalize_text(payload.page, "general")
    category = _normalize_text(payload.category, "other")
    message = (payload.message or "").strip()

    if page not in _VALID_PAGES:
        raise HTTPException(status_code=400, detail=f"Page must be one of: {sorted(_VALID_PAGES)}")
    if category not in _VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Category must be one of: {sorted(_VALID_CATEGORIES)}")
    if not message:
        raise HTTPException(status_code=400, detail="Feedback message is required")
    if len(message) < 5:
        raise HTTPException(status_code=400, detail="Feedback message must be at least 5 characters")
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Feedback message must be 2000 characters or fewer")

    rating = payload.rating
    if rating is not None and not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    pet_id = payload.pet_id
    if pet_id is not None:
        _ensure_owned_pet(db, pet_id, user)

    row = Feedback(
        user_email=user.email,
        pet_id=pet_id,
        page=page,
        category=category,
        rating=rating,
        message=message,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return FeedbackResponse(
        id=row.id,
        user_email=row.user_email,
        pet_id=row.pet_id,
        page=row.page,
        category=row.category,
        rating=row.rating,
        message=row.message,
        created_at=row.created_at.isoformat(),
    )


@router.get("", response_model=list[FeedbackResponse])
def list_my_feedback(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Feedback)
        .filter(Feedback.user_email == user.email)
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())
        .limit(limit)
        .all()
    )
    return [
        FeedbackResponse(
            id=r.id,
            user_email=r.user_email,
            pet_id=r.pet_id,
            page=r.page,
            category=r.category,
            rating=r.rating,
            message=r.message,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
