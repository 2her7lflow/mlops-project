from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..models import Pet, ActivityLog, User
from ..schemas import ActivitySync, ActivityCreate, ActivityResponse
from ..nutrition_calculator import NutritionCalculator

router = APIRouter(prefix="/api/activity", tags=["activity"])


def _ensure_pet_owner(db: Session, pet_id: int, user: User) -> Pet:
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


def _calc_calories(steps: int, active_minutes: int) -> float:
    # NOTE: heuristic estimate (documented in report)
    return round((steps * 0.04) + (active_minutes * 3.5), 2)


# -------------------------
# Authenticated endpoints
# -------------------------
@router.post("/logs", response_model=ActivityResponse)
def add_activity_log(payload: ActivityCreate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet = _ensure_pet_owner(db, payload.pet_id, user)
    target_date = datetime.strptime(payload.date, "%Y-%m-%d").date()

    calories_burned = _calc_calories(payload.steps, payload.active_minutes)

    existing = (
        db.query(ActivityLog)
        .filter(ActivityLog.pet_id == pet.id, ActivityLog.activity_date == target_date)
        .first()
    )

    if existing:
        existing.steps = payload.steps
        existing.active_minutes = payload.active_minutes
        existing.calories_burned = calories_burned
        log = existing
    else:
        log = ActivityLog(
            pet_id=pet.id,
            activity_date=target_date,
            steps=payload.steps,
            active_minutes=payload.active_minutes,
            calories_burned=calories_burned,
        )
        db.add(log)

    db.commit()
    db.refresh(log)

    # Pydantic expects date string
    return ActivityResponse(
        id=log.id,
        pet_id=log.pet_id,
        activity_date=log.activity_date.isoformat(),
        steps=log.steps,
        active_minutes=log.active_minutes,
        calories_burned=log.calories_burned,
    )


@router.get("/logs", response_model=list[ActivityResponse])
def list_activity_logs(
    pet_id: int = Query(...),
    limit: int = Query(30, ge=1, le=365),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    pet = _ensure_pet_owner(db, pet_id, user)
    rows = (
        db.query(ActivityLog)
        .filter(ActivityLog.pet_id == pet.id)
        .order_by(ActivityLog.activity_date.desc())
        .limit(limit)
        .all()
    )
    return [
        ActivityResponse(
            id=r.id,
            pet_id=r.pet_id,
            activity_date=r.activity_date.isoformat(),
            steps=r.steps,
            active_minutes=r.active_minutes,
            calories_burned=r.calories_burned,
        )
        for r in rows
    ]


@router.get("/adjust/{pet_id}")
def adjust_meal_for_activity(
    pet_id: int,
    activity_date: str = Query(..., description="YYYY-MM-DD"),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    pet = _ensure_pet_owner(db, pet_id, user)
    target_date = datetime.strptime(activity_date, "%Y-%m-%d").date()

    activity = (
        db.query(ActivityLog)
        .filter(ActivityLog.pet_id == pet.id, ActivityLog.activity_date == target_date)
        .first()
    )

    calc = NutritionCalculator()
    base_calories = calc.calculate_der(
        weight_kg=pet.weight_kg,
        activity_level=pet.activity_level,
        age_years=pet.age_years,
        is_neutered=pet.is_neutered,
    )

    if activity:
        adjusted_calories = calc.adjust_for_activity(base_calories, activity.steps, activity.active_minutes)
        adjustment_pct = ((adjusted_calories - base_calories) / base_calories) * 100
    else:
        adjusted_calories = base_calories
        adjustment_pct = 0.0

    meal_plan = calc.calculate_food_amount(adjusted_calories, "kibble", age_years=pet.age_years)

    return {
        "base_calories": base_calories,
        "adjusted_calories": adjusted_calories,
        "adjustment_percent": round(adjustment_pct, 1),
        "meal_plan": meal_plan,
        "recommendation": (
            f"วันนี้ {pet.name} "
            f"{'มีกิจกรรมมาก' if adjustment_pct > 5 else 'มีกิจกรรมน้อย' if adjustment_pct < -5 else 'มีกิจกรรมปกติ'} "
            f"ควรให้อาหาร {meal_plan['total_grams_per_day']} กรัมต่อวัน "
            f"({'เพิ่มขึ้น' if adjustment_pct > 0 else 'ลดลง' if adjustment_pct < 0 else 'เท่าเดิม'} {abs(round(adjustment_pct, 1))}%)"
        ),
    }


# -------------------------
# Legacy endpoint (kept)
# -------------------------
@router.post("/sync")
def sync_activity_legacy(activity: ActivitySync, db: Session = Depends(get_db)):
    pet = db.query(Pet).filter(Pet.id == activity.pet_id).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")

    calories_burned = _calc_calories(activity.steps, activity.active_minutes)

    log = ActivityLog(
        pet_id=activity.pet_id,
        activity_date=datetime.strptime(activity.date, "%Y-%m-%d").date(),
        steps=activity.steps,
        active_minutes=activity.active_minutes,
        calories_burned=calories_burned,
    )
    db.add(log)
    db.commit()

    return {"status": "success", "calories_burned": calories_burned, "message": "Activity synced successfully"}
