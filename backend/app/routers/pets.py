from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..models import Pet, NutritionPlan, User
from ..schemas import PetCreate, PetCreateAuth, PetResponse, PetUpdate
from ..nutrition_calculator import NutritionCalculator

router = APIRouter(prefix="/api/pets", tags=["pets"])


def _validate_pet(species: str, activity_level: str) -> None:
    if species not in {"dog", "cat"}:
        raise HTTPException(status_code=400, detail="Species must be 'dog' or 'cat'")
    valid_activities = {"sedentary", "moderate", "active", "very_active"}
    if activity_level not in valid_activities:
        raise HTTPException(status_code=400, detail=f"Activity level must be one of: {sorted(valid_activities)}")


# -------------------------
# Authenticated REST endpoints
# -------------------------
@router.post("", response_model=PetResponse)
def create_pet(pet: PetCreateAuth, user: User = Depends(require_user), db: Session = Depends(get_db)):
    _validate_pet(pet.species, pet.activity_level)

    db_pet = Pet(user_email=user.email, **pet.model_dump())
    db.add(db_pet)
    db.commit()
    db.refresh(db_pet)

    # Create initial nutrition plan (today)
    calc = NutritionCalculator()
    daily_calories = calc.calculate_der(
        weight_kg=db_pet.weight_kg,
        activity_level=db_pet.activity_level,
        age_years=db_pet.age_years,
        is_neutered=db_pet.is_neutered,
    )
    meal_plan = calc.calculate_food_amount(daily_calories, "kibble", age_years=db_pet.age_years)

    plan = NutritionPlan(
        pet_id=db_pet.id,
        daily_calories=daily_calories,
        meal_frequency=meal_plan["meal_frequency"],
        portion_size_grams=meal_plan["grams_per_meal"],
        food_type="kibble",
        notes="Initial calculation based on profile",
    )
    db.add(plan)
    db.commit()

    return db_pet


@router.get("", response_model=list[PetResponse])
def list_my_pets(user: User = Depends(require_user), db: Session = Depends(get_db)):
    return db.query(Pet).filter(Pet.user_email == user.email).order_by(Pet.id.desc()).all()


@router.get("/{pet_id}", response_model=PetResponse)
def get_my_pet(pet_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


@router.put("/{pet_id}", response_model=PetResponse)
def update_my_pet(pet_id: int, patch: PetUpdate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")

    data = patch.model_dump(exclude_unset=True)
    if "species" in data or "activity_level" in data:
        _validate_pet(data.get("species", pet.species), data.get("activity_level", pet.activity_level))

    for k, v in data.items():
        setattr(pet, k, v)

    db.add(pet)
    db.commit()
    db.refresh(pet)
    return pet


@router.delete("/{pet_id}")
def delete_my_pet(pet_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    db.delete(pet)
    db.commit()
    return {"status": "ok"}


# -------------------------
# Legacy endpoints (kept for backward compatibility)
# -------------------------
@router.post("/register", response_model=PetResponse)
def register_pet_legacy(pet: PetCreate, db: Session = Depends(get_db)):
    _validate_pet(pet.species, pet.activity_level)

    if not pet.user_email:
        raise HTTPException(status_code=400, detail="user_email is required for legacy /register endpoint")

    db_pet = Pet(**pet.model_dump())
    db.add(db_pet)
    db.commit()
    db.refresh(db_pet)

    calc = NutritionCalculator()
    daily_calories = calc.calculate_der(
        weight_kg=pet.weight_kg,
        activity_level=pet.activity_level,
        age_years=pet.age_years,
        is_neutered=pet.is_neutered,
    )
    meal_plan = calc.calculate_food_amount(daily_calories, "kibble", age_years=pet.age_years)

    plan = NutritionPlan(
        pet_id=db_pet.id,
        daily_calories=daily_calories,
        meal_frequency=meal_plan["meal_frequency"],
        portion_size_grams=meal_plan["grams_per_meal"],
        food_type="kibble",
        notes="Initial calculation based on profile",
    )
    db.add(plan)
    db.commit()

    return db_pet


@router.get("/user/{email}", response_model=list[PetResponse])
def get_user_pets_legacy(email: str, db: Session = Depends(get_db)):
    return db.query(Pet).filter(Pet.user_email == email).all()
