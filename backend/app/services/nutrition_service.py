from __future__ import annotations

import re
from typing import Any

from ..nutrition_calculator import NutritionCalculator

CALC_HINTS = [
    "portion",
    "portion size",
    "how much food",
    "how much should",
    "how much do i feed",
    "how much to feed",
    "feeding amount",
    "daily calories",
    "calories",
    "kcal",
    "grams",
    "gram",
    "cups",
    "cup",
    "meal size",
    "ration",
    "feed per day",
    "ปริมาณอาหาร",
    "กินกี่กรัม",
    "กี่กรัม",
    "กี่แคล",
    "กี่ถ้วย",
    "portion size",
]

FOOD_TYPE_HINTS = {
    "kibble": ["kibble", "dry food", "dry-food", "dry", "เม็ด", "อาหารเม็ด"],
    "wet": ["wet food", "wet-food", "wet", "pouch", "canned", "อาหารเปียก"],
    "raw": ["raw", "barf", "อาหารดิบ"],
    "mixed": ["mixed", "mix", "ผสม"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_calculation_question(question: str) -> bool:
    q = _normalize(question)
    return any(term in q for term in CALC_HINTS)


def infer_food_type(question: str, default: str = "kibble") -> str:
    q = _normalize(question)
    for food_type, hints in FOOD_TYPE_HINTS.items():
        if any(hint in q for hint in hints):
            return food_type
    return default


def infer_activity_level(question: str, default: str = "moderate") -> str:
    q = _normalize(question)
    if any(x in q for x in ["very active", "highly active", "athletic", "หนักมาก"]):
        return "very_active"
    if any(x in q for x in ["active", "energetic", "exercise a lot", "แอคทีฟ", "กิจกรรมเยอะ"]):
        return "active"
    if any(x in q for x in ["sedentary", "inactive", "lazy", "couch", "ไม่ค่อยออกกำลังกาย"]):
        return "sedentary"
    if any(x in q for x in ["senior", "older", "สูงอายุ"]):
        return "senior"
    return default


def extract_pet_facts(question: str) -> dict[str, Any]:
    q = _normalize(question)

    species = None
    if any(x in q for x in ["cat", "cats", "kitten", "feline", "แมว"]):
        species = "cat"
    elif any(x in q for x in ["dog", "dogs", "puppy", "canine", "หมา", "สุนัข"]):
        species = "dog"

    weight_kg = None
    kg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram|kilograms|กก\.?|กิโล|กิโลกรัม)", q)
    if kg_match:
        weight_kg = float(kg_match.group(1))
    else:
        lb_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)", q)
        if lb_match:
            weight_kg = round(float(lb_match.group(1)) * 0.45359237, 2)

    age_years = None
    year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs|y/o|yo|ปี)", q)
    if year_match:
        age_years = float(year_match.group(1))
    else:
        month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:month|months|mo|mos|เดือน)", q)
        if month_match:
            age_years = round(float(month_match.group(1)) / 12.0, 2)

    if any(x in q for x in ["neutered", "spayed", "fixed", "ทำหมัน"]):
        is_neutered = True
    elif any(x in q for x in ["not neutered", "intact", "ยังไม่ทำหมัน"]):
        is_neutered = False
    else:
        is_neutered = None

    return {
        "species": species,
        "weight_kg": weight_kg,
        "age_years": age_years,
        "is_neutered": is_neutered,
        "activity_level": infer_activity_level(question),
        "food_type": infer_food_type(question),
    }


def calculate_plan(
    weight_kg: float,
    activity_level: str,
    age_years: float,
    is_neutered: bool,
    food_type: str = "kibble",
) -> tuple[float, dict]:
    calc = NutritionCalculator()
    daily_cal = calc.calculate_der(
        weight_kg=weight_kg,
        activity_level=activity_level,
        age_years=age_years,
        is_neutered=is_neutered,
    )
    meal_plan = calc.calculate_food_amount(daily_cal, food_type=food_type, age_years=age_years)
    return daily_cal, meal_plan


def adjust_plan_for_activity(base_calories: float, steps: int, active_minutes: int) -> tuple[float, float]:
    """Return (adjusted_calories, adjustment_percent)."""
    calc = NutritionCalculator()
    adjusted = calc.adjust_for_activity(base_calories, steps, active_minutes)
    pct = ((adjusted - base_calories) / base_calories) * 100 if base_calories else 0.0
    return adjusted, pct
