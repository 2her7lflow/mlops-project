from __future__ import annotations

from .types import PetContext
from ..nutrition_calculator import NutritionCalculator


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
