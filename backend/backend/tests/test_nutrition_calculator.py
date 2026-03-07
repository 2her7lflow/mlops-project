import pytest

from app.nutrition_calculator import NutritionCalculator


def test_rer_positive():
    rer = NutritionCalculator.calculate_rer(10.0)
    assert rer > 0


def test_der_puppy_multiplier():
    # < 4 months -> puppy_4mo multiplier 3.0
    der = NutritionCalculator.calculate_der(weight_kg=5.0, activity_level="moderate", age_years=0.2)
    rer = NutritionCalculator.calculate_rer(5.0)
    assert der == round(rer * 3.0, 2)


def test_der_senior_multiplier():
    der = NutritionCalculator.calculate_der(weight_kg=10.0, activity_level="active", age_years=9.0)
    rer = NutritionCalculator.calculate_rer(10.0)
    assert der == round(rer * NutritionCalculator.ACTIVITY_MULTIPLIERS["senior"], 2)


def test_adjust_for_activity_caps():
    base = 1000.0
    # extreme activity should cap at +30%
    adj_high = NutritionCalculator.adjust_for_activity(base, steps=50000, active_minutes=300)
    assert adj_high <= 1300.0 + 1e-6

    # very low activity should cap at -30%
    adj_low = NutritionCalculator.adjust_for_activity(base, steps=0, active_minutes=0)
    assert adj_low >= 700.0 - 1e-6


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        NutritionCalculator.calculate_rer(0)
