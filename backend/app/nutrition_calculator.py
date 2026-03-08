import math
from typing import Dict, Optional


class NutritionCalculator:
    """Calculate pet nutrition needs based on scientific formulas"""

    # Activity multipliers for DER calculation
    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,      # Neutered adult, minimal activity
        "moderate": 1.4,       # Normal adult activity
        "active": 1.6,         # Working dogs, active play
        "very_active": 1.8,    # Athletic dogs, intense exercise
        "puppy_4mo": 3.0,      # Puppies up to 4 months
        "puppy_12mo": 2.0,     # Puppies 4-12 months
        "senior": 1.1,         # Senior dogs (>7 years)
        "pregnant": 1.8,       # Pregnant females
        "lactating": 3.0,      # Nursing mothers
    }

    @staticmethod
    def _validate_weight(weight_kg: float) -> None:
        if weight_kg is None or weight_kg <= 0:
            raise ValueError("weight_kg must be a positive number")

    @staticmethod
    def calculate_rer(weight_kg: float) -> float:
        """
        Calculate Resting Energy Requirement (RER)
        Formula: RER = 70 * (weight_kg ^ 0.75)
        """
        NutritionCalculator._validate_weight(weight_kg)
        return 70.0 * math.pow(weight_kg, 0.75)

    @staticmethod
    def calculate_der(
        weight_kg: float,
        activity_level: str,
        age_years: Optional[float] = None,
        is_neutered: bool = False,
    ) -> float:
        """
        Calculate Daily Energy Requirement (DER)
        DER = RER × Activity Multiplier
        """
        rer = NutritionCalculator.calculate_rer(weight_kg)

        # Determine activity multiplier
        if age_years is not None:
            if age_years < 0.33:  # < 4 months
                multiplier = NutritionCalculator.ACTIVITY_MULTIPLIERS["puppy_4mo"]
            elif age_years < 1.0:  # 4-12 months
                multiplier = NutritionCalculator.ACTIVITY_MULTIPLIERS["puppy_12mo"]
            elif age_years > 7.0:  # Senior
                multiplier = NutritionCalculator.ACTIVITY_MULTIPLIERS["senior"]
            else:
                multiplier = NutritionCalculator.ACTIVITY_MULTIPLIERS.get(activity_level, 1.4)

                # Adjust for neutered status (small reduction)
                if is_neutered and multiplier > 1.2:
                    multiplier = max(1.2, multiplier - 0.1)
        else:
            multiplier = NutritionCalculator.ACTIVITY_MULTIPLIERS.get(activity_level, 1.4)

        der = rer * multiplier
        return round(der, 2)

    @staticmethod
    def calculate_food_amount(
        daily_calories: float,
        food_type: str = "kibble",
        age_years: Optional[float] = None,
    ) -> Dict:
        """
        Calculate food amount based on food type

        Typical kcal per 100g (rough averages):
        - Dry kibble: ~370
        - Wet food: ~90
        - Raw diet: ~175
        - Mixed: ~250
        """
        if daily_calories is None or daily_calories <= 0:
            raise ValueError("daily_calories must be a positive number")

        calories_per_100g = {
            "kibble": 370,
            "wet": 90,
            "raw": 175,
            "mixed": 250,
        }

        cal_density = calories_per_100g.get(food_type, 370)
        grams_per_day = (daily_calories / cal_density) * 100.0

        # Meal frequency (more realistic)
        if age_years is not None and age_years < 1.0:
            meals = 4 if age_years < 0.33 else 3  # young puppies eat more often
        else:
            if daily_calories >= 1200:
                meals = 3
            else:
                meals = 2

        portion_per_meal = grams_per_day / meals

        return {
            "total_grams_per_day": round(grams_per_day, 1),
            "meal_frequency": meals,
            "grams_per_meal": round(portion_per_meal, 1),
            "cups_per_day": round(grams_per_day / 120.0, 2),  # ~120g per cup assumption
        }


    @staticmethod
    def get_recommendations(species: str, age_years: Optional[float] = None, health_conditions: Optional[str] = None) -> str:
        """Return lightweight nutrition guidance text for calculator responses."""
        species = (species or "pet").lower()
        notes = []

        if species == "cat":
            notes.append("Cats do best on a high-protein diet and should always have access to fresh water.")
        elif species == "dog":
            notes.append("Dogs usually do well when their daily ration is split into consistent meals and body weight is checked regularly.")
        else:
            notes.append("Use the portion as a starting estimate and monitor body weight and body condition score.")

        if age_years is not None:
            if age_years < 1.0:
                notes.append("Young pets typically need more frequent meals and growth-appropriate food.")
            elif age_years > 7.0:
                notes.append("Senior pets may need lower calories and closer monitoring of muscle condition.")

        if health_conditions:
            notes.append("Because there are existing health conditions, confirm the final feeding plan with a veterinarian.")

        return " ".join(notes)

    @staticmethod
    def adjust_for_activity(base_calories: float, steps: int, active_minutes: int) -> float:
        """
        Adjust daily calories based on activity tracking

        Model:
        - 10,000 steps baseline
        - every 2,500 steps diff ≈ ±5%
        - 30 active minutes baseline
        - every 15 minutes diff ≈ ±5%

        Final cap: ±30%
        """
        if base_calories is None or base_calories <= 0:
            raise ValueError("base_calories must be a positive number")

        steps = max(0, int(steps))
        active_minutes = max(0, int(active_minutes))

        # Steps adjustment (cap this component to ±20%)
        step_diff = (steps - 10000) / 2500.0
        step_multiplier = 1.0 + (step_diff * 0.05)
        step_multiplier = max(0.8, min(1.2, step_multiplier))

        # Active minutes adjustment (cap this component to ±20%)
        active_diff = (active_minutes - 30) / 15.0
        active_multiplier = 1.0 + (active_diff * 0.05)
        active_multiplier = max(0.8, min(1.2, active_multiplier))

        # Combined multiplier (cap at ±30%)
        total_multiplier = step_multiplier * active_multiplier
        total_multiplier = max(0.7, min(1.3, total_multiplier))

        adjusted_calories = base_calories * total_multiplier
        return round(adjusted_calories, 2)
