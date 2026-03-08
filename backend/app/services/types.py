from typing import Optional, TypedDict


class PetContext(TypedDict, total=False):
    name: str
    breed: str
    age_years: float
    weight_kg: float
    health_conditions: Optional[str]
    allergies: Optional[str]
