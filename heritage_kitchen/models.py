from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

HealthGoal = Literal[
    "lower_sodium", "heart_health", "weight_management", "blood_sugar", "high_protein", "more_fiber"
]
Restriction = Literal[
    "vegetarian", "vegan", "pescatarian", "dairy_free", "gluten_free", "halal", "kosher",
    "nut_free", "shellfish_free", "soy_free",
]


class TasteProfile(BaseModel):
    """0-10 sliders, the same ones the Squarespace form renders."""

    spicy: float = Field(5, ge=0, le=10)
    sweet: float = Field(5, ge=0, le=10)
    salty: float = Field(5, ge=0, le=10)
    sour: float = Field(5, ge=0, le=10)
    umami: float = Field(5, ge=0, le=10)
    bitter: float = Field(3, ge=0, le=10)

    def as_list(self) -> List[float]:
        return [self.spicy, self.sweet, self.salty, self.sour, self.umami, self.bitter]


class SubstitutionRequest(BaseModel):
    culture: Optional[str] = Field(None, max_length=80, examples=["Filipino-American"])
    cuisine: Optional[str] = Field(None, max_length=80, examples=["Filipino"])
    dish: Optional[str] = Field(None, max_length=120, examples=["chicken adobo"])
    ingredients: List[str] = Field(default_factory=list, max_length=40)
    taste: Optional[TasteProfile] = None
    health_goals: List[HealthGoal] = Field(default_factory=list)
    restrictions: List[Restriction] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=500)
    include_recipe: bool = True

    @model_validator(mode="after")
    def _needs_something(self) -> "SubstitutionRequest":
        if not any([self.culture, self.cuisine, self.dish, self.ingredients]):
            raise ValueError("Provide at least one of culture, cuisine, dish, or ingredients.")
        return self


class Substitution(BaseModel):
    original: str
    substitute: str
    ratio: str
    reason: str
    flavor_note: str
    cultural_note: str
    nutrition: Optional[Dict] = None


class AdaptedRecipe(BaseModel):
    title: str
    servings: int
    ingredients: List[str]
    steps: List[str]


class CuisineRef(BaseModel):
    key: Optional[str]
    name: str
    region: Optional[str] = None


class SubstitutionResponse(BaseModel):
    cuisine: CuisineRef
    dish: Optional[str]
    source: Literal["llm", "offline"]
    model: Optional[str] = None
    summary: str
    substitutions: List[Substitution]
    adapted_recipe: Optional[AdaptedRecipe] = None
    tips: List[str] = Field(default_factory=list)


class TasteMatchRequest(BaseModel):
    taste: TasteProfile
    top_k: int = Field(5, ge=1, le=34)
