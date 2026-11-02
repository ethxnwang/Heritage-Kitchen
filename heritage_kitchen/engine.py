"""Turns a user's request into substitutions.

The flow is: resolve the cuisine, figure out which ingredients to work on,
score candidate swaps with the nutrition table, hand all of that to Claude,
then attach real nutrition deltas to whatever comes back. If the model is
unavailable the ranked candidates are returned directly.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .catalog import Catalog, Cuisine, Ingredient, load_catalog
from .config import Settings, get_settings
from .llm import LLMError, SubstitutionLLM
from .models import (
    AdaptedRecipe,
    CuisineRef,
    Substitution,
    SubstitutionRequest,
    SubstitutionResponse,
)
from .nutrition import RankedSwap, nutrition_delta, rank_substitutes
from .prompts import build_user_prompt

log = logging.getLogger(__name__)

RATIO_BY_CATEGORY = {
    "fat": "Use about 3/4 of the original amount",
    "cream": "1:1 by volume; stir in off the heat so it doesn't split",
    "cheese": "1:1 by weight",
    "grain": "1:1 cooked volume",
    "flour": "Replace up to half the flour 1:1 by weight",
    "bread": "1:1",
    "noodle": "1:1 cooked weight",
    "starch_veg": "1:1 by weight",
    "sweetener": "Start with 2/3 of the original amount and adjust",
    "seasoning": "1:1, then adjust to taste at the end",
    "protein_meat": "1:1 by weight",
    "protein_plant": "1:1 by weight",
    "vegetable": "1:1",
}

GOAL_TIPS = {
    "lower_sodium": "Add salt at the end of cooking, after acids and aromatics have done their work; you'll need less.",
    "heart_health": "Sear in a hot pan with a thin film of oil rather than shallow frying to keep the browning with less fat.",
    "weight_management": "Bulk the dish out with an extra vegetable the cuisine already uses so portions stay generous.",
    "blood_sugar": "Pair the starch with protein and fiber, and let cooked rice or noodles cool before reheating.",
    "high_protein": "Fold in beans, lentils, tofu, or lean meat that the tradition already uses.",
    "more_fiber": "Swap in whole grains gradually, half and half at first, so the texture change isn't jarring.",
}


def taste_vector(req: SubstitutionRequest) -> Optional[np.ndarray]:
    if req.taste is None:
        return None
    return np.asarray(req.taste.as_list(), dtype=float) / 10.0


class SubstitutionEngine:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        catalog: Optional[Catalog] = None,
        llm: Optional[SubstitutionLLM] = None,
    ):
        self.settings = settings or get_settings()
        self.catalog = catalog or load_catalog()
        if llm is not None:
            self.llm: Optional[SubstitutionLLM] = llm
        elif not self.settings.offline and self.settings.has_credentials:
            self.llm = SubstitutionLLM(self.settings)
        else:
            self.llm = None

    def resolve_cuisine(self, req: SubstitutionRequest) -> Optional[Cuisine]:
        for text in (req.cuisine, req.culture):
            if text:
                found = self.catalog.cuisine(text)
                if found:
                    return found
        if req.dish:
            return self.catalog.cuisine_for_dish(req.dish)
        return None

    def target_ingredients(
        self, req: SubstitutionRequest, cuisine: Optional[Cuisine]
    ) -> List[Tuple[str, Optional[Ingredient]]]:
        if req.ingredients:
            return [(text, self.catalog.ingredient(text)) for text in req.ingredients]
        if cuisine is not None:
            return [
                (self.catalog.ingredients[k].name, self.catalog.ingredients[k])
                for k in cuisine.heavy_ingredients
                if k in self.catalog.ingredients
            ]
        return []

    def candidates(
        self, req: SubstitutionRequest, cuisine: Optional[Cuisine]
    ) -> Dict[str, List[RankedSwap]]:
        taste = taste_vector(req)
        out: Dict[str, List[RankedSwap]] = {}
        for text, ing in self.target_ingredients(req, cuisine):
            if ing is None or ing.key in out:
                continue
            swaps = rank_substitutes(
                self.catalog, ing, cuisine=cuisine, taste=taste,
                goals=req.health_goals, restrictions=req.restrictions,
            )
            if swaps:
                out[ing.key] = swaps
        return out

    def run(self, req: SubstitutionRequest) -> SubstitutionResponse:
        cuisine = self.resolve_cuisine(req)
        candidates = self.candidates(req, cuisine)

        if self.llm is not None:
            try:
                data = self.llm.generate(build_user_prompt(req, cuisine, candidates))
                return self._from_llm(req, cuisine, data)
            except LLMError as exc:
                log.warning("falling back to offline ranking: %s", exc)

        return self._offline(req, cuisine, candidates)

    def _attach_nutrition(self, original: str, substitute: str, goals) -> Optional[Dict]:
        a = self.catalog.ingredient(original)
        b = self.catalog.ingredient(substitute)
        if a is None or b is None or a.key == b.key:
            return None
        return nutrition_delta(a, b, goals)

    def _cuisine_ref(self, cuisine: Optional[Cuisine], fallback_name: str) -> CuisineRef:
        if cuisine is not None:
            return CuisineRef(key=cuisine.key, name=cuisine.name, region=cuisine.region)
        return CuisineRef(key=None, name=fallback_name or "Unspecified")

    def _from_llm(
        self, req: SubstitutionRequest, cuisine: Optional[Cuisine], data: Dict
    ) -> SubstitutionResponse:
        subs = []
        for item in data.get("substitutions", []):
            subs.append(Substitution(
                **{k: item.get(k, "") for k in
                   ("original", "substitute", "ratio", "reason", "flavor_note", "cultural_note")},
                nutrition=self._attach_nutrition(item.get("original", ""), item.get("substitute", ""), req.health_goals),
            ))

        recipe = None
        raw_recipe = data.get("adapted_recipe") or {}
        if req.include_recipe and raw_recipe.get("title") and raw_recipe.get("ingredients"):
            recipe = AdaptedRecipe(**raw_recipe)

        return SubstitutionResponse(
            cuisine=self._cuisine_ref(cuisine, data.get("cuisine", "")),
            dish=data.get("dish") or req.dish,
            source="llm",
            model=data.get("_model"),
            summary=data.get("summary", ""),
            substitutions=subs,
            adapted_recipe=recipe,
            tips=data.get("tips", []),
        )

    def _offline(
        self,
        req: SubstitutionRequest,
        cuisine: Optional[Cuisine],
        candidates: Dict[str, List[RankedSwap]],
    ) -> SubstitutionResponse:
        subs = []
        for swaps in candidates.values():
            best = swaps[0]
            subs.append(Substitution(
                original=best.original.name,
                substitute=best.substitute.name,
                ratio=RATIO_BY_CATEGORY.get(best.original.category, "1:1"),
                reason=self._offline_reason(best, req.health_goals),
                flavor_note=self._flavor_note(best.flavor_match),
                cultural_note=self._cultural_note(best, cuisine),
                nutrition=nutrition_delta(best.original, best.substitute, req.health_goals),
            ))

        cuisine_name = cuisine.name if cuisine else (req.cuisine or req.culture or "your")
        subject = req.dish or f"{cuisine_name} home cooking"
        summary = (
            f"{len(subs)} swap{'s' if len(subs) != 1 else ''} for {subject}, ranked by nutrition gain, "
            f"flavor similarity, and how common each substitute already is in {cuisine_name} kitchens."
            if subs else
            f"No healthier swaps were found in the reference table for {subject}."
        )
        tips = [GOAL_TIPS[g] for g in req.health_goals if g in GOAL_TIPS]

        return SubstitutionResponse(
            cuisine=self._cuisine_ref(cuisine, req.cuisine or req.culture or ""),
            dish=req.dish,
            source="offline",
            summary=summary,
            substitutions=subs,
            adapted_recipe=None,
            tips=tips,
        )

    @staticmethod
    def _offline_reason(swap: RankedSwap, goals) -> str:
        delta = nutrition_delta(swap.original, swap.substitute, goals)["change_pct"]
        # (field, label, index, minimum per-100 g amount worth mentioning)
        limits = [
            ("kcal", "fewer calories", 0, 50),
            ("sat_fat_g", "less saturated fat", 1, 1),
            ("sodium_mg", "less sodium", 2, 100),
            ("sugar_g", "less sugar", 3, 2),
        ]
        drops = [
            f"{abs(delta[f]):.0f}% {label}"
            for f, label, i, floor in limits
            if delta.get(f) is not None and delta[f] <= -15 and swap.original.nutrients[i] >= floor
        ]
        gains = []
        if swap.substitute.nutrients[4] - swap.original.nutrients[4] >= 1:
            gains.append("more fiber")
        if swap.substitute.nutrients[5] - swap.original.nutrients[5] >= 3:
            gains.append("more protein")
        parts = drops[:3] + gains
        return ("Per 100 g: " + ", ".join(parts) + ".") if parts else "A modest overall nutrition improvement."

    @staticmethod
    def _flavor_note(match: float) -> str:
        if match >= 0.9:
            return "Very close flavor match."
        if match >= 0.75:
            return "Similar flavor; season a little more boldly to compensate."
        return "Noticeably different flavor; lean on the dish's usual aromatics and spices."

    @staticmethod
    def _cultural_note(swap: RankedSwap, cuisine: Optional[Cuisine]) -> str:
        if cuisine is None:
            return "Check that this fits how your family makes the dish."
        if swap.cultural_fit:
            return f"{swap.substitute.name} is already a staple in {cuisine.name} kitchens."
        return f"Less traditional in {cuisine.name} cooking, so try it in half the batch first."
