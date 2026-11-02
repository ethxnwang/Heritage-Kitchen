"""Nutrition scoring and substitute ranking.

Everything here works on per-100 g nutrient vectors in the order
(kcal, sat_fat_g, sodium_mg, sugar_g, fiber_g, protein_g).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .catalog import NUTRIENT_FIELDS, Catalog, Cuisine, Ingredient

# FDA daily values for a 2,000 kcal diet
DAILY_VALUES = np.array([2000.0, 20.0, 2300.0, 50.0, 28.0, 50.0])

# negative = limit, positive = encourage
BASE_WEIGHTS = np.array([-1.0, -1.5, -1.2, -1.2, 1.5, 0.8])

# extra emphasis per health goal, applied on top of BASE_WEIGHTS
GOAL_WEIGHTS: Dict[str, np.ndarray] = {
    "lower_sodium":       np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0]),
    "heart_health":       np.array([0.0, -1.5, -1.0, 0.0, 0.5, 0.0]),
    "weight_management":  np.array([-1.5, -0.5, 0.0, -0.5, 0.5, 0.5]),
    "blood_sugar":        np.array([-0.3, 0.0, 0.0, -2.0, 1.0, 0.3]),
    "high_protein":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5]),
    "more_fiber":         np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.0]),
}

COMPATIBLE_CATEGORIES: Dict[str, Sequence[str]] = {
    "fat": ("fat",),
    "cream": ("cream",),
    "cheese": ("cheese", "cream"),
    "grain": ("grain", "starch_veg"),
    "flour": ("flour",),
    "bread": ("bread",),
    "noodle": ("noodle",),
    "starch_veg": ("starch_veg", "grain"),
    "sweetener": ("sweetener",),
    "seasoning": ("seasoning",),
    "protein_meat": ("protein_meat", "protein_plant"),
    "protein_plant": ("protein_plant",),
    "vegetable": ("vegetable",),
}

# restriction -> ingredient keys that violate it
RESTRICTION_EXCLUDES: Dict[str, frozenset] = {
    "vegetarian": frozenset({
        "lard", "schmaltz", "pork_belly", "pork_sausage", "chorizo", "bacon", "ground_beef",
        "lean_ground_beef", "beef_ribeye", "sirloin", "pork_tenderloin", "chicken_breast",
        "turkey_ground", "chicken_sausage", "turkey_bacon", "salmon", "shrimp", "fish_sauce",
        "reduced_sodium_fish_sauce", "oyster_sauce",
    }),
    "pescatarian": frozenset({
        "lard", "schmaltz", "pork_belly", "pork_sausage", "chorizo", "bacon", "ground_beef",
        "lean_ground_beef", "beef_ribeye", "sirloin", "pork_tenderloin", "chicken_breast",
        "turkey_ground", "chicken_sausage", "turkey_bacon",
    }),
    "dairy_free": frozenset({
        "butter", "ghee", "heavy_cream", "sour_cream", "condensed_milk", "evaporated_skim_milk",
        "low_fat_milk", "cream_cheese", "greek_yogurt", "cheddar_cheese", "reduced_fat_cheddar",
        "mozzarella", "part_skim_mozzarella", "paneer", "feta", "part_skim_ricotta", "cottage_cheese",
    }),
    "gluten_free": frozenset({
        "white_bread", "whole_wheat_bread", "flour_tortilla", "whole_wheat_tortilla", "white_pasta",
        "whole_wheat_pasta", "egg_noodles", "instant_ramen", "all_purpose_flour", "whole_wheat_flour",
        "bulgur", "barley", "white_couscous", "whole_wheat_couscous", "soy_sauce",
        "low_sodium_soy_sauce", "oyster_sauce",
    }),
    "halal": frozenset({"lard", "pork_belly", "pork_sausage", "chorizo", "bacon", "pork_tenderloin"}),
    "kosher": frozenset({"lard", "pork_belly", "pork_sausage", "chorizo", "bacon", "pork_tenderloin", "shrimp"}),
    "nut_free": frozenset({"cashew_cream"}),
    "shellfish_free": frozenset({"shrimp", "oyster_sauce"}),
    "soy_free": frozenset({"tofu_firm", "tempeh", "soy_sauce", "low_sodium_soy_sauce"}),
}
RESTRICTION_EXCLUDES["vegan"] = (
    RESTRICTION_EXCLUDES["vegetarian"]
    | RESTRICTION_EXCLUDES["dairy_free"]
    | {"salmon", "shrimp"}
)


def goal_weights(goals: Iterable[str]) -> np.ndarray:
    w = BASE_WEIGHTS.copy()
    for g in goals:
        w = w + GOAL_WEIGHTS.get(g, 0.0)
    return w


def health_scores(nutrients: np.ndarray, goals: Iterable[str] = ()) -> np.ndarray:
    """Score one (6,) vector or an (n, 6) matrix. Higher is healthier."""
    matrix = np.atleast_2d(np.asarray(nutrients, dtype=float))
    scores = (matrix / DAILY_VALUES) @ goal_weights(goals)
    return scores if np.ndim(nutrients) > 1 else scores[:1]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector/matrix ``a`` and a vector/matrix ``b``."""
    a = np.atleast_2d(a).astype(float)
    b = np.atleast_2d(b).astype(float)
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return (a / na) @ (b / nb).T


def nutrition_delta(original: Ingredient, substitute: Ingredient, goals: Iterable[str] = ()) -> Dict:
    diff = substitute.nutrients - original.nutrients
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(original.nutrients > 0, diff / original.nutrients * 100.0, np.nan)
    scores = health_scores(np.vstack([original.nutrients, substitute.nutrients]), goals)
    return {
        "basis": "per 100 g",
        "original": original.nutrient_dict(),
        "substitute": substitute.nutrient_dict(),
        "change_pct": {
            f: (None if np.isnan(p) else round(float(p), 1)) for f, p in zip(NUTRIENT_FIELDS, pct)
        },
        "health_score_delta": round(float(scores[1] - scores[0]), 3),
    }


@dataclass
class RankedSwap:
    original: Ingredient
    substitute: Ingredient
    score: float
    health_gain: float
    flavor_match: float
    cultural_fit: float


def rank_substitutes(
    catalog: Catalog,
    original: Ingredient,
    cuisine: Optional[Cuisine] = None,
    taste: Optional[np.ndarray] = None,
    goals: Sequence[str] = (),
    restrictions: Sequence[str] = (),
    top_k: int = 3,
) -> List[RankedSwap]:
    banned = set()
    for r in restrictions:
        banned |= RESTRICTION_EXCLUDES.get(r, frozenset())

    allowed_cats = COMPATIBLE_CATEGORIES.get(original.category, (original.category,))
    pool = [
        ing for ing in catalog.ingredients.values()
        if ing.category in allowed_cats and ing.key != original.key and ing.key not in banned
    ]
    if not pool:
        return []

    nutrients = np.vstack([p.nutrients for p in pool])
    flavors = np.vstack([p.flavor for p in pool])

    base = health_scores(original.nutrients, goals)[0]
    gain = health_scores(nutrients, goals) - base

    flavor_match = cosine(flavors, original.flavor).ravel()
    if taste is not None and np.any(taste):
        flavor_match = 0.7 * flavor_match + 0.3 * cosine(flavors, taste).ravel()

    cultural = np.zeros(len(pool))
    if cuisine is not None:
        staples = set(cuisine.staples)
        cultural = np.array([1.0 if p.key in staples else 0.0 for p in pool])

    # cross-category swaps (meat -> beans) are fine but should lose ties
    same_cat = np.array([1.0 if p.category == original.category else 0.85 for p in pool])

    healthier = gain > 0
    if not np.any(healthier):
        return []

    g = gain[healthier]
    g_norm = g / g.max()
    combined = (0.40 * g_norm + 0.30 * flavor_match[healthier] + 0.30 * cultural[healthier]) * same_cat[healthier]

    idx = np.flatnonzero(healthier)
    order = np.argsort(-combined)[:top_k]
    return [
        RankedSwap(
            original=original,
            substitute=pool[idx[i]],
            score=round(float(combined[i]), 3),
            health_gain=round(float(g[i]), 3),
            flavor_match=round(float(flavor_match[idx[i]]), 3),
            cultural_fit=float(cultural[idx[i]]),
        )
        for i in order
    ]


def match_cuisines(catalog: Catalog, taste: np.ndarray, top_k: int = 5) -> List[Dict]:
    """Cuisines whose flavor fingerprint is closest to a user's taste vector."""
    sims = cosine(catalog.cuisine_flavors, taste).ravel()
    order = np.argsort(-sims)[:top_k]
    return [
        {
            "key": catalog.cuisine_keys[i],
            "name": catalog.cuisines[catalog.cuisine_keys[i]].name,
            "similarity": round(float(sims[i]), 3),
        }
        for i in order
    ]
