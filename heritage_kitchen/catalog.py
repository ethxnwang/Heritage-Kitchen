"""Cuisine and ingredient reference data, plus fuzzy lookup helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

FLAVOR_AXES = ("spicy", "sweet", "salty", "sour", "umami", "bitter")
NUTRIENT_FIELDS = ("kcal", "sat_fat_g", "sodium_mg", "sugar_g", "fiber_g", "protein_g")


@dataclass(frozen=True)
class Ingredient:
    key: str
    name: str
    category: str
    nutrients: np.ndarray
    flavor: np.ndarray
    aliases: Tuple[str, ...] = ()

    def nutrient_dict(self) -> Dict[str, float]:
        return {f: round(float(v), 1) for f, v in zip(NUTRIENT_FIELDS, self.nutrients)}


@dataclass(frozen=True)
class Cuisine:
    key: str
    name: str
    region: str
    flavor: np.ndarray
    staples: Tuple[str, ...]
    heavy_ingredients: Tuple[str, ...]
    dishes: Tuple[str, ...]
    cultures: Tuple[str, ...] = field(default_factory=tuple)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.sub(r"[\s\-]+", " ", text).strip()


class Catalog:
    def __init__(self, ingredients: Dict[str, Ingredient], cuisines: Dict[str, Cuisine]):
        self.ingredients = ingredients
        self.cuisines = cuisines

        self._ingredient_index: Dict[str, str] = {}
        for key, ing in ingredients.items():
            for label in (key.replace("_", " "), ing.name, *ing.aliases):
                self._ingredient_index.setdefault(normalize(label), key)
        # strip parentheticals like "(cooked)" so "white rice" still hits
        for key, ing in ingredients.items():
            self._ingredient_index.setdefault(normalize(re.sub(r"\(.*?\)", "", ing.name)), key)

        self._cuisine_index: Dict[str, str] = {}
        self._dish_index: Dict[str, str] = {}
        for key, c in cuisines.items():
            for label in (key.replace("_", " "), c.name, *c.cultures):
                self._cuisine_index.setdefault(normalize(label), key)
            for dish in c.dishes:
                self._dish_index.setdefault(normalize(dish), key)

        self.cuisine_keys: List[str] = list(cuisines)
        self.cuisine_flavors = np.vstack([cuisines[k].flavor for k in self.cuisine_keys])

    def ingredient(self, text: str) -> Optional[Ingredient]:
        q = normalize(text)
        if not q:
            return None
        if q in self._ingredient_index:
            return self.ingredients[self._ingredient_index[q]]
        if q.endswith("s") and q[:-1] in self._ingredient_index:
            return self.ingredients[self._ingredient_index[q[:-1]]]
        match = get_close_matches(q, self._ingredient_index.keys(), n=1, cutoff=0.85)
        if match:
            return self.ingredients[self._ingredient_index[match[0]]]
        # "2 cups shredded cheddar cheese" -> longest known label contained in the text
        hits = [label for label in self._ingredient_index if re.search(rf"\b{re.escape(label)}\b", q)]
        if hits:
            return self.ingredients[self._ingredient_index[max(hits, key=len)]]
        return None

    def cuisine(self, text: str) -> Optional[Cuisine]:
        q = normalize(text)
        if not q:
            return None
        if q in self._cuisine_index:
            return self.cuisines[self._cuisine_index[q]]
        match = get_close_matches(q, self._cuisine_index.keys(), n=1, cutoff=0.8)
        if match:
            return self.cuisines[self._cuisine_index[match[0]]]
        for label, key in sorted(self._cuisine_index.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(label)}\b", q):
                return self.cuisines[key]
        return None

    def cuisine_for_dish(self, dish: str) -> Optional[Cuisine]:
        q = normalize(dish)
        if not q:
            return None
        if q in self._dish_index:
            return self.cuisines[self._dish_index[q]]
        match = get_close_matches(q, self._dish_index.keys(), n=1, cutoff=0.75)
        if match:
            return self.cuisines[self._dish_index[match[0]]]
        for label, key in self._dish_index.items():
            if label in q or q in label:
                return self.cuisines[key]
        return None


def _load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    raw_ing = _load_json("ingredients.json")["ingredients"]
    raw_cui = _load_json("cuisines.json")["cuisines"]

    ingredients = {
        key: Ingredient(
            key=key,
            name=v["name"],
            category=v["category"],
            nutrients=np.asarray(v["nutrients"], dtype=float),
            flavor=np.asarray(v["flavor"], dtype=float),
            aliases=tuple(v.get("aliases", ())),
        )
        for key, v in raw_ing.items()
    }
    cuisines = {
        key: Cuisine(
            key=key,
            name=v["name"],
            region=v["region"],
            flavor=np.asarray(v["flavor"], dtype=float),
            staples=tuple(v["staples"]),
            heavy_ingredients=tuple(v["heavy_ingredients"]),
            dishes=tuple(v["dishes"]),
            cultures=tuple(v.get("cultures", ())),
        )
        for key, v in raw_cui.items()
    }
    return Catalog(ingredients, cuisines)
