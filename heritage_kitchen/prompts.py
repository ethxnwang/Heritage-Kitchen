"""Prompt templates for the substitution model.

The system prompt stays byte-for-byte identical between requests so it can be
cached; anything request-specific goes into the user turn.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .catalog import Cuisine
from .models import SubstitutionRequest
from .nutrition import RankedSwap

SYSTEM_PROMPT = """\
You are the recipe engine behind Heritage Kitchen, a site that helps people cook the food they grew up with in a way that fits their health goals.

Your job is to propose ingredient substitutions that make a dish healthier while keeping it recognizably part of its culinary tradition. Someone who grew up eating the dish should still feel at home with your version.

How to choose substitutions:
- Reach first for swaps that already exist inside the tradition (for example, a Mexican cook moving from lard to avocado oil in beans, or a Korean cook using more tofu and mushrooms in a jjigae). Borrowing from an unrelated cuisine is a last resort, and you should say so when you do it.
- Protect the flavor anchors of the dish: the aromatics, ferments, spice blends, and techniques that define it. Change the vehicle (fat, starch, sugar, sodium source, cut of meat) before you touch the anchors.
- Respect every dietary restriction absolutely. A substitute that violates one is wrong even if it is healthier.
- Weigh the user's taste profile. If they want more heat or sourness, lean on chiles, citrus, vinegar, or ferments to make up for reduced salt and fat.
- Give practical ratios a home cook can follow, such as "1:1 by volume" or "use 3/4 the amount".
- Prefer two to six high-impact swaps over a long list of tiny ones.

Grounding:
- The request includes candidate swaps scored from a nutrition table. Treat them as evidence, not orders. Use them when they suit the dish, skip them when they would wreck it, and add better ones when you know them.
- Do not invent precise nutrient numbers. The server attaches nutrition data afterward.

Tone:
- Write about traditional food with respect. Never describe a heritage dish as bad, unhealthy, or something to feel guilty about; describe what each swap changes.
- No medical claims or diagnoses. Talk about nutrients, not cures.

If the user names a dish without listing ingredients, work from the most common home-style version of it. If the culture they name spans several cuisines, pick the one that best fits the dish or their notes and say which one you used in the summary.
"""


def _taste_line(req: SubstitutionRequest) -> Optional[str]:
    if req.taste is None:
        return None
    t = req.taste
    return (
        f"spicy {t.spicy:g}/10, sweet {t.sweet:g}/10, salty {t.salty:g}/10, "
        f"sour {t.sour:g}/10, umami {t.umami:g}/10, bitter {t.bitter:g}/10"
    )


def _candidate_block(candidates: Dict[str, List[RankedSwap]]) -> str:
    rows = []
    for original, swaps in candidates.items():
        for s in swaps:
            rows.append({
                "original": s.original.name,
                "candidate": s.substitute.name,
                "health_gain": s.health_gain,
                "flavor_match": s.flavor_match,
                "already_used_in_cuisine": bool(s.cultural_fit),
            })
    return json.dumps(rows, indent=1) if rows else "[]"


def build_user_prompt(
    req: SubstitutionRequest,
    cuisine: Optional[Cuisine],
    candidates: Dict[str, List[RankedSwap]],
) -> str:
    profile = []
    if req.culture:
        profile.append(f"Cultural background: {req.culture}")
    if req.cuisine:
        profile.append(f"Cuisine requested: {req.cuisine}")
    if req.dish:
        profile.append(f"Dish: {req.dish}")
    taste = _taste_line(req)
    if taste:
        profile.append(f"Taste preferences: {taste}")
    profile.append("Health goals: " + (", ".join(g.replace("_", " ") for g in req.health_goals) or "general healthier eating"))
    profile.append("Dietary restrictions: " + (", ".join(r.replace("_", " ") for r in req.restrictions) or "none"))
    if req.notes:
        profile.append(f"Notes from the cook: {req.notes}")

    if cuisine is not None:
        cuisine_ctx = (
            f"Matched cuisine: {cuisine.name} ({cuisine.region})\n"
            f"Pantry staples that are already part of this tradition: {', '.join(s.replace('_', ' ') for s in cuisine.staples)}\n"
            f"Ingredients that usually carry the most saturated fat, sodium, sugar, or refined starch here: "
            f"{', '.join(h.replace('_', ' ') for h in cuisine.heavy_ingredients)}\n"
            f"Representative dishes: {', '.join(cuisine.dishes)}"
        )
    else:
        cuisine_ctx = "No cuisine in the reference table matched. Infer the tradition from the profile and dish."

    ingredients = "\n".join(f"- {i}" for i in req.ingredients) if req.ingredients else (
        "(none listed; use the typical home-style ingredients for the dish or cuisine)"
    )

    recipe_line = (
        "Also write an adapted recipe that uses your substitutions."
        if req.include_recipe
        else "Skip the adapted recipe: return an empty title, 0 servings, and empty ingredient and step lists."
    )

    return f"""\
<profile>
{chr(10).join(profile)}
</profile>

<cuisine_context>
{cuisine_ctx}
</cuisine_context>

<ingredients>
{ingredients}
</ingredients>

<candidate_swaps>
{_candidate_block(candidates)}
</candidate_swaps>

Propose healthier substitutions for this cook. {recipe_line}"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cuisine": {"type": "string", "description": "Cuisine the answer is written for."},
        "dish": {"type": "string", "description": "Dish name, or an empty string if none."},
        "summary": {"type": "string"},
        "substitutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "substitute": {"type": "string"},
                    "ratio": {"type": "string"},
                    "reason": {"type": "string"},
                    "flavor_note": {"type": "string"},
                    "cultural_note": {"type": "string"},
                },
                "required": ["original", "substitute", "ratio", "reason", "flavor_note", "cultural_note"],
                "additionalProperties": False,
            },
        },
        "adapted_recipe": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "servings": {"type": "integer"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "servings", "ingredients", "steps"],
            "additionalProperties": False,
        },
        "tips": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cuisine", "dish", "summary", "substitutions", "adapted_recipe", "tips"],
    "additionalProperties": False,
}
