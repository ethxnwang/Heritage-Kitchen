import numpy as np
import pytest

from heritage_kitchen.nutrition import (
    RESTRICTION_EXCLUDES,
    cosine,
    health_scores,
    match_cuisines,
    nutrition_delta,
    rank_substitutes,
)


def test_health_score_prefers_obvious_swaps(catalog):
    pairs = [
        ("lard", "olive_oil"),
        ("heavy_cream", "greek_yogurt"),
        ("soy_sauce", "low_sodium_soy_sauce"),
        ("white_rice", "brown_rice"),
        ("pork_belly", "chicken_breast"),
    ]
    for bad, good in pairs:
        a = health_scores(catalog.ingredients[bad].nutrients)[0]
        b = health_scores(catalog.ingredients[good].nutrients)[0]
        assert b > a, (bad, good)


def test_health_scores_vectorized_matches_single(catalog):
    keys = ["butter", "tofu_firm", "white_sugar"]
    matrix = np.vstack([catalog.ingredients[k].nutrients for k in keys])
    batch = health_scores(matrix, goals=["lower_sodium"])
    singles = [health_scores(catalog.ingredients[k].nutrients, goals=["lower_sodium"])[0] for k in keys]
    assert np.allclose(batch, singles)


def test_goal_weights_shift_ranking(catalog):
    soy = catalog.ingredients["soy_sauce"]
    low = catalog.ingredients["low_sodium_soy_sauce"]
    plain = nutrition_delta(soy, low)["health_score_delta"]
    salt_focused = nutrition_delta(soy, low, goals=["lower_sodium"])["health_score_delta"]
    assert salt_focused > plain > 0


def test_cosine_basics():
    a = np.array([1.0, 0, 0])
    assert cosine(a, a)[0, 0] == pytest.approx(1.0)
    assert cosine(a, np.array([0, 1.0, 0]))[0, 0] == pytest.approx(0.0)
    assert cosine(a, np.zeros(3))[0, 0] == 0.0


def test_rank_substitutes_only_returns_healthier_options(catalog):
    original = catalog.ingredients["coconut_milk"]
    base = health_scores(original.nutrients)[0]
    swaps = rank_substitutes(catalog, original, cuisine=catalog.cuisines["thai"])
    assert swaps
    assert swaps[0].substitute.key == "light_coconut_milk"
    for s in swaps:
        assert health_scores(s.substitute.nutrients)[0] > base


@pytest.mark.parametrize("restriction", sorted(RESTRICTION_EXCLUDES))
def test_rank_substitutes_respects_restrictions(catalog, restriction):
    banned = RESTRICTION_EXCLUDES[restriction]
    for ing in catalog.ingredients.values():
        for s in rank_substitutes(catalog, ing, restrictions=[restriction], top_k=5):
            assert s.substitute.key not in banned


def test_match_cuisines_finds_spicy_sour_profiles(catalog):
    taste = np.array([1.0, 0.5, 0.6, 0.7, 0.8, 0.1])
    keys = [m["key"] for m in match_cuisines(catalog, taste, top_k=3)]
    assert "thai" in keys
