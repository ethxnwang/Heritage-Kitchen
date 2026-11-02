import numpy as np

from heritage_kitchen.catalog import FLAVOR_AXES, NUTRIENT_FIELDS


def test_covers_at_least_thirty_cuisines(catalog):
    assert len(catalog.cuisines) >= 30


def test_every_cuisine_reference_resolves(catalog):
    for c in catalog.cuisines.values():
        assert c.flavor.shape == (len(FLAVOR_AXES),)
        for key in (*c.staples, *c.heavy_ingredients):
            assert key in catalog.ingredients, f"{c.key} references unknown ingredient {key}"


def test_ingredient_vectors_have_expected_shape(catalog):
    for ing in catalog.ingredients.values():
        assert ing.nutrients.shape == (len(NUTRIENT_FIELDS),)
        assert ing.flavor.shape == (len(FLAVOR_AXES),)
        assert np.all(ing.nutrients >= 0)


def test_ingredient_lookup_handles_aliases_and_quantities(catalog):
    assert catalog.ingredient("Patis").key == "fish_sauce"
    assert catalog.ingredient("2 cups shredded cheddar").key == "cheddar_cheese"
    assert catalog.ingredient("basmati rice").key == "white_rice"
    assert catalog.ingredient("brown rice").key == "brown_rice"
    assert catalog.ingredient("tomatoes").key == "tomato"
    assert catalog.ingredient("unobtainium") is None


def test_cuisine_lookup_by_culture_and_dish(catalog):
    assert catalog.cuisine("Filipino-American").key == "filipino"
    assert catalog.cuisine("Punjabi").key == "north_indian"
    assert catalog.cuisine("nigerian").key == "west_african"
    assert catalog.cuisine_for_dish("Jollof Rice").key == "west_african"
    assert catalog.cuisine_for_dish("mapo tofu").key == "sichuan"
    assert catalog.cuisine_for_dish("chicken adobo").key == "filipino"
