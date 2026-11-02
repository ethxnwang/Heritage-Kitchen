from heritage_kitchen.engine import SubstitutionEngine
from heritage_kitchen.models import SubstitutionRequest, TasteProfile
from heritage_kitchen.prompts import SYSTEM_PROMPT, build_user_prompt


def test_offline_mode_from_culture_only(settings, catalog):
    engine = SubstitutionEngine(settings=settings, catalog=catalog)
    assert engine.llm is None
    res = engine.run(SubstitutionRequest(culture="Korean", health_goals=["lower_sodium"]))
    assert res.source == "offline"
    assert res.cuisine.key == "korean"
    assert res.substitutions
    originals = {s.original for s in res.substitutions}
    assert "Soy sauce" in originals
    assert all(s.nutrition and s.nutrition["health_score_delta"] > 0 for s in res.substitutions)


def test_offline_with_explicit_ingredients_and_restriction(settings, catalog):
    engine = SubstitutionEngine(settings=settings, catalog=catalog)
    req = SubstitutionRequest(
        dish="butter chicken",
        ingredients=["3 tbsp ghee", "1 cup heavy cream", "basmati rice"],
        restrictions=["dairy_free"],
    )
    res = engine.run(req)
    assert res.cuisine.key == "north_indian"
    subs = {s.original: s.substitute for s in res.substitutions}
    assert subs["Heavy cream"] != "Plain nonfat Greek yogurt"
    assert "Ghee" in subs


def test_llm_path_attaches_nutrition(settings, catalog, fake_llm):
    engine = SubstitutionEngine(settings=settings, catalog=catalog, llm=fake_llm)
    req = SubstitutionRequest(
        culture="Filipino-American",
        dish="chicken adobo",
        taste=TasteProfile(sour=9, salty=6),
        health_goals=["lower_sodium"],
    )
    res = engine.run(req)
    assert res.source == "llm"
    assert res.model == "claude-opus-5-5"
    assert res.adapted_recipe and res.adapted_recipe.servings == 4

    by_original = {s.original: s for s in res.substitutions}
    soy = by_original["soy sauce"].nutrition
    assert soy["change_pct"]["sodium_mg"] < 0
    assert by_original["pork belly"].nutrition["health_score_delta"] > 0

    prompt = fake_llm.prompts[0]
    assert "Filipino" in prompt
    assert "sour 9/10" in prompt
    assert "<candidate_swaps>" in prompt
    assert "Low-sodium soy sauce" in prompt


def test_llm_failure_falls_back_to_offline(settings, catalog, failing_llm):
    engine = SubstitutionEngine(settings=settings, catalog=catalog, llm=failing_llm)
    res = engine.run(SubstitutionRequest(cuisine="Mexican", dish="enchiladas"))
    assert res.source == "offline"
    assert res.substitutions


def test_unknown_cuisine_still_prompts_model(settings, catalog, fake_llm):
    engine = SubstitutionEngine(settings=settings, catalog=catalog, llm=fake_llm)
    engine.run(SubstitutionRequest(culture="Hmong", dish="khaub poob"))
    assert "No cuisine in the reference table matched" in fake_llm.prompts[0]


def test_system_prompt_is_static():
    assert "{" not in SYSTEM_PROMPT
    a = build_user_prompt(SubstitutionRequest(cuisine="Thai"), None, {})
    b = build_user_prompt(SubstitutionRequest(cuisine="Greek"), None, {})
    assert a != b
