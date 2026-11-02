import os

import pytest

os.environ.setdefault("HK_OFFLINE", "1")

from heritage_kitchen.catalog import load_catalog  # noqa: E402
from heritage_kitchen.config import Settings  # noqa: E402
from heritage_kitchen.llm import LLMError  # noqa: E402


@pytest.fixture
def catalog():
    return load_catalog()


@pytest.fixture
def settings():
    s = Settings()
    s.offline = True
    s.rate_limit_per_minute = 0
    return s


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return dict(self.reply)


ADOBO_REPLY = {
    "cuisine": "Filipino",
    "dish": "Chicken adobo",
    "summary": "Leaner adobo that keeps the vinegar, garlic, and bay leaf front and center.",
    "substitutions": [
        {
            "original": "pork belly",
            "substitute": "skinless chicken thigh",
            "ratio": "1:1 by weight",
            "reason": "Far less saturated fat while staying juicy in a long braise.",
            "flavor_note": "Chicken soaks up the vinegar-soy braise well.",
            "cultural_note": "Chicken adobo is just as common as pork adobo across the Philippines.",
        },
        {
            "original": "soy sauce",
            "substitute": "low-sodium soy sauce",
            "ratio": "1:1, add a splash more cane vinegar",
            "reason": "Cuts sodium by roughly a third.",
            "flavor_note": "Extra vinegar keeps the sharpness.",
            "cultural_note": "Toyo is still the base, just a lighter bottle.",
        },
        {
            "original": "patis",
            "substitute": "a squeeze of calamansi",
            "ratio": "Replace half",
            "reason": "Less sodium.",
            "flavor_note": "Brighter finish.",
            "cultural_note": "Calamansi is everywhere in Filipino cooking.",
        },
    ],
    "adapted_recipe": {
        "title": "Lighter Chicken Adobo",
        "servings": 4,
        "ingredients": ["1.5 lb chicken", "1/3 cup low-sodium soy sauce", "1/2 cup cane vinegar"],
        "steps": ["Combine and marinate.", "Braise 35 minutes.", "Reduce the sauce."],
    },
    "tips": ["Serve over half brown rice, half white rice."],
    "_model": "claude-opus-5-5",
}


@pytest.fixture
def fake_llm():
    return FakeLLM(reply=ADOBO_REPLY)


@pytest.fixture
def failing_llm():
    return FakeLLM(error=LLMError("simulated outage"))
