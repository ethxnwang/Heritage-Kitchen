import pytest
from fastapi.testclient import TestClient

from heritage_kitchen.api import create_app
from heritage_kitchen.engine import SubstitutionEngine


@pytest.fixture
def client(settings, catalog, fake_llm):
    engine = SubstitutionEngine(settings=settings, catalog=catalog, llm=fake_llm)
    return TestClient(create_app(settings=settings, engine=engine))


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["cuisines"] >= 30


def test_cuisine_list_and_detail(client):
    cuisines = client.get("/api/cuisines").json()
    assert len(cuisines) >= 30
    detail = client.get("/api/cuisines/ethiopian").json()
    assert "Teff (cooked)" in detail["staples"]
    assert client.get("/api/cuisines/atlantis").status_code == 404


def test_ingredient_lookup(client):
    body = client.get("/api/ingredients", params={"q": "nam pla"}).json()
    assert body["key"] == "fish_sauce"


def test_substitutions_endpoint(client):
    r = client.post("/api/substitutions", json={"culture": "Filipino", "dish": "adobo"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "llm"
    assert len(body["substitutions"]) == 3


def test_substitutions_requires_some_input(client):
    assert client.post("/api/substitutions", json={}).status_code == 422


def test_rejects_unknown_restriction(client):
    r = client.post("/api/substitutions", json={"cuisine": "Thai", "restrictions": ["carnivore"]})
    assert r.status_code == 422


def test_taste_match(client):
    r = client.post("/api/taste-match", json={"taste": {"spicy": 10, "sour": 8, "umami": 8}, "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()["matches"]) == 3


@pytest.mark.parametrize("origin", [
    "https://www.heritagekitchencabinets.com",
    "https://heritagekitchencabinets.com",
    "https://shiny-owl-a1b2.squarespace.com",
])
def test_cors_allows_squarespace_origins(client, origin):
    r = client.options(
        "/api/substitutions",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


def test_cors_blocks_other_origins(client):
    r = client.options(
        "/api/substitutions",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_rate_limit(settings, catalog, fake_llm):
    settings.rate_limit_per_minute = 2
    engine = SubstitutionEngine(settings=settings, catalog=catalog, llm=fake_llm)
    client = TestClient(create_app(settings=settings, engine=engine))
    payload = {"cuisine": "Greek"}
    assert client.post("/api/substitutions", json=payload).status_code == 200
    assert client.post("/api/substitutions", json=payload).status_code == 200
    assert client.post("/api/substitutions", json=payload).status_code == 429
