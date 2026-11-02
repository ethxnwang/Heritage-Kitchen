from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .catalog import FLAVOR_AXES, load_catalog
from .config import Settings, get_settings
from .engine import SubstitutionEngine
from .models import SubstitutionRequest, SubstitutionResponse, TasteMatchRequest
from .nutrition import health_scores, match_cuisines

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding one-minute window per client IP. Good enough for a single instance."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(settings: Optional[Settings] = None, engine: Optional[SubstitutionEngine] = None) -> FastAPI:
    settings = settings or get_settings()
    catalog = load_catalog()
    engine = engine or SubstitutionEngine(settings=settings, catalog=catalog)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app = FastAPI(
        title="Heritage Kitchen API",
        version=__version__,
        description="Culturally grounded, healthier ingredient substitutions.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=settings.allowed_origin_regex or None,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    def rate_limited(request: Request) -> None:
        if not limiter.check(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")

    @app.get("/")
    def root():
        return {"service": "heritage-kitchen", "version": __version__, "docs": "/docs"}

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "llm_enabled": engine.llm is not None,
            "model": settings.model if engine.llm is not None else None,
            "cuisines": len(catalog.cuisines),
            "ingredients": len(catalog.ingredients),
        }

    @app.get("/api/cuisines")
    def list_cuisines():
        return [
            {"key": c.key, "name": c.name, "region": c.region, "dishes": list(c.dishes)}
            for c in sorted(catalog.cuisines.values(), key=lambda c: c.name)
        ]

    @app.get("/api/cuisines/{key}")
    def get_cuisine(key: str):
        c = catalog.cuisines.get(key)
        if c is None:
            raise HTTPException(status_code=404, detail="Unknown cuisine")
        return {
            "key": c.key,
            "name": c.name,
            "region": c.region,
            "flavor": dict(zip(FLAVOR_AXES, c.flavor.round(2).tolist())),
            "staples": [catalog.ingredients[s].name for s in c.staples if s in catalog.ingredients],
            "heavy_ingredients": [catalog.ingredients[s].name for s in c.heavy_ingredients if s in catalog.ingredients],
            "dishes": list(c.dishes),
            "cultures": list(c.cultures),
        }

    @app.get("/api/ingredients")
    def lookup_ingredient(q: str = Query(..., min_length=2, max_length=80)):
        ing = catalog.ingredient(q)
        if ing is None:
            raise HTTPException(status_code=404, detail="Ingredient not in the reference table")
        return {
            "key": ing.key,
            "name": ing.name,
            "category": ing.category,
            "nutrients_per_100g": ing.nutrient_dict(),
            "health_score": round(float(health_scores(ing.nutrients)[0]), 3),
        }

    @app.post("/api/substitutions", response_model=SubstitutionResponse, dependencies=[Depends(rate_limited)])
    def substitutions(req: SubstitutionRequest):
        return engine.run(req)

    @app.post("/api/taste-match", dependencies=[Depends(rate_limited)])
    def taste_match(req: TasteMatchRequest):
        taste = np.asarray(req.taste.as_list(), dtype=float) / 10.0
        return {"matches": match_cuisines(catalog, taste, top_k=req.top_k)}

    return app


app = create_app()
