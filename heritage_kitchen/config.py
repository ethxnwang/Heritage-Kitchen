from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

DEFAULT_ORIGINS = (
    "https://heritagekitchencabinets.com",
    "https://www.heritagekitchencabinets.com",
    "http://heritagekitchencabinets.com",
    "http://www.heritagekitchencabinets.com",
)


def _env_list(name: str, default: tuple) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("HK_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.getenv("HK_EFFORT", "medium"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("HK_MAX_TOKENS", "16000")))
    timeout_s: float = field(default_factory=lambda: float(os.getenv("HK_TIMEOUT_S", "120")))
    use_fallbacks: bool = field(default_factory=lambda: _env_bool("HK_USE_FALLBACKS", True))
    offline: bool = field(default_factory=lambda: _env_bool("HK_OFFLINE", False))
    allowed_origins: List[str] = field(default_factory=lambda: _env_list("HK_ALLOWED_ORIGINS", DEFAULT_ORIGINS))
    # Squarespace previews and the editor are served from *.squarespace.com
    allowed_origin_regex: str = field(
        default_factory=lambda: os.getenv("HK_ALLOWED_ORIGIN_REGEX", r"https://([a-z0-9-]+\.)*squarespace\.com")
    )
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("HK_RATE_LIMIT_PER_MINUTE", "20")))

    @property
    def has_credentials(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def get_settings() -> Settings:
    return Settings()
