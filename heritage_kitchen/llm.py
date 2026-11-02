"""Thin wrapper around the Anthropic Messages API."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import anthropic

from .config import Settings
from .prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model call failed in a way the caller can recover from."""


class LLMRefusal(LLMError):
    pass


class SubstitutionLLM:
    def __init__(self, settings: Settings, client: Optional[anthropic.Anthropic] = None):
        self.settings = settings
        self.client = client or anthropic.Anthropic(timeout=settings.timeout_s, max_retries=2)

    def _request_kwargs(self, user_prompt: str) -> Dict[str, Any]:
        return dict(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        )

    def generate(self, user_prompt: str) -> Dict[str, Any]:
        kwargs = self._request_kwargs(user_prompt)
        try:
            if self.settings.use_fallbacks:
                response = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                response = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic credentials were rejected") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit") from exc
        except anthropic.BadRequestError as exc:
            log.error("bad request to Anthropic: %s", exc)
            raise LLMError("request rejected by the model API") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API returned {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Anthropic API") from exc

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMRefusal(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("model output was truncated; raise HK_MAX_TOKENS")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            raise LLMError("model returned no text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("model returned invalid JSON") from exc

        data["_model"] = getattr(response, "model", self.settings.model)
        return data
