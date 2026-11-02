import json
from types import SimpleNamespace

import pytest

from heritage_kitchen.llm import FALLBACK_BETA, LLMError, LLMRefusal, SubstitutionLLM
from heritage_kitchen.prompts import OUTPUT_SCHEMA


class StubMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_client(response):
    messages = StubMessages(response)
    beta_messages = StubMessages(response)
    client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta_messages))
    return client, messages, beta_messages


def text_response(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=json.dumps(payload)),
        ],
    )


def test_request_shape_with_fallbacks(settings):
    client, _, beta = make_client(text_response({"summary": "ok"}))
    settings.use_fallbacks = True
    out = SubstitutionLLM(settings, client=client).generate("hello")
    call = beta.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["format"]["schema"] == OUTPUT_SCHEMA
    assert "temperature" not in call
    assert out["summary"] == "ok"
    assert out["_model"] == "claude-opus-5-5"


def test_request_without_fallbacks_uses_ga_endpoint(settings):
    client, messages, beta = make_client(text_response({"summary": "ok"}))
    settings.use_fallbacks = False
    SubstitutionLLM(settings, client=client).generate("hello")
    assert messages.calls and not beta.calls


def test_refusal_raises(settings):
    resp = text_response({})
    resp.stop_reason = "refusal"
    client, _, _ = make_client(resp)
    with pytest.raises(LLMRefusal):
        SubstitutionLLM(settings, client=client).generate("x")


def test_truncation_raises(settings):
    client, _, _ = make_client(text_response({}, stop_reason="max_tokens"))
    with pytest.raises(LLMError):
        SubstitutionLLM(settings, client=client).generate("x")


def test_schema_is_strict():
    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(OUTPUT_SCHEMA)
