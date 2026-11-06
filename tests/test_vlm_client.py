"""VLM transport, parsing, retries, and cache. No network and no API key are needed."""

from __future__ import annotations

import json
import urllib.error

import pytest

from egowhale.vlm.cache import ResponseCache, request_json
from egowhale.vlm.client import MockClient, OpenAICompatibleClient, VLMError
from egowhale.vlm.config import VLMConfig


class _Response:
    def __init__(self, body: dict):
        self.body = body

    def read(self) -> bytes:
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _reply(text: str) -> _Response:
    return _Response({"choices": [{"message": {"content": text}}]})


def test_mock_client_returns_queued_json():
    client = MockClient(['{"ok": true}'])
    assert client.complete_json([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert len(client.calls) == 1


def test_http_429_is_retried_then_succeeds():
    calls = []

    def opener(_request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.HTTPError("http://x", 429, "slow down", None, None)
        return _reply("hello")

    client = OpenAICompatibleClient(
        VLMConfig(model="m", api_key="k", retries=1, backoff_s=0),
        opener=opener,
        sleep=lambda _seconds: None,
    )
    assert client.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert len(calls) == 2


def test_http_401_is_not_retried():
    calls = []

    def opener(_request, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError("http://x", 401, "bad key", None, None)

    client = OpenAICompatibleClient(
        VLMConfig(model="m", api_key="k", retries=3, backoff_s=0),
        opener=opener,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(VLMError) as caught:
        client.complete([{"role": "user", "content": "hi"}])
    assert caught.value.kind == "auth"
    assert len(calls) == 1


def test_missing_key_or_model_is_reported_without_a_request():
    assert VLMConfig(model="m", api_key="").missing() == "EGOWHALE_VLM_API_KEY is unset"
    assert VLMConfig(model="", api_key="k").missing() == "vlm.model is unset"


def test_cache_returns_the_first_reply(tmp_path):
    cache = ResponseCache(tmp_path, enabled=True)
    client = MockClient(['{"answer": 1}', '{"answer": 2}'])
    messages = [{"role": "user", "content": "same"}]
    first = request_json(client, messages, cache, model="m", prompt_version="v1")
    second = request_json(client, messages, cache, model="m", prompt_version="v1")
    assert first == second == {"answer": 1}
    assert len(client.calls) == 1


def test_invalid_json_gets_one_repair_round():
    client = MockClient(["sure, here it is", '{"answer": 1}'])
    messages = [{"role": "user", "content": "hi"}]
    payload = request_json(client, messages, None, schema='{"answer": 0}')
    assert payload == {"answer": 1}
    assert len(client.calls) == 2
    assert "not valid JSON" in client.calls[1][1]["content"]


def test_invalid_json_without_a_schema_raises():
    client = MockClient(["no json here"])
    with pytest.raises(ValueError):
        request_json(client, [{"role": "user", "content": "hi"}], None)


def test_repair_failure_raises_after_one_round():
    client = MockClient(["no json", "still no json"])
    with pytest.raises(ValueError):
        request_json(client, [{"role": "user", "content": "hi"}], None, schema='{"answer": 0}')
    assert len(client.calls) == 2
