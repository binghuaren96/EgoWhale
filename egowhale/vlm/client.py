"""OpenAI-compatible chat client, plus a deterministic mock for tests and local runs."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Callable

from egowhale.vlm.config import VLMConfig
from egowhale.vlm.schema import extract_json


class VLMError(RuntimeError):
    """A request that could not produce a usable reply. `kind` separates auth from transient errors."""

    def __init__(self, message: str, kind: str = "error", retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status


class VLMClient:
    def complete(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError

    def complete_json(self, messages: list[dict], **kwargs) -> dict:
        return extract_json(self.complete(messages, **kwargs))


class OpenAICompatibleClient(VLMClient):
    """POST to `{base_url}/chat/completions`, retrying only errors that can heal."""

    def __init__(
        self,
        config: VLMConfig,
        opener: Callable | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.config = config
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def complete(self, messages: list[dict], **kwargs) -> str:
        missing = self.config.missing()
        if missing:
            kind = "config" if not self.config.model else "auth"
            raise VLMError(missing, kind=kind)
        body = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            **({"max_tokens": kwargs["max_tokens"]} if "max_tokens" in kwargs else {}),
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        attempt = 0
        while True:
            try:
                with self._opener(request, timeout=self.config.timeout_s) as response:
                    reply = json.loads(response.read().decode("utf-8"))
                return _content(reply)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 409, 429) or 500 <= exc.code < 600
                if not retryable or attempt >= self.config.retries:
                    raise VLMError(f"HTTP {exc.code}: {exc.reason}", kind="auth" if exc.code in (401, 403) else "http", retryable=retryable, status=exc.code) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.config.retries:
                    raise VLMError(f"{type(exc).__name__}: {exc}", kind="network", retryable=True) from exc
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise VLMError(f"malformed reply: {exc}", kind="parse") from exc
            self._sleep(self.config.backoff_s * (2**attempt) + random.random() * 0.5)
            attempt += 1


class MockClient(VLMClient):
    """Replies from a queue. An exception in the queue is raised on that call."""

    def __init__(self, replies: list[str | Exception] | None = None, default: str | Exception | None = None):
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        if self.replies:
            reply = self.replies.pop(0)
        elif self.default is not None:
            reply = self.default
        else:
            raise AssertionError("MockClient has no reply left")
        if isinstance(reply, Exception):
            raise reply
        return reply


def _content(reply: dict) -> str:
    content = reply["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)
