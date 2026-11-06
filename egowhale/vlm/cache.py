"""Content-addressed JSON cache for VLM replies. Images are hashed, never stored here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from egowhale.vlm.client import VLMClient
from egowhale.vlm.prompts import repair_prompt
from egowhale.vlm.schema import extract_json


class ResponseCache:
    def __init__(self, root: Path | str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def key(self, messages: list[dict], **meta: Any) -> str:
        blob = json.dumps({"messages": messages, "meta": meta}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def put(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def request_json(
    client: VLMClient,
    messages: list[dict],
    cache: ResponseCache | None = None,
    schema: str = "",
    **meta: Any,
) -> dict:
    """Return a parsed reply, using the cache when it has the same model, prompt, and images.

    A reply that is not valid JSON gets one repair round when `schema` names the expected shape;
    only the final parsed payload is cached.
    """
    key = cache.key(messages, **meta) if cache is not None and cache.enabled else ""
    if key:
        hit = cache.get(key)
        if hit is not None:
            return hit
    try:
        payload = extract_json(client.complete(messages))
    except ValueError:
        if not schema:
            raise
        payload = extract_json(client.complete([*messages, {"role": "user", "content": repair_prompt(schema)}]))
    if key and cache is not None:
        cache.put(key, payload)
    return payload
