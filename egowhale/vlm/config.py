"""Provider settings shared by the coarse annotator and the L3 audit.

The API key is read from an environment variable named by `vlm.api_key_env`. It is never read from
the YAML config and is kept out of `repr`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_KEY_ENV = "EGOWHALE_VLM_API_KEY"


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str = field(default="", repr=False)
    api_key_env: str = DEFAULT_KEY_ENV
    timeout_s: float = 120.0
    retries: int = 2
    backoff_s: float = 2.0
    cache: bool = True

    @classmethod
    def from_config(cls, config: Mapping | None = None, environ: Mapping | None = None) -> VLMConfig:
        """Read the `vlm` section, falling back to the matching environment variables."""
        raw = _section(config, "vlm")
        if not raw and isinstance(config, Mapping):
            keys = ("model", "base_url", "api_key_env", "timeout_s", "retries", "backoff_s", "cache")
            if any(key in config for key in keys):
                raw = dict(config)
        env = environ if environ is not None else os.environ
        key_env = str(raw.get("api_key_env") or DEFAULT_KEY_ENV)
        model = str(raw.get("model") or env.get("EGOWHALE_VLM_MODEL", "")).strip()
        base_url = str(raw.get("base_url") or env.get("EGOWHALE_VLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        return cls(
            base_url=base_url,
            model=model,
            api_key=str(env.get(key_env, "")).strip(),
            api_key_env=key_env,
            timeout_s=float(raw.get("timeout_s", 120)),
            retries=int(raw.get("retries", 2)),
            backoff_s=float(raw.get("backoff_s", 2.0)),
            cache=bool(raw.get("cache", True)),
        )

    def missing(self) -> str:
        """Why a real request cannot be made, or an empty string when it can."""
        if not self.model:
            return "vlm.model is unset"
        if not self.api_key:
            return f"{self.api_key_env} is unset"
        return ""


def _section(config: Mapping | None, name: str) -> dict:
    if not isinstance(config, Mapping):
        return {}
    section = config.get(name)
    return dict(section) if isinstance(section, Mapping) else {}
