"""Compatibility shim for the old L3 helpers.

New code uses `egowhale.quality.vlm_audit.VlmAudit` and `egowhale.vlm`. `description` and `_verdict`
remain here because older scripts and tests imported them directly.
"""

from __future__ import annotations

from pathlib import Path

from egowhale.vlm.metadata import read_description
from egowhale.vlm.schema import verdict_from_text


def description(path: Path) -> str:
    return read_description(path, default="manipulation")


def _verdict(text: str) -> dict:
    return verdict_from_text(text)


def audit(*_args, **_kwargs) -> dict:
    return {"status": "skipped", "reason": "L3 moved to egowhale.quality.vlm_audit"}
