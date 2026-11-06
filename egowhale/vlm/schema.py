"""JSON contracts for the coarse labels and the L3 verdict, plus their normalisation."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict:
    """Return the first JSON object in a model reply. Code fences and prose are tolerated."""
    value = str(text).strip()
    value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    match = re.search(r"\{.*\}", value, re.DOTALL)
    if match is None:
        raise ValueError("reply has no JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("reply JSON is not an object")
    return payload


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def verdict_from_payload(payload: dict) -> dict:
    """The audit reply as a stable dict. A missing verdict becomes `None`, not `False`."""
    consistent = payload.get("is_consistent", payload.get("is consistent", payload.get("consistent")))
    confidence = payload.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return {
        "status": "ok",
        "is_consistent": _bool(consistent),
        "confidence": confidence,
        "reasoning": payload.get("reasoning") or payload.get("reason"),
        "mismatches": payload.get("mismatches") or [],
    }


def verdict_from_text(text: str) -> dict:
    try:
        return verdict_from_payload(extract_json(text))
    except (ValueError, TypeError):
        return {"status": "error", "raw": str(text)}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame(item: dict, keys: tuple[str, ...], seconds: tuple[str, ...], fps: float) -> float | None:
    for key in keys:
        value = _number(item.get(key))
        if value is not None:
            return value
    for key in seconds:
        value = _number(item.get(key))
        if value is not None:
            return value * fps
    return None


def _join(left: str, right: str) -> str:
    parts = [part for part in (left.strip(), right.strip()) if part]
    return " / ".join(dict.fromkeys(parts))


def normalize_subtasks(
    raw: Any,
    frames: int,
    fps: float,
    max_subtasks: int,
    min_subtask_frames: int,
) -> list[dict]:
    """Sort, cover [0, frames), merge short or excess segments, and clamp to the video."""
    if frames <= 0:
        return []
    items: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        start = _frame(item, ("start_frame", "start"), ("start_time", "start_sec"), fps)
        end = _frame(item, ("end_frame", "end"), ("end_time", "end_sec"), fps)
        if start is None or end is None:
            continue
        start = max(0, min(int(round(start)), frames))
        end = max(0, min(int(round(end)), frames))
        if end <= start:
            continue
        text = item.get("instruction") or item.get("label") or item.get("text") or ""
        items.append({"start_frame": start, "end_frame": end, "instruction": str(text).strip()})
    if not items:
        return []

    items.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    merged: list[dict] = []
    for item in items:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        if item["start_frame"] < previous["end_frame"]:
            item["start_frame"] = previous["end_frame"]
        elif item["start_frame"] > previous["end_frame"]:
            previous["end_frame"] = item["start_frame"]
        if item["end_frame"] <= item["start_frame"]:
            previous["instruction"] = _join(previous["instruction"], item["instruction"])
            continue
        merged.append(item)

    merged[0]["start_frame"] = 0
    merged[-1]["end_frame"] = frames

    # Merge pieces shorter than the coarse minimum into a neighbour.
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, item in enumerate(merged):
            if item["end_frame"] - item["start_frame"] >= min_subtask_frames:
                continue
            if index > 0:
                previous = merged[index - 1]
                previous["end_frame"] = item["end_frame"]
                previous["instruction"] = _join(previous["instruction"], item["instruction"])
                del merged[index]
            else:
                following = merged[1]
                following["start_frame"] = item["start_frame"]
                following["instruction"] = _join(item["instruction"], following["instruction"])
                del merged[0]
            changed = True
            break

    # Too many segments: merge the shortest one into its predecessor.
    while len(merged) > max(1, max_subtasks):
        index = min(range(len(merged)), key=lambda position: merged[position]["end_frame"] - merged[position]["start_frame"])
        index = max(1, index)
        previous = merged[index - 1]
        previous["end_frame"] = merged[index]["end_frame"]
        previous["instruction"] = _join(previous["instruction"], merged[index]["instruction"])
        del merged[index]

    merged[0]["start_frame"] = 0
    merged[-1]["end_frame"] = frames
    return merged
