"""Uniform frame sampling and the image/text message shape the providers expect."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from egowhale.media import read_frames


def sample_indices(frames: int, fps: float, sample_fps: float, max_frames: int) -> list[int]:
    """Frame indexes at about `sample_fps`, evenly thinned to at most `max_frames`."""
    if frames <= 0 or max_frames <= 0:
        return []
    step = max(float(fps) / max(float(sample_fps), 1e-6), 1.0)
    indices = np.unique(np.round(np.arange(0, frames, step)).astype(int))
    indices = indices[(indices >= 0) & (indices < frames)]
    if len(indices) > max_frames:
        picks = np.unique(np.round(np.linspace(0, len(indices) - 1, max_frames)).astype(int))
        indices = indices[picks]
    return [int(index) for index in indices]


def frame_message(video: Path | str, indices: list[int], prompt: str, quality: int = 80) -> list[dict]:
    """One user message: JPEG-encoded frames as data URLs, then the text prompt.

    The return value is a complete `messages` list for an OpenAI-compatible chat request, not just
    its content, so callers can pass it straight through and append follow-ups to it.
    """
    if not indices:
        raise ValueError(f"no frames selected from {video}")
    frames = read_frames(Path(video), indices)
    content = []
    for frame in frames:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            continue
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
    if not content:
        raise ValueError(f"could not encode frames from {video}")
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def frame_list(indices: list[int], fps: float) -> str:
    """The sampled indexes and timestamps shown to the model."""
    safe_fps = float(fps) or 1.0
    return ", ".join(f"{index} ({index / safe_fps:.2f}s)" for index in indices)
