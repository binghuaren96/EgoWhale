"""Coarse annotation: an episode summary, coarse subtasks, or both, written to annotation/coarse.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from egowhale.config import DEFAULTS
from egowhale.media import video_shape
from egowhale.paths import ANNOTATION
from egowhale.step import Step
from egowhale.vlm.cache import ResponseCache, request_json
from egowhale.vlm.client import OpenAICompatibleClient, VLMClient, VLMError
from egowhale.vlm.config import VLMConfig
from egowhale.vlm.metadata import description_and_source
from egowhale.vlm.prompts import (
    BOTH_SCHEMA,
    DESCRIPTION_SCHEMA,
    SUBTASKS_SCHEMA,
    both_prompt,
    description_prompt,
    subtasks_prompt,
)
from egowhale.vlm.sampling import frame_message, sample_indices
from egowhale.vlm.schema import normalize_subtasks


class Annotate(Step):
    """Read the source video, never the synthesized outputs, so this branch can start at time zero."""

    name = "annotate"
    needs = ()
    makes = (ANNOTATION,)
    gpus = 0

    def __init__(self, config: dict | None = None, client: VLMClient | None = None):
        super().__init__()
        self.config = config or {}
        self.settings = _section(config, "annotate")
        self.mode = str(self.settings["mode"])
        self.sample_fps = float(self.settings["sample_fps"])
        self.max_frames = int(self.settings["max_frames"])
        self.max_subtasks = int(self.settings["max_subtasks"])
        self.min_subtask_frames = int(self.settings["min_subtask_frames"])
        self.prompt_version = str(self.settings["prompt_version"])
        self.on_error = str(self.settings["on_error"])
        self.vlm_config = VLMConfig.from_config(config)
        self.client = client

    def run(self, src: Path, dst: Path) -> None:
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        frames, _height, _width, fps = video_shape(src.with_suffix(".mp4"))
        existing, existing_source = description_and_source(src, default="")

        if self.mode == "off":
            self._write(dst, "disabled", frames, fps, existing, existing_source, [])
            return
        if self.mode == "description" and existing:
            self._write(dst, "ok", frames, fps, existing, existing_source, [])
            return

        client, reason = self._client()
        if client is None:
            self._write(dst, "skipped", frames, fps, existing, existing_source, [], error=reason)
            return

        # Sampling and decoding live inside the try: a video that will not decode is an error like any
        # other, and on_error=keep must still leave the episode and its coarse.json behind.
        cache = ResponseCache(dst / "vlm_cache", self.vlm_config.cache)
        try:
            indices = sample_indices(frames, fps, self.sample_fps, self.max_frames)
            if not indices:
                raise ValueError("no frames to sample")
            prompt = self._prompt(frames, fps, indices, existing)
            messages = frame_message(src.with_suffix(".mp4"), indices, prompt)
            payload = request_json(
                client,
                messages,
                cache,
                model=self.vlm_config.model,
                prompt_version=self.prompt_version,
                mode=self.mode,
                schema=self._schema(existing),
            )
            summary, subtasks, source = self._parse(payload, existing, frames, fps)
        except (VLMError, ValueError, OSError) as exc:
            self._fail(dst, frames, fps, existing, existing_source, f"{type(exc).__name__}: {exc}")
            return
        self._write(dst, "ok", frames, fps, summary, source, subtasks, sampled=len(indices))

    def _prompt(self, frames: int, fps: float, indices: list[int], existing: str) -> str:
        if self.mode == "description":
            return description_prompt(frames, fps, indices)
        if self.mode == "subtasks":
            return subtasks_prompt(frames, fps, indices, existing)
        if existing:
            # The description is already known; ask only for the subtasks.
            return subtasks_prompt(frames, fps, indices, existing)
        return both_prompt(frames, fps, indices)

    def _schema(self, existing: str) -> str:
        """The reply shape the repair round asks for, matching what `_prompt` requested."""
        if self.mode == "description":
            return DESCRIPTION_SCHEMA
        if self.mode == "subtasks" or existing:
            # `both` with a known description only asks for the subtasks.
            return SUBTASKS_SCHEMA
        return BOTH_SCHEMA

    def _parse(self, payload: dict, existing: str, frames: int, fps: float) -> tuple[str, list[dict], str]:
        summary = str(payload.get("episode_summary") or payload.get("description") or existing).strip()
        vlm_summary = bool(str(payload.get("episode_summary") or payload.get("description") or "").strip())
        subtasks = []
        if self.mode in ("subtasks", "both"):
            subtasks = normalize_subtasks(
                payload.get("subtasks"),
                frames,
                fps,
                self.max_subtasks,
                self.min_subtask_frames,
            )
        if self.mode in ("description", "both") and not summary:
            raise ValueError("reply has no episode_summary")
        if self.mode in ("subtasks", "both") and not subtasks:
            raise ValueError("reply has no usable subtasks")
        if existing and vlm_summary:
            source = "mixed"
        elif existing:
            source = "hdf5"
        elif vlm_summary:
            source = "vlm"
        else:
            source = "none"
        return summary, subtasks, source

    def _client(self) -> tuple[VLMClient | None, str]:
        if self.client is not None:
            return self.client, ""
        missing = self.vlm_config.missing()
        if missing:
            return None, missing
        return OpenAICompatibleClient(self.vlm_config), ""

    def _fail(
        self,
        dst: Path,
        frames: int,
        fps: float,
        summary: str,
        source: str,
        error: str,
    ) -> None:
        if self.on_error == "fail":
            raise VLMError(error)
        self._write(dst, "error", frames, fps, summary, source, [], error=error)

    def _write(
        self,
        dst: Path,
        status: str,
        frames: int,
        fps: float,
        summary: str,
        source: str,
        subtasks: list[dict],
        sampled: int = 0,
        error: str | None = None,
    ) -> None:
        report = {
            "status": status,
            "meta": {
                "mode": self.mode,
                "prompt_version": self.prompt_version,
                "model": self.vlm_config.model,
                "frames": frames,
                "fps": fps,
                "description_source": source,
                "sampled_frames": sampled,
                "max_subtasks": self.max_subtasks,
                "min_subtask_frames": self.min_subtask_frames,
            },
            "episode_summary": summary,
            "subtasks": subtasks,
            "error": error,
        }
        path = dst / ANNOTATION
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _section(config: dict | None, name: str) -> dict:
    values = copy.deepcopy(DEFAULTS[name])
    if isinstance(config, dict) and isinstance(config.get(name), dict):
        values.update(config[name])
    elif isinstance(config, dict) and any(key in config for key in values):
        values.update(config)
    return values
