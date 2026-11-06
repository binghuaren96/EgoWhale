"""L3: ask the VLM whether the composite matches the description, then write the final decision.

This stage is deliberately separate from `curate.py`: L1/L2 stay deterministic and local, while a
missing key, timeout, or malformed reply never prevents their evidence from being written.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from egowhale.config import DEFAULTS
from egowhale.media import FPS, video_shape
from egowhale.paths import ANNOTATION, COMPOSITE, FINAL, PREFIX, QUALITY, VLM_AUDIT
from egowhale.quality.finalize import decide
from egowhale.step import Step
from egowhale.vlm.cache import ResponseCache, request_json
from egowhale.vlm.client import OpenAICompatibleClient, VLMClient, VLMError
from egowhale.vlm.config import VLMConfig
from egowhale.vlm.metadata import read_description
from egowhale.vlm.prompts import AUDIT_SCHEMA, audit_prompt
from egowhale.vlm.sampling import frame_message, sample_indices
from egowhale.vlm.schema import verdict_from_payload


class VlmAudit(Step):
    name = "vlm_audit"
    needs = (QUALITY, COMPOSITE)
    makes = (VLM_AUDIT, FINAL)
    gpus = 0

    def __init__(self, config: dict | None = None, client: VLMClient | None = None):
        super().__init__()
        self.config = config or {}
        self.settings = _section(config, "audit")
        self.enabled = bool(self.settings["enabled"])
        self.sample_fps = float(self.settings["sample_fps"])
        self.max_frames = int(self.settings["max_frames"])
        self.prompt_version = str(self.settings["prompt_version"])
        self.fail_open = bool(self.settings["fail_open"])
        self.vlm_config = VLMConfig.from_config(config)
        self.client = client

    def run(self, src: Path, dst: Path) -> None:
        src, dst = Path(src), Path(dst)
        quality = json.loads((dst / QUALITY).read_text())
        audit = self._audit(src, dst, quality)
        audit.setdefault("error", None)
        self._write(dst / VLM_AUDIT, audit)
        final = decide(quality, audit, fail_open=self.fail_open)
        self._write(dst / FINAL, final)
        print(f"  vlm {audit['status']}  final {final['decision']}")

    def _audit(self, src: Path, dst: Path, quality: dict) -> dict:
        if not bool(quality.get("keep_l1_l2", quality.get("keep", False))):
            return {"status": "skipped", "reason": "l1_l2 rejected the episode"}
        if not self.enabled:
            return {"status": "disabled", "reason": "audit.enabled is false"}
        client, reason = self._client()
        if client is None:
            return {"status": "skipped", "reason": reason}

        description, subtasks = self._description(src, dst, quality)
        composite = dst / COMPOSITE
        cache = ResponseCache(dst / "vlm_cache", self.vlm_config.cache)
        # Sampling and decoding live inside the try: a composite that will not decode is an audit
        # error, and fail_open decides what it means for the episode instead of failing it here.
        try:
            frames, fps, approach = _geometry(quality, dst, composite)
            count = min(frames, video_shape(composite)[0] - approach)
            if count <= 0:
                return {"status": "error", "error": "composite has no frames after the approach"}
            indices = sample_indices(count, fps, self.sample_fps, self.max_frames)
            if not indices:
                return {"status": "error", "error": "no frames to sample"}
            prompt = audit_prompt(description, subtasks)
            messages = frame_message(composite, [index + approach for index in indices], prompt)
            payload = request_json(
                client,
                messages,
                cache,
                model=self.vlm_config.model,
                prompt_version=self.prompt_version,
                schema=AUDIT_SCHEMA,
            )
            verdict = verdict_from_payload(payload)
        except (VLMError, ValueError, OSError) as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        verdict.update({
            "model": self.vlm_config.model,
            "prompt_version": self.prompt_version,
            "scope": "episode",
            "description": description,
        })
        return verdict

    def _description(self, src: Path, dst: Path, quality: dict) -> tuple[str, list[dict]]:
        """What the audit checks against, and the subtask context shown with it.

        Ground truth from the HDF5 beats a summary the VLM wrote itself; the annotation is only the
        reference for episodes that never carried a description.
        """
        summary, subtasks = "", []
        path = dst / ANNOTATION
        if path.is_file():
            try:
                annotation = json.loads(path.read_text())
                summary = str(annotation.get("episode_summary") or "").strip()
                raw = annotation.get("subtasks")
                if isinstance(raw, list):
                    subtasks = [
                        item
                        for item in raw
                        if isinstance(item, dict) and "start_frame" in item and "end_frame" in item
                    ]
            except (OSError, ValueError, TypeError):
                pass
        source = str(quality.get("description_source") or "")
        if source in ("hdf5", "mixed") and str(quality.get("description") or "").strip():
            return str(quality["description"]).strip(), subtasks
        if summary:
            return summary, subtasks
        return read_description(src, default="manipulation"), subtasks

    def _client(self) -> tuple[VLMClient | None, str]:
        if self.client is not None:
            return self.client, ""
        missing = self.vlm_config.missing()
        if missing:
            return None, missing
        return OpenAICompatibleClient(self.vlm_config), ""

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _geometry(quality: dict, dst: Path, composite: Path) -> tuple[int, float, int]:
    """Episode frames, fps, and the approach prefix length.

    Curate writes all three, but a quality.json from before `vlm_audit` existed does not carry them;
    those are rebuilt from `action/prefix.npz` and the composite itself so an old output directory
    still gets a real audit instead of a silent `keep_vlm_unavailable`.
    """
    video_frames, _height, _width, video_fps = video_shape(composite)
    approach = quality.get("approach")
    if approach is None:
        prefix = dst / PREFIX
        approach = len(np.load(prefix)["qpos"]) - 1 if prefix.is_file() else 0
    frames = int(quality.get("frames") or max(video_frames - int(approach), 0))
    fps = float(quality.get("fps") or video_fps or FPS)
    return frames, fps, int(approach)


def _section(config: dict | None, name: str) -> dict:
    values = copy.deepcopy(DEFAULTS[name])
    if isinstance(config, dict) and isinstance(config.get(name), dict):
        values.update(config[name])
    elif isinstance(config, dict) and any(key in config for key in values):
        values.update(config)
    return values
