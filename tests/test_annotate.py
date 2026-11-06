"""Coarse annotation modes with a mock client. No API key or GPU."""

from __future__ import annotations

import json

import cv2
import h5py
import numpy as np
import pytest

from egowhale.annotate import Annotate
from egowhale.paths import ANNOTATION
from egowhale.vlm.client import MockClient, VLMError


def _episode(tmp_path, description: str | None = None, frames: int = 4):
    src = tmp_path / "task" / "0.hdf5"
    src.parent.mkdir(parents=True)
    with h5py.File(src, "w") as handle:
        handle["transforms/camera"] = np.repeat(np.eye(4)[None], frames, axis=0)
        if description is not None:
            handle.attrs["llm_description"] = description
    writer = cv2.VideoWriter(str(src.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return src


def _config(mode: str, *, on_error: str = "keep") -> dict:
    return {
        "annotate": {
            "mode": mode,
            "sample_fps": 1,
            "max_frames": 4,
            "max_subtasks": 8,
            "min_subtask_frames": 1,
            "prompt_version": "coarse_v1",
            "on_error": on_error,
        },
        "vlm": {"model": "mock", "api_key_env": "TEST_VLM_KEY", "cache": False},
    }


def _run(tmp_path, src, config, client):
    dst = tmp_path / "out"
    Annotate(config, client=client).run(src, dst)
    return json.loads((dst / ANNOTATION).read_text())


def test_both_mode_writes_summary_and_subtasks(tmp_path):
    src = _episode(tmp_path)
    reply = json.dumps({
        "episode_summary": "Pick up the block.",
        "subtasks": [
            {"start_frame": 0, "end_frame": 2, "instruction": "reach"},
            {"start_frame": 2, "end_frame": 4, "instruction": "lift"},
        ],
    })
    report = _run(tmp_path, src, _config("both"), MockClient([reply]))
    assert report["status"] == "ok"
    assert report["episode_summary"] == "Pick up the block."
    assert report["meta"]["mode"] == "both"
    assert [(item["start_frame"], item["end_frame"]) for item in report["subtasks"]] == [(0, 2), (2, 4)]


def test_existing_description_is_reused_without_a_call(tmp_path):
    src = _episode(tmp_path, description="Open the drawer.")
    report = _run(tmp_path, src, _config("description"), MockClient([]))
    assert report["status"] == "ok"
    assert report["episode_summary"] == "Open the drawer."
    assert report["meta"]["description_source"] == "hdf5"


def test_both_with_existing_description_only_asks_for_subtasks(tmp_path):
    src = _episode(tmp_path, description="Open the drawer.")
    client = MockClient([json.dumps({"subtasks": [{"start_frame": 0, "end_frame": 4, "instruction": "pull"}]})])
    report = _run(tmp_path, src, _config("both"), client)
    assert report["episode_summary"] == "Open the drawer."
    assert report["meta"]["description_source"] == "hdf5"
    assert len(client.calls) == 1


def test_bad_reply_becomes_error_when_on_error_is_keep(tmp_path):
    src = _episode(tmp_path)
    # Two bad replies: the second one answers the repair round.
    report = _run(tmp_path, src, _config("both"), MockClient(["not json", "still not json"]))
    assert report["status"] == "error"
    assert report["error"]


def test_bad_reply_raises_when_on_error_is_fail(tmp_path):
    src = _episode(tmp_path)
    with pytest.raises(VLMError):
        _run(tmp_path, src, _config("both", on_error="fail"), MockClient(["not json", "still not json"]))


def test_off_mode_makes_a_disabled_placeholder(tmp_path):
    src = _episode(tmp_path, description="Open the drawer.")
    report = _run(tmp_path, src, _config("off"), MockClient([]))
    assert report["status"] == "disabled"
    assert report["episode_summary"] == "Open the drawer."


def test_no_client_and_no_key_is_skipped(tmp_path, monkeypatch):
    src = _episode(tmp_path)
    monkeypatch.delenv("TEST_VLM_KEY", raising=False)
    config = _config("both")
    config["vlm"]["model"] = ""
    dst = tmp_path / "out"
    Annotate(config).run(src, dst)
    report = json.loads((dst / ANNOTATION).read_text())
    assert report["status"] == "skipped"
    assert "model" in report["error"]


def test_decode_failure_is_an_error_not_a_crash(tmp_path, monkeypatch):
    from egowhale.annotate import coarse

    def boom(*_args, **_kwargs):
        raise ValueError("decode failed")

    monkeypatch.setattr(coarse, "frame_message", boom)
    src = _episode(tmp_path)
    report = _run(tmp_path, src, _config("both"), MockClient([]))
    assert report["status"] == "error"
    assert "decode failed" in report["error"]
