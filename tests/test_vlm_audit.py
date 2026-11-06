"""L3 audit and the final keep decision, with no API key or GPU."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from egowhale.paths import ANNOTATION, COMPOSITE, FINAL, PREFIX, QUALITY, VLM_AUDIT
from egowhale.quality.finalize import decide
from egowhale.quality.vlm_audit import VlmAudit
from egowhale.vlm.client import MockClient, VLMError


def _out(tmp_path, *, keep: bool = True, frames: int = 4, quality: dict | None = None, where: str = "out"):
    out = tmp_path / where
    out.mkdir()
    given = {
        "keep_l1_l2": keep,
        "keep": keep,
        "frames": frames,
        "fps": 30.0,
        "approach": 0,
        "description": "Pick up the block.",
        "description_source": "hdf5",
    }
    given.update(quality or {})
    (out / QUALITY).parent.mkdir(parents=True, exist_ok=True)
    (out / QUALITY).write_text(json.dumps(given))
    (out / COMPOSITE).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / COMPOSITE), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return out


def _config(*, fail_open: bool = True, enabled: bool = True) -> dict:
    return {
        "audit": {
            "enabled": enabled,
            "sample_fps": 1,
            "max_frames": 4,
            "prompt_version": "audit_v1",
            "fail_open": fail_open,
            "scope": "episode",
        },
        "vlm": {"model": "mock", "api_key_env": "TEST_VLM_KEY", "cache": False},
    }


def test_consistent_episode_is_kept(tmp_path):
    out = _out(tmp_path)
    client = MockClient([json.dumps({"is_consistent": True, "confidence": 0.9, "reasoning": "matches"})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    final = json.loads((out / FINAL).read_text())
    assert audit["status"] == "ok" and audit["is_consistent"] is True
    assert final["keep"] is True and final["decision"] == "keep"


def test_explicit_mismatch_is_dropped(tmp_path):
    out = _out(tmp_path)
    client = MockClient([json.dumps({"is_consistent": False, "reasoning": "wrong object"})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    final = json.loads((out / FINAL).read_text())
    assert final["keep"] is False and final["decision"] == "drop_vlm"


def test_l1_l2_failure_skips_the_api(tmp_path):
    out = _out(tmp_path, keep=False)
    client = MockClient([])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    final = json.loads((out / FINAL).read_text())
    assert audit["status"] == "skipped" and not client.calls
    assert final["keep"] is False and final["decision"] == "drop_l1_l2"


def test_api_error_is_recorded_and_kept_when_fail_open(tmp_path):
    out = _out(tmp_path)
    client = MockClient([VLMError("timeout", kind="network")])
    VlmAudit(_config(fail_open=True), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    final = json.loads((out / FINAL).read_text())
    assert audit["status"] == "error"
    assert final["keep"] is True and final["decision"] == "keep_vlm_unavailable"


def test_api_error_is_dropped_when_fail_closed(tmp_path):
    out = _out(tmp_path)
    client = MockClient([VLMError("timeout", kind="network")])
    VlmAudit(_config(fail_open=False), client=client).run(tmp_path / "0.hdf5", out)
    final = json.loads((out / FINAL).read_text())
    assert final["keep"] is False and final["decision"] == "drop_vlm_unavailable"


def test_disabled_audit_keeps_l1_l2(tmp_path):
    out = _out(tmp_path)
    VlmAudit(_config(enabled=False), client=MockClient([])).run(tmp_path / "0.hdf5", out)
    final = json.loads((out / FINAL).read_text())
    assert final["keep"] is True and final["decision"] == "keep_audit_disabled"


def test_ok_and_skipped_replies_both_carry_an_error_field(tmp_path):
    out = _out(tmp_path, where="ok")
    client = MockClient([json.dumps({"is_consistent": True})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    assert audit["status"] == "ok" and audit["error"] is None

    out = _out(tmp_path, keep=False, where="skipped")
    VlmAudit(_config(), client=MockClient([])).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    assert audit["status"] == "skipped" and audit["error"] is None and audit["reason"]


def test_invalid_reply_gets_one_repair_round(tmp_path):
    out = _out(tmp_path)
    client = MockClient(["certainly! here you go", json.dumps({"is_consistent": True})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    assert audit["status"] == "ok" and audit["is_consistent"] is True
    assert len(client.calls) == 2


def test_old_quality_json_without_geometry_still_audits(tmp_path):
    # quality.json from before vlm_audit existed: no frames, fps, or approach fields.
    out = _out(tmp_path, frames=6)
    quality = json.loads((out / QUALITY).read_text())
    del quality["frames"], quality["fps"], quality["approach"]
    (out / QUALITY).write_text(json.dumps(quality))
    (out / PREFIX).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out / PREFIX, qpos=np.zeros((3, 7)))
    client = MockClient([json.dumps({"is_consistent": True})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    assert audit["status"] == "ok" and audit["error"] is None


def test_ground_truth_description_beats_the_vlm_summary(tmp_path):
    out = _out(tmp_path)
    (out / ANNOTATION).parent.mkdir(parents=True, exist_ok=True)
    (out / ANNOTATION).write_text(json.dumps({
        "episode_summary": "VLM guess",
        "subtasks": [{"start_frame": 0, "end_frame": 4, "instruction": "reach"}],
    }))
    client = MockClient([json.dumps({"is_consistent": True})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    prompt = client.calls[0][0]["content"][-1]["text"]
    assert "Pick up the block." in prompt
    assert "VLM guess" not in prompt
    assert "reach" in prompt  # subtasks stay as context either way


def test_unlabelled_episode_falls_back_to_the_vlm_summary(tmp_path):
    out = _out(tmp_path, quality={"description": "manipulation", "description_source": "default"})
    (out / ANNOTATION).parent.mkdir(parents=True, exist_ok=True)
    (out / ANNOTATION).write_text(json.dumps({"episode_summary": "VLM guess", "subtasks": []}))
    client = MockClient([json.dumps({"is_consistent": True})])
    VlmAudit(_config(), client=client).run(tmp_path / "0.hdf5", out)
    prompt = client.calls[0][0]["content"][-1]["text"]
    assert "Task Description: VLM guess" in prompt


@pytest.mark.parametrize(
    "keep_l1,status,consistent,fail_open,decision",
    [
        (False, "skipped", None, True, "drop_l1_l2"),
        (True, "ok", True, True, "keep"),
        (True, "ok", False, True, "drop_vlm"),
        (True, "skipped", None, True, "keep_vlm_unavailable"),
        (True, "error", None, False, "drop_vlm_unavailable"),
        (True, "disabled", None, True, "keep_audit_disabled"),
    ],
)
def test_decide_matrix(keep_l1, status, consistent, fail_open, decision):
    quality = {"keep_l1_l2": keep_l1}
    audit = {"status": status, "is_consistent": consistent}
    assert decide(quality, audit, fail_open=fail_open)["decision"] == decision


def test_decide_reasons_distinguish_a_missing_verdict():
    missing = decide({"keep_l1_l2": True}, {"status": "ok", "is_consistent": None})
    agree = decide({"keep_l1_l2": True}, {"status": "ok", "is_consistent": True})
    assert missing["decision"] == agree["decision"] == "keep"
    assert "no verdict" in missing["reason"]
    assert "agree" in agree["reason"]
