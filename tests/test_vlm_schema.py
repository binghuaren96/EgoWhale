"""Pure JSON parsing and subtask normalisation."""

from __future__ import annotations

from egowhale.vlm.schema import extract_json, normalize_subtasks, verdict_from_text


def test_extract_json_accepts_a_code_fence_and_surrounding_text():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```\n') == {"a": 1}


def test_verdict_reads_the_old_and_new_key_spellings():
    old = verdict_from_text('{"is consistent": false, "confidence": "0.8", "reasoning": "wrong object"}')
    assert old["status"] == "ok" and old["is_consistent"] is False and old["confidence"] == 0.8
    new = verdict_from_text('{"is_consistent": true, "mismatches": []}')
    assert new["is_consistent"] is True
    assert verdict_from_text("not json")["status"] == "error"


def test_subtasks_fill_gaps_and_remove_overlaps():
    raw = [
        {"start_frame": 20, "end_frame": 30, "instruction": "place"},
        {"start_frame": 0, "end_frame": 10, "instruction": "reach"},
    ]
    out = normalize_subtasks(raw, frames=30, fps=30, max_subtasks=8, min_subtask_frames=1)
    assert [(item["start_frame"], item["end_frame"]) for item in out] == [(0, 20), (20, 30)]

    overlapped = [
        {"start_frame": 0, "end_frame": 20, "instruction": "a"},
        {"start_frame": 10, "end_frame": 30, "instruction": "b"},
    ]
    out = normalize_subtasks(overlapped, frames=30, fps=30, max_subtasks=8, min_subtask_frames=1)
    assert [(item["start_frame"], item["end_frame"]) for item in out] == [(0, 20), (20, 30)]


def test_short_segments_merge_and_coverage_is_enforced():
    raw = [{"start_frame": 0, "end_frame": 5, "instruction": "tiny"}]
    out = normalize_subtasks(raw, frames=100, fps=30, max_subtasks=8, min_subtask_frames=15)
    assert len(out) == 1
    assert out[0]["start_frame"] == 0 and out[0]["end_frame"] == 100


def test_too_many_subtasks_are_merged_and_seconds_are_converted():
    raw = [
        {"start_time": 0.0, "end_time": 1.0, "instruction": "a"},
        {"start_time": 1.0, "end_time": 2.0, "instruction": "b"},
        {"start_time": 2.0, "end_time": 3.0, "instruction": "c"},
    ]
    out = normalize_subtasks(raw, frames=90, fps=30, max_subtasks=2, min_subtask_frames=1)
    assert len(out) == 2
    assert out[0]["start_frame"] == 0 and out[-1]["end_frame"] == 90
