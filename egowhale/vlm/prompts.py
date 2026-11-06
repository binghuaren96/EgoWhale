"""Prompt templates. Their version strings are part of the output metadata and cache key."""

from __future__ import annotations

from egowhale.vlm.sampling import frame_list

COARSE_VERSION = "coarse_v1"
AUDIT_VERSION = "audit_v1"

# The reply shapes, shown to the model again when its first reply is not valid JSON.
DESCRIPTION_SCHEMA = '{"episode_summary": "..."}'
SUBTASKS_SCHEMA = '{"subtasks": [{"start_frame": 0, "end_frame": 120, "instruction": "..."}]}'
BOTH_SCHEMA = '{"episode_summary": "...", "subtasks": [{"start_frame": 0, "end_frame": 120, "instruction": "..."}]}'
AUDIT_SCHEMA = '{"is_consistent": true, "confidence": 0.0, "reasoning": "...", "mismatches": []}'

_RULES = """Rules:
- The video is a first-person manipulation recording.
- Use coarse boundaries; this is not frame-accurate segmentation.
- `start_frame` is inclusive and `end_frame` is exclusive.
- Subtasks must cover the whole video in order, without gaps or overlaps.
- Return only one JSON object, with no prose and no code fence."""


def description_prompt(frames: int, fps: float, indices: list[int]) -> str:
    return f"""You are annotating an egocentric manipulation video.
The video has {frames} frames at {fps:.1f} fps.
Sampled frames: {frame_list(indices, fps)}.

Describe the whole episode in one or two concrete sentences: the object, the action, and the target.
Return one JSON object only:
{{"episode_summary": "..."}}

{_RULES}"""


def subtasks_prompt(frames: int, fps: float, indices: list[int], description: str = "") -> str:
    context = f"\nKnown episode description: {description}\n" if description.strip() else ""
    return f"""You are annotating an egocentric manipulation video.
The video has {frames} frames at {fps:.1f} fps.
Sampled frames: {frame_list(indices, fps)}.
{context}
Split the episode into 3 to 8 coarse subtasks.
Return one JSON object only:
{{"subtasks": [{{"start_frame": 0, "end_frame": 120, "instruction": "..."}}]}}

{_RULES}"""


def both_prompt(frames: int, fps: float, indices: list[int]) -> str:
    return f"""You are annotating an egocentric manipulation video.
The video has {frames} frames at {fps:.1f} fps.
Sampled frames: {frame_list(indices, fps)}.

Describe the episode and split it into 3 to 8 coarse subtasks.
Return one JSON object only:
{{
  "episode_summary": "one or two sentences",
  "subtasks": [{{"start_frame": 0, "end_frame": 120, "instruction": "..."}}]
}}

{_RULES}"""


def audit_prompt(description: str, subtasks: list[dict] | None = None) -> str:
    context = ""
    if subtasks:
        lines = [
            f"- frames [{item['start_frame']}, {item['end_frame']}): {item.get('instruction', '')}"
            for item in subtasks
        ]
        context = "\nCoarse subtask context:\n" + "\n".join(lines) + "\n"
    return f"""You are evaluating whether a manipulation video matches its text description.
This is a ROBOT manipulation dataset. "Hand" refers to the robot's gripper/end-effector.
Many tasks use FAKE or SIMULATED objects as stand-ins for real objects. This is EXPECTED.
Task Description: {description}
{context}
Determine if the robot's actions match the described task.
Flag MAJOR MISMATCHES: wrong action type, wrong object category, wrong target location, or failed execution.
Be tolerant of: fake/toy objects, minor appearance variations, small spatial deviations, different grasping approaches.
Return one JSON object only:
{{"is_consistent": true, "confidence": 0.0, "reasoning": "...", "mismatches": []}}"""


def repair_prompt(schema: str) -> str:
    return f"""Your previous reply was not valid JSON for this schema.
Return only one JSON object matching this shape, with no prose and no code fence:
{schema}"""
