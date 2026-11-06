"""Batch configuration: experiment name, output directory, episodes, node pools, and memory limits.

A file in cfgs/ only lists what it changes; everything else comes from DEFAULTS.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from egowhale.paths import ROOT
from egowhale.pieces import PIECE


class ConfigError(ValueError):
    """A config that cannot run. The command line prints it and exits."""


DEFAULTS = {
    "name": "run",
    # A .txt list of HDF5 paths, an HDF5 file, or a dataset directory; or a list of those.
    "episodes": None,
    # Results go to <out>/<name>/. Relative paths are taken from the repository root.
    "out": "outputs",
    # 0 runs every episode.
    "limit": 0,
    # "list" keeps the listed order; "shuffle" mixes long and short episodes evenly, with `seed`.
    "order": "list",
    "seed": 0,
    # Longer episodes are split into equal pieces of at most this many frames.
    "piece_frames": PIECE,
    "nodes": {
        # SAM 3 + Cutie. Actors on one card share it evenly.
        "segment": {"cards": 1, "actors_per_card": 1},
        # One actor per card. `model`: sttn (fast) or propainter (slower, sharper).
        "inpaint": {"cards": 1, "model": "sttn"},
        # DA3-GIANT. `window` views share one DA3 call; `frames_per_forward` frames go in one forward.
        "depth": {"cards": 1, "window": 1, "frames_per_forward": 96},
        # cuRobo. `ride: inpaint|depth` puts `actors_per_card` actors on every card of that node,
        # each holding `gpu_share` of it; otherwise the action node gets `cards` cards of its own.
        "action": {"ride": None, "cards": 1, "actors_per_card": 1, "gpu_share": 0.1},
        # MuJoCo EGL render actors. They draw on the action cards.
        "render": {"actors": 4},
        # Composite and curate actors.
        "cpu": {"actors": 8},
        # Coarse annotation and L3 audit; these actors make network calls, not GPU work.
        "vlm": {"actors": 4},
    },
    # Provider settings shared by annotate and vlm_audit. The key is never read from this file.
    "vlm": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "",
        "api_key_env": "EGOWHALE_VLM_API_KEY",
        "timeout_s": 120,
        "retries": 2,
        "backoff_s": 2.0,
        "cache": True,
    },
    "annotate": {
        # off keeps batch runs from calling the API by default; a config that wants coarse labels sets this.
        "mode": "off",
        "sample_fps": 1.0,
        "max_frames": 24,
        "max_subtasks": 8,
        "min_subtask_frames": 15,
        "prompt_version": "coarse_v1",
        "on_error": "keep",
    },
    "audit": {
        "enabled": True,
        "sample_fps": 4.0,
        "max_frames": 96,
        "prompt_version": "audit_v1",
        "fail_open": True,
        "scope": "episode",
    },
    "io": {
        # CPU decoders for each visual GPU actor.
        "readers_per_lane": 2,
        # Pieces one visual GPU actor may hold: decoding, decoded, and on the GPU.
        "queue_per_lane": 4,
        # Frames one visual GPU actor may hold across those pieces. One piece always fits.
        "queue_frames": 1200,
        # CPUs each actor reserves for its whole life. Ray queues actors silently when they do not fit.
        "cpus": {"cpu": 1, "render": 1, "feed": 2, "write": 1, "action_read": 1, "action_write": 1, "vlm": 1},
    },
    "memory": {
        # Episodes open at once. Open episodes cost disk, not RAM; this only sets how far fast nodes run ahead.
        "inflight": 64,
        # No new piece is decoded while the host has less than this free.
        "reserve_gb": 64,
        # Ray shared memory. Beyond it objects spill to disk instead of growing RAM. 0 keeps Ray's default.
        "object_store_gb": 120,
    },
}


def load(path: Path | str, **overrides) -> dict:
    """Defaults, then the file, then non-None keyword overrides of top-level keys."""
    path = Path(path)
    given = yaml.safe_load(path.read_text()) or {}
    config = _merge(copy.deepcopy(DEFAULTS), given, "")
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if not config["episodes"]:
        raise ConfigError(f"{path}: no episodes")
    from egowhale.visual.inpaint import MODELS

    if config["nodes"]["inpaint"]["model"] not in MODELS:
        raise ConfigError(f"{path}: nodes.inpaint.model must be one of {', '.join(MODELS)}")
    action = config["nodes"]["action"]
    if action["ride"] not in (None, "inpaint", "depth"):
        raise ConfigError(f"{path}: nodes.action.ride must be inpaint, depth, or empty")
    if action["ride"] and action["actors_per_card"] * action["gpu_share"] >= 1:
        raise ConfigError(f"{path}: riding action actors would take the whole card")
    if config["annotate"]["mode"] not in ("off", "description", "subtasks", "both"):
        raise ConfigError(f"{path}: annotate.mode must be off, description, subtasks, or both")
    if config["annotate"]["on_error"] not in ("keep", "fail"):
        raise ConfigError(f"{path}: annotate.on_error must be keep or fail")
    for key in ("sample_fps", "max_frames", "max_subtasks", "min_subtask_frames"):
        if float(config["annotate"][key]) <= 0:
            raise ConfigError(f"{path}: annotate.{key} must be positive")
    if config["audit"]["scope"] != "episode":
        raise ConfigError(f"{path}: audit.scope must be episode")
    if float(config["audit"]["sample_fps"]) <= 0 or int(config["audit"]["max_frames"]) <= 0:
        raise ConfigError(f"{path}: audit.sample_fps and audit.max_frames must be positive")
    if int(config["vlm"]["retries"]) < 0 or float(config["vlm"]["timeout_s"]) <= 0:
        raise ConfigError(f"{path}: vlm.retries must be >= 0 and vlm.timeout_s must be positive")
    if not config["vlm"]["api_key_env"]:
        raise ConfigError(f"{path}: vlm.api_key_env must name an environment variable")
    if int(config["nodes"]["vlm"]["actors"]) < 1:
        raise ConfigError(f"{path}: nodes.vlm.actors must be at least 1")
    config["out"] = str(_resolve(config["out"]) / config["name"])
    config["source"] = str(path.resolve())
    return config


def episodes(config: dict) -> list[Path]:
    entries = config["episodes"]
    found: list[Path] = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = _resolve(entry)
        if path.suffix == ".txt":
            found.extend(Path(line.strip()).expanduser() for line in path.read_text().splitlines() if line.strip())
        elif path.is_dir():
            found.extend(sorted(item for item in path.rglob("*.hdf5") if item.with_suffix(".mp4").is_file()))
        else:
            found.append(path)
    if config["order"] == "shuffle":
        import random

        random.Random(config["seed"]).shuffle(found)
    elif config["order"] != "list":
        raise ConfigError(f"order must be list or shuffle, not {config['order']}")
    if config["limit"]:
        found = found[: config["limit"]]
    return [path.resolve() for path in found]


def _resolve(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _merge(base: dict, given: dict, where: str) -> dict:
    for key, value in given.items():
        if key not in base:
            raise ConfigError(f"unknown config key {where}{key}")
        if isinstance(base[key], dict) and base[key] and isinstance(value, dict):
            _merge(base[key], value, f"{where}{key}.")
        else:
            base[key] = value
    return base
