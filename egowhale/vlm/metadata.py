"""Read the task description EgoDex already carries in the HDF5 attributes."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def read_description(path: Path | str, default: str = "") -> str:
    return description_and_source(path, default)[0]


def description_and_source(path: Path | str, default: str = "") -> tuple[str, str]:
    """The first non-empty `llm_description` or `task` attribute, and where it came from."""
    with h5py.File(path, "r") as handle:
        for key in ("llm_description", "task"):
            if key not in handle.attrs:
                continue
            value = handle.attrs[key]
            if isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            text = str(value).strip()
            if text and text.lower() != "none":
                return text, "hdf5"
    return default, "default"
