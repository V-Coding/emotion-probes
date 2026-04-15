"""Data I/O helpers, checkpointing, and topic hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file


# ---------------------------------------------------------------------------
# Topic hashing — deterministic, filesystem-safe identifier for topic strings
# ---------------------------------------------------------------------------

def topic_hash(topic: str) -> str:
    """Return the first 8 hex chars of the SHA-256 of *topic*."""
    return hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_json_write(payload: Any, path: Path) -> None:
    """Write JSON atomically: write to a temp file, then rename.

    This prevents corrupt files if the process is interrupted mid-write,
    which would otherwise block resumability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # Atomic rename (same filesystem)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Story I/O — per-(emotion, topic) JSON files
# ---------------------------------------------------------------------------

def stories_path(output_dir: Path, emotion: str, topic: str) -> Path:
    """Return the path where stories for (emotion, topic) are saved."""
    return output_dir / "stories" / emotion / f"{topic_hash(topic)}.json"


def save_stories(
    output_dir: Path,
    emotion: str,
    topic: str,
    stories: list[str],
    model_name: str,
    n_requested: int | None = None,
) -> Path:
    """Save generated stories to a JSON file. Returns the path written."""
    path = stories_path(output_dir, emotion, topic)
    payload = {
        "emotion": emotion,
        "topic": topic,
        "stories": stories,
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_requested": n_requested if n_requested is not None else len(stories),
            "n_parsed": len(stories),
        },
    }
    _atomic_json_write(payload, path)
    return path


def load_stories(output_dir: Path, emotion: str, topic: str) -> list[str]:
    """Load previously saved stories for (emotion, topic)."""
    path = stories_path(output_dir, emotion, topic)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["stories"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Malformed story file {path}: {e}") from e


def stories_exist(output_dir: Path, emotion: str, topic: str) -> bool:
    """Check whether stories for (emotion, topic) have already been generated."""
    return stories_path(output_dir, emotion, topic).exists()


# ---------------------------------------------------------------------------
# Neutral dialogue I/O — per-topic JSON files
# ---------------------------------------------------------------------------

def neutral_path(output_dir: Path, topic: str) -> Path:
    """Return the path where neutral dialogues for a topic are saved."""
    return output_dir / "neutral" / f"{topic_hash(topic)}.json"


def save_neutral_dialogues(
    output_dir: Path,
    topic: str,
    dialogues: list[str],
    model_name: str,
    n_requested: int | None = None,
) -> Path:
    """Save generated neutral dialogues to a JSON file. Returns the path written."""
    path = neutral_path(output_dir, topic)
    payload = {
        "topic": topic,
        "dialogues": dialogues,
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_requested": n_requested if n_requested is not None else len(dialogues),
            "n_parsed": len(dialogues),
        },
    }
    _atomic_json_write(payload, path)
    return path


def load_neutral_dialogues(output_dir: Path, topic: str) -> list[str]:
    """Load previously saved neutral dialogues for a topic."""
    path = neutral_path(output_dir, topic)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["dialogues"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Malformed neutral dialogue file {path}: {e}") from e


def neutral_exists(output_dir: Path, topic: str) -> bool:
    """Check whether neutral dialogues for a topic have been generated."""
    return neutral_path(output_dir, topic).exists()


# ---------------------------------------------------------------------------
# Tensor I/O — safetensors with metadata
# ---------------------------------------------------------------------------

def save_tensors(
    tensors: dict[str, np.ndarray],
    path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save a dict of named numpy arrays to a safetensors file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    """Load a dict of named numpy arrays from a safetensors file."""
    return load_file(str(path))


def load_vectors_with_metadata(path: Path) -> tuple[np.ndarray, list[str], int]:
    """Load emotion vectors and their metadata from a safetensors file.

    Returns:
        (vectors array, emotion_names list, layer index)
    """
    from safetensors import safe_open

    try:
        tensors = load_file(str(path))
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata()
        vectors = tensors["emotion_vectors"]
        emotion_names = json.loads(meta["emotions"])
        layer = int(meta["layer"])
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Malformed vector file {path}: {e}") from e
    return vectors, emotion_names, layer


# ---------------------------------------------------------------------------
# JSON metadata I/O
# ---------------------------------------------------------------------------

def save_metadata(data: dict[str, Any], path: Path) -> None:
    """Save a JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load a JSON metadata file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
