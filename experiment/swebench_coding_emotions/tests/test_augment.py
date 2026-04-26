"""Tests for the augmented neutral set + PCA-denoising refit.

The model-loading path is exercised end-to-end at run time and is left out
of unit tests. We test the pure-numpy bits: the deterministic text
fixtures, structural-surface coverage, and the local PCA helper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emotion_probes.data import save_tensors
from swebench_coding_emotions import augment as AUG


# ---------------------------------------------------------------------------
# Fixture coverage
# ---------------------------------------------------------------------------

def test_build_augmented_neutral_texts_is_deterministic():
    a = AUG.build_augmented_neutral_texts()
    b = AUG.build_augmented_neutral_texts()
    assert a == b
    assert len(a) > 0


def test_augmented_neutral_texts_cover_each_structural_surface():
    """At least one stimulus per surface so the PCA can absorb each."""
    texts = AUG.build_augmented_neutral_texts()
    blob = "\n".join(texts)
    assert "<|im_start|>" in blob
    assert "<|im_end|>" in blob
    assert "<tool_call>" in blob
    assert "<tool_response>" in blob
    assert "```python" in blob
    assert "diff --git" in blob
    assert "<think>" in blob
    assert "</think>" in blob


def test_augmented_neutral_texts_have_no_overt_emotion_words():
    """Sanity check: no emotion-naming words in the neutral set. This is not
    exhaustive — it catches obvious mistakes if someone edits the templates."""
    texts = AUG.build_augmented_neutral_texts()
    blob = "\n".join(texts).lower()
    forbidden = [
        "frustrat", "angry", "anger", "happy", "sad", "anxious", "anxiety",
        "afraid", "fear", "disappoint", "overwhelm", "proud", "ashamed",
        "delighted", "excited", "worried", "annoyed",
    ]
    hits = [w for w in forbidden if w in blob]
    assert hits == [], f"emotion-laden words leaked into augmented neutrals: {hits}"


# ---------------------------------------------------------------------------
# PCA denoising helper
# ---------------------------------------------------------------------------

def test_denoise_with_pca_removes_dominant_neutral_direction():
    """Concentrate all neutral *variance* along axis 0; the helper should
    project that axis out of the emotion vectors. (PCA is fit on
    mean-centered neutrals, so it's the variance — not the mean — along an
    axis that drives which components get removed.)"""
    rng = np.random.default_rng(0)
    D = 8

    axis0 = rng.normal(0, 5.0, size=50).astype(np.float32)
    other = rng.normal(0, 0.01, size=(50, D - 1)).astype(np.float32)
    neutral = np.concatenate([axis0[:, None], other], axis=1)

    bad_dir = np.zeros(D, dtype=np.float32)
    bad_dir[0] = 1.0
    mixed = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    raw_vectors = np.stack([bad_dir, mixed])

    denoised, info = AUG._denoise_with_pca(raw_vectors, neutral, variance_threshold=0.5)

    # Pure-bad direction should be almost entirely removed.
    assert np.linalg.norm(denoised[0]) < 0.1
    # Mixed direction should retain its non-axis-0 component.
    assert abs(denoised[1, 1] - 1.0) < 1e-2
    assert info["n_components"] >= 1
    assert info["total_neutral_samples"] == 50


def test_denoise_with_pca_higher_threshold_removes_more_components():
    rng = np.random.default_rng(0)
    D = 6
    neutral = rng.normal(size=(40, D)).astype(np.float32)
    raw = rng.normal(size=(3, D)).astype(np.float32)

    _, info_low = AUG._denoise_with_pca(raw, neutral, variance_threshold=0.3)
    _, info_high = AUG._denoise_with_pca(raw, neutral, variance_threshold=0.9)
    assert info_high["n_components"] >= info_low["n_components"]


# ---------------------------------------------------------------------------
# run_augment requires a model — verify the precondition errors
# ---------------------------------------------------------------------------

def test_run_augment_raises_without_model(tmp_path: Path):
    probes_dir = tmp_path
    (probes_dir / "vectors").mkdir()
    (probes_dir / "activations").mkdir()
    save_tensors(
        {"raw_vectors": np.zeros((2, 4), dtype=np.float32)},
        probes_dir / "vectors" / "raw_vectors_layer_5.safetensors",
        metadata={"layer": "5", "emotions": json.dumps(["a", "b"])},
    )
    with pytest.raises(ValueError, match="EmotionProbeModel"):
        AUG.run_augment(
            probes_dir=probes_dir,
            layer=5,
            token_offset=50,
            batch_size=1,
            max_length=128,
            pca_variance_threshold=0.5,
            model=None,
        )


def test_run_augment_raises_when_raw_vectors_missing(tmp_path: Path):
    (tmp_path / "vectors").mkdir()
    (tmp_path / "activations").mkdir()
    with pytest.raises(FileNotFoundError, match="raw_vectors"):
        AUG.run_augment(
            probes_dir=tmp_path,
            layer=5,
            token_offset=50,
            batch_size=1,
            max_length=128,
            pca_variance_threshold=0.5,
            model=object(),  # never reached
        )
