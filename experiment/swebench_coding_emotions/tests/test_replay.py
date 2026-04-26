"""Unit tests for replay.py helpers that don't require a GPU / real model.

The full sliding-window replay is exercised by a tiny toy-model integration
test (guarded by `pytest.importorskip('torch')`). If torch isn't installed
the test is skipped rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swebench_coding_emotions import replay as RP


# ---------------------------------------------------------------------------
# Section markers
# ---------------------------------------------------------------------------

def test_find_section_markers_detects_think_and_diff():
    txt = "before <think>planning</think> middle diff --git a/x b/x\n+foo\n"
    think_end, patch_start = RP._find_section_markers(txt)
    assert think_end == txt.index("</think>") + len("</think>")
    assert patch_start == txt.index("diff --git")


def test_find_section_markers_returns_negative_when_absent():
    think_end, patch_start = RP._find_section_markers("plain text, no markers")
    assert think_end == -1
    assert patch_start == -1


def test_find_section_markers_uses_last_think_block():
    txt = "<think>one</think> stuff <think>two</think> tail"
    think_end, _ = RP._find_section_markers(txt)
    # Last </think> ends at position of "two</think>" + len(</think>)
    expected = txt.rindex("</think>") + len("</think>")
    assert think_end == expected


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_and_config_sensitive():
    a = RP.compute_fingerprint("m", "bfloat16", "4bit", 42)
    b = RP.compute_fingerprint("m", "bfloat16", "4bit", 42)
    c = RP.compute_fingerprint("m", "bfloat16", None, 42)
    d = RP.compute_fingerprint("m", "bfloat16", "4bit", 43)
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 16


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def _make_replay_result(T: int = 7, K: int = 3) -> RP.ReplayResult:
    return RP.ReplayResult(
        instance_id="demo-1",
        scores=np.random.randn(T, K).astype(np.float16),
        cosine=np.random.randn(T, K).astype(np.float16),
        token_ids=np.arange(T, dtype=np.int32),
        char_starts=np.arange(T, dtype=np.int32),
        char_ends=np.arange(1, T + 1, dtype=np.int32),
        layer=42,
        emotions=["a", "b", "c"],
        resolved=True,
        difficulty="<15 min fix",
        thinking_end_char=5,
        patch_start_char=-1,
        model_fingerprint="deadbeefdeadbeef",
        special_token_ids=[1, 2],
        probe_variant="augmented",
    )


def test_save_load_roundtrip(tmp_path: Path):
    result = _make_replay_result()
    out = tmp_path / "demo.safetensors"
    RP.save_replay(result, out)
    loaded = RP.load_replay(out)

    assert loaded.instance_id == "demo-1"
    assert loaded.layer == 42
    assert loaded.emotions == ["a", "b", "c"]
    assert loaded.resolved is True
    assert loaded.difficulty == "<15 min fix"
    assert loaded.thinking_end_char == 5
    assert loaded.patch_start_char == -1
    assert loaded.model_fingerprint == "deadbeefdeadbeef"
    assert loaded.special_token_ids == [1, 2]
    assert loaded.probe_variant == "augmented"
    np.testing.assert_array_equal(loaded.scores, result.scores)
    np.testing.assert_array_equal(loaded.cosine, result.cosine)
    np.testing.assert_array_equal(loaded.token_ids, result.token_ids)
    np.testing.assert_array_equal(loaded.char_starts, result.char_starts)
    np.testing.assert_array_equal(loaded.char_ends, result.char_ends)


def test_load_replay_back_compat_when_metadata_missing(tmp_path: Path):
    """Files written before the special_token_ids/probe_variant fields existed
    should still load — metadata fall-back paths must work."""
    from emotion_probes.data import save_tensors as _save_tensors

    T, K = 4, 2
    out = tmp_path / "old.safetensors"
    _save_tensors(
        {
            "scores": np.zeros((T, K), dtype=np.float16),
            "cosine": np.zeros((T, K), dtype=np.float16),
            "token_ids": np.arange(T, dtype=np.int32),
            "char_starts": np.arange(T, dtype=np.int32),
            "char_ends": np.arange(1, T + 1, dtype=np.int32),
        },
        out,
        metadata={
            "instance_id": "old-1",
            "layer": "42",
            "emotions": '["a","b"]',
            "resolved": "0",
            "difficulty": "1-4 hours",
            "thinking_end_char": "-1",
            "patch_start_char": "-1",
            "model_fingerprint": "00",
            # no special_token_ids, no probe_variant — emulating an older file
        },
    )
    loaded = RP.load_replay(out)
    assert loaded.special_token_ids == []
    assert loaded.probe_variant == "denoised"


# ---------------------------------------------------------------------------
# load_probes — round-trip against a synthetic output layout
# ---------------------------------------------------------------------------

def test_load_probes_reads_denoised_and_global_mean(tmp_path: Path):
    import json
    from emotion_probes.data import save_tensors

    probes_dir = tmp_path
    vectors_dir = probes_dir / "vectors"
    act_dir = probes_dir / "activations"
    vectors_dir.mkdir(parents=True)
    act_dir.mkdir(parents=True)

    vecs = np.random.randn(4, 8).astype(np.float32)
    gm = np.random.randn(8).astype(np.float32)

    save_tensors(
        {"emotion_vectors": vecs},
        vectors_dir / "emotion_vectors_layer_5.safetensors",
        metadata={"layer": "5", "emotions": json.dumps(["a", "b", "c", "d"])},
    )
    save_tensors(
        {"global_mean": gm},
        act_dir / "global_mean_layer_5.safetensors",
    )

    v, emos, g = RP.load_probes(probes_dir, layer=5, variant="denoised")
    np.testing.assert_array_equal(v, vecs)
    np.testing.assert_array_equal(g, gm)
    assert emos == ["a", "b", "c", "d"]


def test_load_probes_augmented_variant(tmp_path: Path):
    import json
    from emotion_probes.data import save_tensors

    probes_dir = tmp_path
    vectors_dir = probes_dir / "vectors"
    act_dir = probes_dir / "activations"
    vectors_dir.mkdir(parents=True)
    act_dir.mkdir(parents=True)

    vecs = np.random.randn(3, 6).astype(np.float32)
    gm = np.random.randn(6).astype(np.float32)

    save_tensors(
        {"emotion_vectors": vecs},
        vectors_dir / "emotion_vectors_augmented_layer_7.safetensors",
        metadata={"layer": "7", "emotions": json.dumps(["a", "b", "c"]), "variant": "augmented"},
    )
    save_tensors(
        {"global_mean": gm},
        act_dir / "global_mean_layer_7.safetensors",
    )

    v, emos, g = RP.load_probes(probes_dir, layer=7, variant="augmented")
    np.testing.assert_array_equal(v, vecs)
    np.testing.assert_array_equal(g, gm)
    assert emos == ["a", "b", "c"]


def test_load_probes_unknown_variant_raises(tmp_path: Path):
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Unknown variant"):
        RP.load_probes(tmp_path, layer=1, variant="nonsense")


def test_load_probes_missing_augmented_file_raises(tmp_path: Path):
    import pytest as _pytest
    (tmp_path / "vectors").mkdir()
    (tmp_path / "activations").mkdir()
    with _pytest.raises(FileNotFoundError, match="augment"):
        RP.load_probes(tmp_path, layer=1, variant="augmented")


def test_load_probes_raw_variant_layer_mismatch_raises(tmp_path: Path):
    """The raw branch should refuse files whose metadata claims a
    different layer than was requested."""
    import json as _json
    import pytest as _pytest
    from emotion_probes.data import save_tensors as _save

    (tmp_path / "vectors").mkdir()
    (tmp_path / "activations").mkdir()
    _save(
        {"raw_vectors": np.zeros((2, 3), dtype=np.float32)},
        tmp_path / "vectors" / "raw_vectors_layer_5.safetensors",
        metadata={"layer": "5", "emotions": _json.dumps(["a", "b"])},
    )
    _save(
        {"global_mean": np.zeros(3, dtype=np.float32)},
        tmp_path / "activations" / "global_mean_layer_5.safetensors",
    )
    # Sanity: matching layer loads cleanly.
    v, _, _ = RP.load_probes(tmp_path, layer=5, variant="raw")
    assert v.shape == (2, 3)
    # Tamper: write a file at layer 7 whose metadata claims it's layer 5.
    _save(
        {"raw_vectors": np.zeros((2, 3), dtype=np.float32)},
        tmp_path / "vectors" / "raw_vectors_layer_7.safetensors",
        metadata={"layer": "5", "emotions": _json.dumps(["a", "b"])},
    )
    _save(
        {"global_mean": np.zeros(3, dtype=np.float32)},
        tmp_path / "activations" / "global_mean_layer_7.safetensors",
    )
    with _pytest.raises(ValueError, match="Layer mismatch"):
        RP.load_probes(tmp_path, layer=7, variant="raw")


def test_read_replay_variant_returns_none_for_missing_metadata(tmp_path: Path):
    """`read_replay_variant` should return None for files without the
    metadata field — used by the replay CLI to gate the variant-mismatch
    refusal logic."""
    from emotion_probes.data import save_tensors as _save
    p = tmp_path / "old.safetensors"
    _save(
        {"scores": np.zeros((1, 1), dtype=np.float16)},
        p,
        metadata={"instance_id": "x"},
    )
    assert RP.read_replay_variant(p) is None


def test_read_replay_variant_returns_metadata_value(tmp_path: Path):
    result = _make_replay_result()
    p = tmp_path / "demo.safetensors"
    RP.save_replay(result, p)
    assert RP.read_replay_variant(p) == "augmented"
