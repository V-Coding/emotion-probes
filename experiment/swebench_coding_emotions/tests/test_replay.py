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

def test_save_load_roundtrip(tmp_path: Path):
    T, K = 7, 3
    result = RP.ReplayResult(
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
    )
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
    np.testing.assert_array_equal(loaded.scores, result.scores)
    np.testing.assert_array_equal(loaded.cosine, result.cosine)
    np.testing.assert_array_equal(loaded.token_ids, result.token_ids)
    np.testing.assert_array_equal(loaded.char_starts, result.char_starts)
    np.testing.assert_array_equal(loaded.char_ends, result.char_ends)


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

    v, emos, g = RP.load_probes(probes_dir, layer=5, use_raw=False)
    np.testing.assert_array_equal(v, vecs)
    np.testing.assert_array_equal(g, gm)
    assert emos == ["a", "b", "c", "d"]
