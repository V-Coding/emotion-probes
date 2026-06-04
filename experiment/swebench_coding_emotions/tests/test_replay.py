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
# Span-based section markers (multi-turn R2E-agent format)
# ---------------------------------------------------------------------------

def _two_turn_transcript() -> str:
    # Turn 0: closes </think>, then an editing file_editor call (str_replace).
    # Turn 1: never closes </think> (the common case), then a read-only view call.
    return (
        "System prompt: wrap reasoning in <think> tags and write Thought: first.\n"
        "================\nStep 0:\n\nThought:\n\n<think>\nreason one\n</think>\n"
        "Action:\n\n<function=file_editor>\n  <parameter=command>str_replace</parameter>\n"
        "  <parameter=old_str>a</parameter>\n  <parameter=new_str>b</parameter>\n</function>\n"
        "Observation:\n\nedit applied\n"
        "================\nStep 1:\n\nThought:\n\n<think>\nreason two, no close tag\n"
        "Action:\n\n<function=file_editor>\n  <parameter=command>view</parameter>\n"
        "  <parameter=path>/x</parameter>\n</function>\n"
        "Observation:\n\nfile contents\n"
    )


def test_find_section_spans_thinking_uses_thought_to_action():
    txt = _two_turn_transcript()
    thinking, _ = RP._find_section_spans(txt)
    assert len(thinking) == 2
    # Turn 0 reasoning is bounded by the closed </think>.
    assert txt[thinking[0][0]:thinking[0][1]] == "\nreason one\n"
    # Turn 1 never closes </think>, so it runs up to the next Action:.
    s1, e1 = thinking[1]
    assert txt[s1:e1].startswith("\nreason two, no close tag")
    assert "Action:" not in txt[s1:e1]


def test_find_section_spans_patch_only_editing_calls():
    txt = _two_turn_transcript()
    _, patch = RP._find_section_spans(txt)
    # Only the str_replace call is a patch span; the view call is excluded.
    assert len(patch) == 1
    block = txt[patch[0][0]:patch[0][1]]
    assert block.startswith("<function=file_editor>")
    assert block.endswith("</function>")
    assert "str_replace" in block
    assert "view" not in block


def test_find_section_spans_includes_create_and_insert():
    txt = (
        "Step 0:\nThought:\n<think>x</think>\nAction:\n"
        "<function=file_editor>\n<parameter=command>create</parameter>\n"
        "<parameter=file_text>new file</parameter>\n</function>\nObservation:\nok\n"
        "Step 1:\nThought:\n<think>y</think>\nAction:\n"
        "<function=file_editor>\n<parameter=command>insert</parameter>\n</function>\nObservation:\nok\n"
        "Step 2:\nThought:\n<think>z</think>\nAction:\n"
        "<function=search>\n<parameter=search_term>q</parameter>\n</function>\nObservation:\nok\n"
    )
    _, patch = RP._find_section_spans(txt)
    assert len(patch) == 2  # create + insert; search is not a file edit


def test_find_section_spans_empty_when_no_turns():
    thinking, patch = RP._find_section_spans("plain prose, no markers at all")
    assert thinking == []
    assert patch == []


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
    # No span metadata => None => analysis falls back to legacy single markers.
    assert loaded.thinking_spans is None
    assert loaded.patch_spans is None


def test_save_load_roundtrip_spans(tmp_path: Path):
    result = _make_replay_result()
    result.thinking_spans = [(3, 9), (20, 25)]
    result.patch_spans = [(30, 48)]
    out = tmp_path / "spans.safetensors"
    RP.save_replay(result, out)
    loaded = RP.load_replay(out)
    assert loaded.thinking_spans == [(3, 9), (20, 25)]
    assert loaded.patch_spans == [(30, 48)]


def test_save_load_roundtrip_empty_spans_distinct_from_missing(tmp_path: Path):
    """A replay written with no spans found ([]) must load as [] (span logic,
    no spans) — distinct from a legacy file with no field at all (None)."""
    result = _make_replay_result()
    result.thinking_spans = []
    result.patch_spans = []
    out = tmp_path / "empty_spans.safetensors"
    RP.save_replay(result, out)
    loaded = RP.load_replay(out)
    assert loaded.thinking_spans == []
    assert loaded.patch_spans == []


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
