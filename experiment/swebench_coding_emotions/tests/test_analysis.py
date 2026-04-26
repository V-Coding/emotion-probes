"""Tests for the token-axis aggregation mask and per-task aggregation.

These do not require torch / a real model: we synthesize a few replay
files on disk and exercise ``aggregate`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emotion_probes.data import save_tensors
from swebench_coding_emotions import analysis as A
from swebench_coding_emotions import replay as RP


# ---------------------------------------------------------------------------
# _aggregation_mask
# ---------------------------------------------------------------------------

def test_aggregation_mask_drops_first_n_positions():
    token_ids = np.arange(10, dtype=np.int32)
    mask = A._aggregation_mask(token_ids, special_token_ids=None, token_offset=3)
    assert mask.tolist() == [False, False, False, True, True, True, True, True, True, True]


def test_aggregation_mask_drops_special_tokens_keeps_content():
    # tokens 100, 101 are special markers (e.g. <|im_start|>, <|im_end|>);
    # 5, 6, 7 are content. Specials at positions 0, 4, 7 should be dropped;
    # everything else (after offset=0) kept.
    token_ids = np.array([100, 5, 6, 7, 101, 8, 9, 100, 11], dtype=np.int32)
    mask = A._aggregation_mask(token_ids, special_token_ids=[100, 101], token_offset=0)
    assert mask.tolist() == [False, True, True, True, False, True, True, False, True]


def test_aggregation_mask_combines_offset_and_special():
    token_ids = np.array([100, 100, 5, 100, 6, 7, 101], dtype=np.int32)
    mask = A._aggregation_mask(token_ids, special_token_ids=[100, 101], token_offset=2)
    # First 2 positions dropped by offset; positions 3 and 6 dropped as special.
    assert mask.tolist() == [False, False, True, False, True, True, False]


def test_aggregation_mask_offset_zero_no_specials_keeps_everything():
    token_ids = np.arange(5, dtype=np.int32)
    mask = A._aggregation_mask(token_ids, special_token_ids=None, token_offset=0)
    assert mask.all()


def test_aggregation_mask_handles_offset_larger_than_T():
    token_ids = np.arange(3, dtype=np.int32)
    mask = A._aggregation_mask(token_ids, special_token_ids=None, token_offset=10)
    assert not mask.any()


# ---------------------------------------------------------------------------
# aggregate(): end-to-end on synthetic replay files
# ---------------------------------------------------------------------------

def _write_synthetic_replay(
    path: Path,
    *,
    instance_id: str,
    resolved: bool,
    difficulty: str,
    token_ids: np.ndarray,
    scores: np.ndarray,
    emotions: list[str],
    special_token_ids: list[int],
    thinking_end_char: int = -1,
    patch_start_char: int = -1,
) -> None:
    T = token_ids.shape[0]
    char_starts = np.arange(T, dtype=np.int32)
    char_ends = np.arange(1, T + 1, dtype=np.int32)
    save_tensors(
        {
            "scores": scores.astype(np.float16),
            "cosine": scores.astype(np.float16),
            "token_ids": token_ids.astype(np.int32),
            "char_starts": char_starts,
            "char_ends": char_ends,
        },
        path,
        metadata={
            "instance_id": instance_id,
            "layer": "42",
            "emotions": json.dumps(emotions),
            "resolved": "1" if resolved else "0",
            "difficulty": difficulty,
            "thinking_end_char": str(thinking_end_char),
            "patch_start_char": str(patch_start_char),
            "model_fingerprint": "test",
            "special_token_ids": json.dumps(special_token_ids),
            "probe_variant": "augmented",
        },
    )


def test_aggregate_drops_offset_and_special_tokens(tmp_path: Path):
    # 60 tokens; first 50 should be skipped by offset, then token id 999 is
    # special and appears at positions 52 and 55. Score at every token is 1.0
    # except at specials and the first 50 (set to 100.0). Per-task mean over
    # the kept window should therefore equal exactly 1.0.
    T, K = 60, 2
    token_ids = np.full(T, 1, dtype=np.int32)
    token_ids[52] = 999
    token_ids[55] = 999
    scores = np.full((T, K), 1.0, dtype=np.float32)
    scores[:50] = 100.0
    scores[52] = 100.0
    scores[55] = 100.0

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_synthetic_replay(
        replay_dir / "task_a.safetensors",
        instance_id="task_a",
        resolved=True,
        difficulty="<15 min fix",
        token_ids=token_ids,
        scores=scores,
        emotions=["frustrated", "calm"],
        special_token_ids=[999],
    )

    df = A.aggregate(replay_dir, token_offset=50, drop_special_tokens=True)
    means = df[df["section"] == "all"]["mean"].unique()
    assert means.shape == (1,)
    assert means[0] == 1.0
    n_kept = df[df["section"] == "all"]["n_kept"].unique()
    assert n_kept.tolist() == [60 - 50 - 2]


def test_aggregate_keep_special_tokens_flag(tmp_path: Path):
    # Same as above but with drop_special_tokens=False: the two special-token
    # positions still count, raising the mean above 1.0.
    T, K = 60, 1
    token_ids = np.full(T, 1, dtype=np.int32)
    token_ids[55] = 999
    scores = np.full((T, K), 1.0, dtype=np.float32)
    scores[:50] = 100.0
    scores[55] = 11.0  # special-token position kept => contributes 11

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_synthetic_replay(
        replay_dir / "task_b.safetensors",
        instance_id="task_b",
        resolved=False,
        difficulty="1-4 hours",
        token_ids=token_ids,
        scores=scores,
        emotions=["frustrated"],
        special_token_ids=[999],
    )

    df_drop = A.aggregate(replay_dir, token_offset=50, drop_special_tokens=True)
    df_keep = A.aggregate(replay_dir, token_offset=50, drop_special_tokens=False)
    mean_drop = df_drop[df_drop["section"] == "all"]["mean"].iloc[0]
    mean_keep = df_keep[df_keep["section"] == "all"]["mean"].iloc[0]
    assert mean_drop == 1.0
    assert mean_keep > mean_drop


def test_aggregate_no_auc_column(tmp_path: Path):
    """The legacy ``auc`` (= mean × T) column should not be produced anymore;
    it was a length-statistic in disguise."""
    T = 60
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_synthetic_replay(
        replay_dir / "task_c.safetensors",
        instance_id="task_c",
        resolved=True,
        difficulty="<15 min fix",
        token_ids=np.full(T, 1, dtype=np.int32),
        scores=np.full((T, 1), 0.5, dtype=np.float32),
        emotions=["frustrated"],
        special_token_ids=[],
    )
    df = A.aggregate(replay_dir, token_offset=50, drop_special_tokens=True)
    assert "auc" not in df.columns
    assert {"mean", "max", "p90", "n_kept", "n_tokens"}.issubset(df.columns)


def test_run_full_analysis_records_empty_cells(tmp_path: Path, caplog):
    """If a section has no kept tokens for some instance, summary.json must
    list it under empty_section_cells and a WARNING must be logged. Without
    this, downstream MW / OLS would silently compute on unequal sample sizes."""
    import logging

    T = 60
    # Trajectory with no </think> => "thinking" section is empty.
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_synthetic_replay(
        replay_dir / "task_e.safetensors",
        instance_id="task_e",
        resolved=True,
        difficulty="<15 min fix",
        token_ids=np.full(T, 1, dtype=np.int32),
        scores=np.full((T, 1), 1.0, dtype=np.float32),
        emotions=["frustrated"],
        special_token_ids=[],
        thinking_end_char=-1,
        patch_start_char=-1,
    )
    out_dir = tmp_path / "analysis"
    with caplog.at_level(logging.WARNING, logger="swebench_coding_emotions.analysis"):
        from swebench_coding_emotions import analysis as _A
        _A.run_full_analysis(replay_dir, out_dir, token_offset=0)

    summary = json.loads((out_dir / "summary.json").read_text())
    cells = {(c["instance_id"], c["section"]) for c in summary["empty_section_cells"]}
    assert ("task_e", "thinking") in cells
    assert ("task_e", "patch") in cells
    assert any("Aggregation produced no rows" in r.message for r in caplog.records)


def test_aggregate_section_returns_empty_when_all_kept_tokens_filtered(tmp_path: Path):
    """If every token in a section is dropped (offset + specials), that
    (instance, section, emotion) row should simply be omitted rather than
    yielding NaN summaries."""
    T = 60
    token_ids = np.full(T, 999, dtype=np.int32)  # every token is special
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_synthetic_replay(
        replay_dir / "task_d.safetensors",
        instance_id="task_d",
        resolved=True,
        difficulty="<15 min fix",
        token_ids=token_ids,
        scores=np.zeros((T, 1), dtype=np.float32),
        emotions=["frustrated"],
        special_token_ids=[999],
    )
    df = A.aggregate(replay_dir, token_offset=50, drop_special_tokens=True)
    assert df.empty
