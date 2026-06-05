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
# _section_mask: span-based (new) and legacy single-marker fallback
# ---------------------------------------------------------------------------

def test_tokens_in_spans_membership_by_start():
    char_starts = np.array([0, 5, 10, 15, 20], dtype=np.int32)
    mask = A._tokens_in_spans(char_starts, [(5, 16)])
    assert mask.tolist() == [False, True, True, True, False]


def test_section_mask_spans_partition_all():
    char_starts = np.arange(0, 40, 4, dtype=np.int32)  # 0,4,...,36
    char_ends = char_starts + 4
    thinking = [(0, 8)]    # token starts 0,4  -> idx 0,1
    patch = [(20, 32)]     # token starts 20,24,28 -> idx 5,6,7
    kw = dict(thinking_spans=thinking, patch_spans=patch)
    tmask = A._section_mask(char_starts, char_ends, -1, -1, "thinking", **kw)
    pmask = A._section_mask(char_starts, char_ends, -1, -1, "patch", **kw)
    amask = A._section_mask(char_starts, char_ends, -1, -1, "agent", **kw)
    allm = A._section_mask(char_starts, char_ends, -1, -1, "all", **kw)

    assert tmask.tolist() == [True, True, False, False, False, False, False, False, False, False]
    assert pmask.tolist() == [False, False, False, False, False, True, True, True, False, False]
    # thinking / patch / agent form a partition of "all", no overlap.
    assert (tmask | pmask | amask).tolist() == allm.tolist()
    assert not (tmask & pmask).any()
    assert not (tmask & amask).any()
    assert not (pmask & amask).any()


def test_section_mask_legacy_fallback_when_spans_none():
    char_starts = np.arange(0, 40, 4, dtype=np.int32)
    char_ends = char_starts + 4
    # spans None => legacy: thinking = char_ends <= think_end_char.
    tmask = A._section_mask(char_starts, char_ends, 8, -1, "thinking")
    assert tmask.tolist() == [True, True, False, False, False, False, False, False, False, False]
    # patch_start_char = -1 => empty patch under legacy path.
    pmask = A._section_mask(char_starts, char_ends, 8, -1, "patch")
    assert not pmask.any()


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
    thinking_spans: list[tuple[int, int]] | None = None,
    patch_spans: list[tuple[int, int]] | None = None,
) -> None:
    T = token_ids.shape[0]
    char_starts = np.arange(T, dtype=np.int32)
    char_ends = np.arange(1, T + 1, dtype=np.int32)
    metadata = {
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
    }
    # Only emit span keys when provided, so tests that omit them exercise the
    # legacy (None => single-marker) fallback path, matching real old files.
    if thinking_spans is not None:
        metadata["thinking_spans"] = json.dumps([list(s) for s in thinking_spans])
    if patch_spans is not None:
        metadata["patch_spans"] = json.dumps([list(s) for s in patch_spans])
    save_tensors(
        {
            "scores": scores.astype(np.float16),
            "cosine": scores.astype(np.float16),
            "token_ids": token_ids.astype(np.int32),
            "char_starts": char_starts,
            "char_ends": char_ends,
        },
        path,
        metadata=metadata,
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


def test_aggregate_patch_and_thinking_via_spans(tmp_path: Path):
    """Span-bearing replay files drive the patch/thinking/agent partition;
    this is the case the empty-patch bug fix produces on future replays."""
    T = 20
    token_ids = np.zeros(T, dtype=np.int32)
    scores = np.ones((T, 1), dtype=np.float32)
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    # char_starts = arange(T): token i covers [i, i+1).
    _write_synthetic_replay(
        replay_dir / "t.safetensors",
        instance_id="t",
        resolved=True,
        difficulty="<15 min fix",
        token_ids=token_ids,
        scores=scores,
        emotions=["frustrated"],
        special_token_ids=[],
        thinking_spans=[(0, 3)],   # tokens 0,1,2
        patch_spans=[(5, 9)],      # tokens 5,6,7,8
    )
    df = A.aggregate(replay_dir, token_offset=0, drop_special_tokens=False)
    kept = {row["section"]: row["n_kept"] for _, row in df.iterrows()}
    assert kept["thinking"] == 3
    assert kept["patch"] == 4
    assert kept["agent"] == T - 3 - 4   # complement of thinking|patch
    assert kept["all"] == T


# ---------------------------------------------------------------------------
# Valence composite
# ---------------------------------------------------------------------------

def _valence_long_df():
    """Synthetic per-task long frame: pass tasks calm, fail tasks distressed."""
    import pandas as pd

    rows = []
    spec = {
        "p1": (True, "easy", {"frustrated": -2.0, "hopeful": 2.0}),
        "p2": (True, "hard", {"frustrated": -1.0, "hopeful": 1.0}),
        "p3": (True, "hard", {"frustrated": -1.5, "hopeful": 1.5}),
        "f1": (False, "easy", {"frustrated": 2.0, "hopeful": -2.0}),
        "f2": (False, "hard", {"frustrated": 1.0, "hopeful": -1.0}),
        "f3": (False, "hard", {"frustrated": 1.5, "hopeful": -1.5}),
    }
    for iid, (res, bucket, emo) in spec.items():
        for e, v in emo.items():
            rows.append(
                {
                    "instance_id": iid, "resolved": res, "difficulty_bucket": bucket,
                    "section": "all", "emotion": e, "mean": v,
                }
            )
    return pd.DataFrame(rows)


def test_compute_valence_orients_distress_positive():
    df = _valence_long_df()
    val = A.compute_valence(df, "all").set_index("instance_id")
    # frustrated is +valence, hopeful is -valence; fail tasks should score high.
    assert val.loc["f1", "valence"] > 0
    assert val.loc["p1", "valence"] < 0
    fail_mean = val.loc[["f1", "f2", "f3"], "valence"].mean()
    pass_mean = val.loc[["p1", "p2", "p3"], "valence"].mean()
    assert fail_mean > pass_mean


def test_compute_valence_excludes_unvalenced_emotions():
    import pandas as pd
    df = pd.DataFrame(
        [
            {"instance_id": "a", "resolved": True, "difficulty_bucket": "easy",
             "section": "all", "emotion": "curious", "mean": 5.0},
            {"instance_id": "b", "resolved": False, "difficulty_bucket": "hard",
             "section": "all", "emotion": "curious", "mean": -5.0},
        ]
    )
    # Only curious present, which is unvalenced => no usable composite.
    val = A.compute_valence(df, "all")
    assert val.empty


def test_valence_tests_pass_vs_fail_direction_and_shape():
    df = _valence_long_df()
    vt = A.valence_tests(df, "all", n_perm=2000, seed=0)
    assert set(vt["emotion"]) == {"__valence__"}
    pf = vt[vt["comparison"] == "pass_vs_fail"].iloc[0]
    assert pf["n_a"] == 3 and pf["n_b"] == 3
    assert pf["rank_biserial"] > 0          # fail tasks more distressed
    assert pf["p"] <= 0.1                    # perfect 3-vs-3 separation
    assert 0.0 < pf["p_perm"] <= 1.0


def _valence_df_with_lengths():
    """8 tasks; outcome drives distress, section length (n_kept) interleaved so
    it is not confounded with outcome."""
    import pandas as pd

    specs = [
        ("p1", True, "easy", 1000, -1.0), ("p2", True, "hard", 3000, -1.4),
        ("p3", True, "easy", 5000, -1.1), ("p4", True, "hard", 7000, -1.5),
        ("f1", False, "easy", 2000, 1.0), ("f2", False, "hard", 4000, 1.4),
        ("f3", False, "easy", 6000, 1.1), ("f4", False, "hard", 8000, 1.5),
    ]
    rows = []
    for iid, res, bucket, nkept, d in specs:
        for e, sign in [("frustrated", 1.0), ("hopeful", -1.0)]:
            rows.append(
                {
                    "instance_id": iid, "resolved": res, "difficulty_bucket": bucket,
                    "section": "all", "emotion": e, "mean": d * sign,
                    "n_tokens": nkept, "n_kept": nkept,
                }
            )
    return pd.DataFrame(rows)


def test_valence_length_control_detects_outcome_effect():
    df = _valence_df_with_lengths()
    out = A.valence_length_control_ols(df, "all")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["emotion"] == "__valence__"
    assert row["section"] == "all"
    assert int(row["n"]) == 8
    # Passing reduces distress even with the section-length term in the model.
    assert row["resolved_coef"] < 0
    assert row["resolved_p"] < 0.05
    assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])


def test_valence_length_control_guards_insufficient_data():
    import pandas as pd
    # Only pass tasks => resolved has a single level => no fit, empty frame.
    df = pd.DataFrame(
        [
            {"instance_id": iid, "resolved": True, "difficulty_bucket": "easy",
             "section": "all", "emotion": e, "mean": v, "n_tokens": 100, "n_kept": 100}
            for iid in ("a", "b", "c", "d")
            for e, v in [("frustrated", 1.0), ("hopeful", -1.0)]
        ]
    )
    out = A.valence_length_control_ols(df, "all")
    assert out.empty
