"""Unit tests for trajectory parsing and stratified sampling.

These tests avoid network access — ``load_difficulty_map`` is monkey-patched
and HTTP fetching is not exercised. The replay-end-to-end and HF-dataset
paths are covered by integration checks in ``tests/test_replay.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from swebench_coding_emotions import trajectories as TR


SAMPLE_TRANSCRIPT = """<|im_start|>system
You are a helpful coding agent.<|im_end|>
<|im_start|>user
Fix the bug in foo.py.<|im_end|>
<|im_start|>assistant
<think>Let me read the file first.</think>
I'll start by reading foo.py.
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-bad
+good
<|im_end|>
"""


def test_split_prompt_stops_at_first_assistant_turn():
    prompt = TR._split_prompt(SAMPLE_TRANSCRIPT)
    assert "Fix the bug in foo.py" in prompt
    assert "<think>" not in prompt
    assert "diff --git" not in prompt


def test_split_prompt_handles_missing_marker():
    prompt = TR._split_prompt("no markers here")
    assert prompt == "no markers here"


def test_extract_patch_returns_last_diff():
    patch = TR._extract_patch(SAMPLE_TRANSCRIPT)
    assert patch.startswith("diff --git a/foo.py b/foo.py")
    assert "+good" in patch


def test_extract_patch_empty_when_absent():
    assert TR._extract_patch("just some text") == ""


def _make_trajs(n_pass_easy: int, n_pass_hard: int, n_fail_easy: int, n_fail_hard: int) -> list[TR.Trajectory]:
    trajs: list[TR.Trajectory] = []

    def mk(prefix: str, n: int, resolved: bool, diff: str) -> None:
        for i in range(n):
            trajs.append(
                TR.Trajectory(
                    instance_id=f"{prefix}-{i}",
                    prompt="prompt",
                    full_text="text",
                    patch="",
                    resolved=resolved,
                    difficulty=diff,
                )
            )

    mk("pe", n_pass_easy, True, "<15 min fix")
    mk("ph", n_pass_hard, True, "1-4 hours")
    mk("fe", n_fail_easy, False, "15 min - 1 hour")
    mk("fh", n_fail_hard, False, ">4 hours")
    # Throw in some "unknown"s to confirm they're dropped.
    mk("uk", 5, True, "unknown")
    return trajs


def test_stratified_sample_picks_exact_per_cell():
    trajs = _make_trajs(20, 20, 20, 20)
    sample = TR.stratified_sample(trajs, n=24, seed=0)
    assert len(sample) == 24
    cells = {
        (t.resolved, "easy" if t.difficulty in TR.EASY_LABELS else "hard") for t in sample
    }
    # All four cells represented
    assert cells == {(True, "easy"), (True, "hard"), (False, "easy"), (False, "hard")}

    # Exactly 6 per cell
    from collections import Counter
    c = Counter(
        (t.resolved, "easy" if t.difficulty in TR.EASY_LABELS else "hard") for t in sample
    )
    assert all(v == 6 for v in c.values())


def test_stratified_sample_is_deterministic_for_seed():
    trajs = _make_trajs(20, 20, 20, 20)
    a = TR.stratified_sample(trajs, n=24, seed=7)
    b = TR.stratified_sample(trajs, n=24, seed=7)
    assert [t.instance_id for t in a] == [t.instance_id for t in b]


def test_stratified_sample_takes_all_when_cell_short(caplog):
    trajs = _make_trajs(2, 20, 20, 20)   # pass-easy only has 2
    with caplog.at_level("WARNING"):
        sample = TR.stratified_sample(trajs, n=24, seed=1)
    pass_easy = [t for t in sample if t.resolved and t.difficulty in TR.EASY_LABELS]
    assert len(pass_easy) == 2
    assert any("pass" in r.msg.lower() or "cell" in r.msg.lower() for r in caplog.records)


def test_stratified_sample_requires_n_divisible_by_4():
    with pytest.raises(ValueError):
        TR.stratified_sample([], n=25)
