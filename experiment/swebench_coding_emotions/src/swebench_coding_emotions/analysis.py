"""Aggregate per-task replay scores and run statistical tests.

Pipeline:

1. ``aggregate(replay_dir)`` → long-format pandas.DataFrame with one row per
   (instance_id, emotion, section). Sections are {"thinking", "agent", "patch",
   "all"} where ``thinking`` is the span ending at the last ``</think>``,
   ``patch`` is from the first ``diff --git`` onward, and ``agent`` is
   everything between.

2. ``mann_whitney_tests(df)`` → DataFrame of Mann-Whitney U tests per emotion
   (pass vs fail, easy vs hard), with rank-biserial effect sizes and
   Benjamini-Hochberg-adjusted p-values.

3. ``length_control_ols(df)`` → OLS ``emotion_mean ~ resolved + log1p(n_tokens)
   + C(difficulty)`` per emotion; reports the ``resolved`` coefficient.

4. ``decile_permutation(replay_dir)`` → cluster-based permutation test on
   time-course: bin each trajectory's tokens into 10 equal-count deciles,
   compare decile means pass vs fail with 1000 label shuffles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

from .replay import load_replay
from .trajectories import EASY_LABELS, HARD_LABELS

logger = logging.getLogger(__name__)


SECTIONS = ("thinking", "agent", "patch", "all")
DEFAULT_TOKEN_OFFSET = 50  # matches upstream extract-activations token_offset


# ---------------------------------------------------------------------------
# Section masks over the token axis
# ---------------------------------------------------------------------------

def _section_mask(
    char_starts: np.ndarray,
    char_ends: np.ndarray,
    think_end_char: int,
    patch_start_char: int,
    section: str,
) -> np.ndarray:
    """Boolean mask (T,) selecting tokens belonging to a given section."""
    if section == "all":
        return np.ones_like(char_starts, dtype=bool)
    if section == "thinking":
        if think_end_char < 0:
            return np.zeros_like(char_starts, dtype=bool)
        return char_ends <= think_end_char
    if section == "patch":
        if patch_start_char < 0:
            return np.zeros_like(char_starts, dtype=bool)
        return char_starts >= patch_start_char
    if section == "agent":
        lo = think_end_char if think_end_char >= 0 else 0
        hi = patch_start_char if patch_start_char >= 0 else char_ends[-1] + 1
        return (char_starts >= lo) & (char_ends <= hi)
    raise ValueError(f"Unknown section: {section}")


def _aggregation_mask(
    token_ids: np.ndarray,
    special_token_ids: list[int] | None,
    token_offset: int,
) -> np.ndarray:
    """Boolean mask (T,) keeping tokens eligible for per-task aggregation.

    Drops:
      - the first ``token_offset`` token positions (matches probe-build offset;
        the residual stream there is dominated by template/system-prompt
        boilerplate the probe never saw)
      - any position whose token id is a tokenizer special token
        (e.g. ``<|im_start|>``, ``<|im_end|>``, ``<think>``, ``</think>``,
        BOS/EOS/PAD). Content *between* these markers is kept; only the
        marker positions themselves are dropped.

    Special-token filtering only runs when ``special_token_ids`` is provided
    (non-empty). Older replay files without the metadata get only the offset.
    """
    T = token_ids.shape[0]
    mask = np.ones(T, dtype=bool)
    if token_offset > 0:
        mask[: min(token_offset, T)] = False
    if special_token_ids:
        special_arr = np.asarray(list(special_token_ids), dtype=token_ids.dtype)
        mask &= ~np.isin(token_ids, special_arr)
    return mask


def _difficulty_bucket(difficulty: str) -> str:
    if difficulty in EASY_LABELS:
        return "easy"
    if difficulty in HARD_LABELS:
        return "hard"
    return "unknown"


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(
    replay_dir: Path,
    use_cosine: bool = False,
    *,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
    drop_special_tokens: bool = True,
) -> pd.DataFrame:
    """Load every .safetensors replay and summarize per (task, emotion, section).

    The first ``token_offset`` token positions are excluded (matches the
    upstream probe-build offset) and tokens whose ids are tokenizer special
    tokens (``<|im_start|>``, ``</think>``, etc.) are dropped — content
    between markers is kept. Both filters are applied in addition to the
    section mask.

    Columns: instance_id, resolved, difficulty, difficulty_bucket,
    n_tokens, n_kept, section, emotion, mean, max, p90.
    """
    rows: list[dict] = []
    files = sorted(Path(replay_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No replay files in {replay_dir}")

    for f in files:
        r = load_replay(f)
        scores = r.cosine.astype(np.float32) if use_cosine else r.scores.astype(np.float32)
        T = scores.shape[0]
        special_ids = r.special_token_ids if drop_special_tokens else None
        agg_mask = _aggregation_mask(r.token_ids, special_ids, token_offset)
        for section in SECTIONS:
            section_mask = _section_mask(
                r.char_starts, r.char_ends, r.thinking_end_char, r.patch_start_char, section
            )
            mask = section_mask & agg_mask
            if not mask.any():
                continue
            sel = scores[mask]  # (T', K)
            n_kept = int(mask.sum())
            for k, emo in enumerate(r.emotions):
                col = sel[:, k]
                rows.append(
                    {
                        "instance_id": r.instance_id,
                        "resolved": r.resolved,
                        "difficulty": r.difficulty,
                        "difficulty_bucket": _difficulty_bucket(r.difficulty),
                        "n_tokens": T,
                        "n_kept": n_kept,
                        "section": section,
                        "emotion": emo,
                        "mean": float(col.mean()),
                        "max": float(col.max()),
                        "p90": float(np.percentile(col, 90)),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mann-Whitney U + rank-biserial + Benjamini-Hochberg
# ---------------------------------------------------------------------------

def _rank_biserial(u: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return 1.0 - (2.0 * u) / (n_a * n_b)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR. Returns adjusted q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the right
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def mann_whitney_tests(
    df: pd.DataFrame,
    section: str = "all",
) -> pd.DataFrame:
    """Per-emotion Mann-Whitney U on per-task ``mean`` within a section.

    Comparisons:
      - pass vs fail        (all tasks)
      - easy vs hard        (all tasks)
      - pass vs fail | hard (only hard tasks)
    """
    sub = df[df["section"] == section].copy()
    emotions = sorted(sub["emotion"].unique())
    comparisons = [
        ("pass_vs_fail", lambda d: (d[d["resolved"]]["mean"].values, d[~d["resolved"]]["mean"].values)),
        (
            "easy_vs_hard",
            lambda d: (
                d[d["difficulty_bucket"] == "easy"]["mean"].values,
                d[d["difficulty_bucket"] == "hard"]["mean"].values,
            ),
        ),
        (
            "pass_vs_fail_hard",
            lambda d: (
                d[(d["difficulty_bucket"] == "hard") & d["resolved"]]["mean"].values,
                d[(d["difficulty_bucket"] == "hard") & ~d["resolved"]]["mean"].values,
            ),
        ),
    ]

    rows: list[dict] = []
    for comp_name, split in comparisons:
        for emo in emotions:
            d = sub[sub["emotion"] == emo]
            a, b = split(d)
            if len(a) < 2 or len(b) < 2:
                u, p, rb = float("nan"), float("nan"), float("nan")
            else:
                res = stats.mannwhitneyu(a, b, alternative="two-sided")
                u, p = float(res.statistic), float(res.pvalue)
                rb = _rank_biserial(u, len(a), len(b))
            rows.append(
                {
                    "comparison": comp_name,
                    "section": section,
                    "emotion": emo,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": float(np.mean(a)) if len(a) else float("nan"),
                    "mean_b": float(np.mean(b)) if len(b) else float("nan"),
                    "u": u,
                    "p": p,
                    "rank_biserial": rb,
                }
            )

    out = pd.DataFrame(rows)
    # BH-adjust within each (comparison, section) family.
    out["q_bh"] = float("nan")
    for comp_name in out["comparison"].unique():
        mask = (out["comparison"] == comp_name) & out["p"].notna()
        if mask.any():
            out.loc[mask, "q_bh"] = _bh_adjust(out.loc[mask, "p"].values)
    return out


# ---------------------------------------------------------------------------
# OLS length control
# ---------------------------------------------------------------------------

def length_control_ols(df: pd.DataFrame, section: str = "all") -> pd.DataFrame:
    """Per emotion, fit ``mean ~ resolved + log1p(n_tokens) + C(difficulty)``."""
    sub = df[df["section"] == section].copy()
    sub["log_n_tokens"] = np.log1p(sub["n_tokens"])
    sub["resolved_int"] = sub["resolved"].astype(int)

    rows: list[dict] = []
    for emo in sorted(sub["emotion"].unique()):
        d = sub[sub["emotion"] == emo]
        if d["resolved"].nunique() < 2 or len(d) < 4:
            continue
        try:
            model = ols(
                "mean ~ resolved_int + log_n_tokens + C(difficulty_bucket)", data=d
            ).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("OLS failed for %s: %s", emo, e)
            continue
        coef = model.params.get("resolved_int", float("nan"))
        pval = model.pvalues.get("resolved_int", float("nan"))
        ci_low, ci_high = model.conf_int().loc["resolved_int"] if "resolved_int" in model.params else (float("nan"), float("nan"))
        rows.append(
            {
                "emotion": emo,
                "section": section,
                "resolved_coef": float(coef),
                "resolved_p": float(pval),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "r2": float(model.rsquared),
                "n": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decile time-course permutation test
# ---------------------------------------------------------------------------

def _decile_means(scores: np.ndarray, mask: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Return (n_bins, K) array of mean score per decile within ``mask``."""
    idx = np.where(mask)[0]
    if len(idx) < n_bins:
        return np.full((n_bins, scores.shape[1]), np.nan)
    bins = np.array_split(idx, n_bins)
    return np.stack([scores[b].mean(axis=0) for b in bins])


def decile_permutation(
    replay_dir: Path,
    section: str = "all",
    n_perm: int = 1000,
    seed: int = 42,
    *,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
    drop_special_tokens: bool = True,
) -> pd.DataFrame:
    """Cluster-based permutation test on pass-vs-fail decile time courses.

    For each emotion, compute the per-trajectory (10, K) decile profile, then
    test whether the decile-wise mean difference (pass - fail) exceeds what
    label-shuffling produces. Reports the summed absolute decile difference
    ("cluster mass") and its permutation p-value.
    """
    rng = np.random.default_rng(seed)
    files = sorted(Path(replay_dir).glob("*.safetensors"))
    per_task_profiles: list[tuple[str, bool, np.ndarray]] = []
    emotions_ref: list[str] | None = None

    for f in files:
        r = load_replay(f)
        section_mask = _section_mask(
            r.char_starts, r.char_ends, r.thinking_end_char, r.patch_start_char, section
        )
        special_ids = r.special_token_ids if drop_special_tokens else None
        agg_mask = _aggregation_mask(r.token_ids, special_ids, token_offset)
        mask = section_mask & agg_mask
        if not mask.any():
            continue
        prof = _decile_means(r.scores.astype(np.float32), mask)
        if np.isnan(prof).any():
            continue
        per_task_profiles.append((r.instance_id, r.resolved, prof))
        if emotions_ref is None:
            emotions_ref = r.emotions

    if not per_task_profiles or emotions_ref is None:
        return pd.DataFrame(columns=["emotion", "section", "cluster_mass", "p_perm", "n_pass", "n_fail"])

    labels = np.array([p[1] for p in per_task_profiles])
    stack = np.stack([p[2] for p in per_task_profiles])  # (N, 10, K)

    n_pass = int(labels.sum())
    n_fail = int((~labels).sum())
    if n_pass < 2 or n_fail < 2:
        logger.warning("Too few tasks for permutation test (pass=%d fail=%d)", n_pass, n_fail)
        return pd.DataFrame(columns=["emotion", "section", "cluster_mass", "p_perm", "n_pass", "n_fail"])

    def stat(labs: np.ndarray) -> np.ndarray:
        # (10, K)
        pass_mean = stack[labs].mean(axis=0)
        fail_mean = stack[~labs].mean(axis=0)
        return np.abs(pass_mean - fail_mean).sum(axis=0)  # (K,)

    observed = stat(labels)
    ge_count = np.zeros_like(observed, dtype=np.int64)
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        null = stat(perm)
        ge_count += null >= observed

    p = (ge_count + 1) / (n_perm + 1)

    rows = [
        {
            "emotion": emo,
            "section": section,
            "cluster_mass": float(observed[k]),
            "p_perm": float(p[k]),
            "n_pass": n_pass,
            "n_fail": n_fail,
        }
        for k, emo in enumerate(emotions_ref)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_full_analysis(
    replay_dir: Path,
    out_dir: Path,
    *,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
    drop_special_tokens: bool = True,
) -> None:
    """Write per_task.csv, stats.csv, summary.json to ``out_dir``.

    Token-axis filtering applied uniformly across `aggregate` and
    `decile_permutation`: drop the first ``token_offset`` positions, drop
    tokenizer special-token positions (preserving content between markers).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate(
        replay_dir, use_cosine=False,
        token_offset=token_offset, drop_special_tokens=drop_special_tokens,
    )
    df.to_csv(out_dir / "per_task.csv", index=False)

    df_cos = aggregate(
        replay_dir, use_cosine=True,
        token_offset=token_offset, drop_special_tokens=drop_special_tokens,
    )
    df_cos.to_csv(out_dir / "per_task_cosine.csv", index=False)

    stats_rows: list[pd.DataFrame] = []
    for section in SECTIONS:
        mw = mann_whitney_tests(df, section=section)
        mw["metric"] = "mannwhitney"
        stats_rows.append(mw)
        ols_df = length_control_ols(df, section=section)
        ols_df["metric"] = "ols_length_control"
        stats_rows.append(ols_df)
        dec = decile_permutation(
            replay_dir, section=section,
            token_offset=token_offset, drop_special_tokens=drop_special_tokens,
        )
        dec["metric"] = "decile_permutation"
        stats_rows.append(dec)

    all_stats = pd.concat(stats_rows, ignore_index=True, sort=False)
    all_stats.to_csv(out_dir / "stats.csv", index=False)

    n_tasks = df["instance_id"].nunique()
    summary = {
        "n_tasks": int(n_tasks),
        "pass_count": int(df.drop_duplicates("instance_id")["resolved"].sum()),
        "sections": list(SECTIONS),
        "emotions": sorted(df["emotion"].unique().tolist()),
        "token_offset": int(token_offset),
        "drop_special_tokens": bool(drop_special_tokens),
        "significant_mw_pass_vs_fail": sorted(
            all_stats[
                (all_stats["metric"] == "mannwhitney")
                & (all_stats["comparison"] == "pass_vs_fail")
                & (all_stats["q_bh"] < 0.05)
            ][["emotion", "section", "p", "q_bh", "rank_biserial"]]
            .to_dict("records"),
            key=lambda r: r["q_bh"],
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
