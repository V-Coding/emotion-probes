"""Plots: per-task timelines, outcome box plots, task-emotion heatmap, decile curves.

Visualization intentionally applies only ``_section_mask`` from analysis.py,
not the per-task aggregation mask (token-offset + special-token filtering).
The aggregation mask shapes the *statistics* (Mann-Whitney, OLS,
permutation); the plots show the underlying per-token series so the user
can eyeball spikes at template markers, the first 50 tokens, etc., and
reconcile the figures with the stats. Don't add ``_aggregation_mask`` here
without also revisiting that contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import _section_mask
from .replay import load_replay
from .trajectories import EASY_LABELS, HARD_LABELS

logger = logging.getLogger(__name__)

TARGET_EMOTIONS = ("frustrated", "angry", "disappointed", "overwhelmed", "confused")


def _difficulty_bucket(d: str) -> str:
    if d in EASY_LABELS:
        return "easy"
    if d in HARD_LABELS:
        return "hard"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-task timeline: one panel per emotion
# ---------------------------------------------------------------------------

def plot_timeline(replay_path: Path, out_path: Path) -> None:
    r = load_replay(replay_path)
    scores = r.scores.astype(np.float32)
    T, K = scores.shape

    # Find char positions of section markers on token axis
    think_tok = int(np.searchsorted(r.char_ends, r.thinking_end_char)) if r.thinking_end_char > 0 else -1
    patch_tok = int(np.searchsorted(r.char_starts, r.patch_start_char)) if r.patch_start_char > 0 else -1

    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), sharex=True)
    axes = np.atleast_2d(axes)

    x = np.arange(T)
    for k, emo in enumerate(r.emotions):
        ax = axes[k // ncol, k % ncol]
        ax.plot(x, scores[:, k], linewidth=0.6, color="steelblue")
        if think_tok > 0:
            ax.axvline(think_tok, color="darkorange", linestyle="--", linewidth=0.8, label="</think>")
        if patch_tok > 0:
            ax.axvline(patch_tok, color="firebrick", linestyle="--", linewidth=0.8, label="diff start")
        ax.set_title(emo, fontsize=9)
        ax.tick_params(labelsize=7)
    # Turn off any unused axes
    for k in range(K, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")

    title = f"{r.instance_id} — resolved={r.resolved}, difficulty={r.difficulty}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Box plot: 5 target emotions x {pass,fail} x {easy,hard}
# ---------------------------------------------------------------------------

def plot_box_by_outcome(per_task_df: pd.DataFrame, out_path: Path, section: str = "all") -> None:
    sub = per_task_df[
        (per_task_df["section"] == section)
        & (per_task_df["emotion"].isin(TARGET_EMOTIONS))
        & (per_task_df["difficulty_bucket"].isin(["easy", "hard"]))
    ].copy()
    sub["outcome"] = sub["resolved"].map({True: "pass", False: "fail"})
    sub["group"] = sub["outcome"] + "-" + sub["difficulty_bucket"]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=sub,
        x="emotion",
        y="mean",
        hue="group",
        order=list(TARGET_EMOTIONS),
        hue_order=["pass-easy", "pass-hard", "fail-easy", "fail-hard"],
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="emotion",
        y="mean",
        hue="group",
        order=list(TARGET_EMOTIONS),
        hue_order=["pass-easy", "pass-hard", "fail-easy", "fail-hard"],
        dodge=True,
        alpha=0.5,
        ax=ax,
        legend=False,
    )
    ax.set_title(f"Per-task mean emotion probe score by outcome (section={section})")
    ax.set_xlabel("")
    ax.set_ylabel("probe score (dot product, centered)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap: 24 tasks x 15 emotions
# ---------------------------------------------------------------------------

def plot_task_heatmap(per_task_df: pd.DataFrame, out_path: Path, section: str = "all") -> None:
    sub = per_task_df[per_task_df["section"] == section].copy()
    sub["outcome"] = sub["resolved"].map({True: "pass", False: "fail"})
    sub["row_key"] = sub["outcome"] + " | " + sub["difficulty_bucket"] + " | " + sub["instance_id"]

    mat = sub.pivot(index="row_key", columns="emotion", values="mean")
    # Row sort: pass before fail, then easy before hard
    order = sorted(
        mat.index,
        key=lambda k: (
            0 if k.startswith("pass ") else 1,
            0 if " | easy " in f" {k} " else 1,
            k,
        ),
    )
    mat = mat.reindex(order)

    # z-score each column for readability
    mat_z = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0, 1.0)

    fig, ax = plt.subplots(figsize=(1.0 * mat.shape[1] + 2, 0.35 * mat.shape[0] + 2))
    sns.heatmap(mat_z, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title(f"Per-task mean probe score (z-scored by emotion), section={section}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decile curves: mean decile score per emotion, pass vs fail
# ---------------------------------------------------------------------------

def plot_decile_curves(replay_dir: Path, out_path: Path, section: str = "all", n_boot: int = 500) -> None:
    files = sorted(Path(replay_dir).glob("*.safetensors"))
    profiles: list[tuple[bool, np.ndarray]] = []
    emotions_ref: list[str] | None = None
    for f in files:
        r = load_replay(f)
        mask = _section_mask(
            r.char_starts, r.char_ends, r.thinking_end_char, r.patch_start_char, section
        )
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        bins = np.array_split(idx, 10)
        prof = np.stack([r.scores.astype(np.float32)[b].mean(axis=0) for b in bins])
        profiles.append((r.resolved, prof))
        if emotions_ref is None:
            emotions_ref = r.emotions

    if not profiles or emotions_ref is None:
        logger.warning("No profiles to plot for decile_curves (section=%s)", section)
        return

    stack = np.stack([p[1] for p in profiles])  # (N, 10, K)
    labels = np.array([p[0] for p in profiles])

    def _mean_ci(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = arr.mean(axis=0)
        rng = np.random.default_rng(0)
        boots = np.stack(
            [arr[rng.integers(0, arr.shape[0], arr.shape[0])].mean(axis=0) for _ in range(n_boot)]
        )
        return mean, np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)

    K = stack.shape[2]
    ncol = 5
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.2 * nrow), sharex=True)
    axes = np.atleast_2d(axes)

    x = np.arange(10)
    for k, emo in enumerate(emotions_ref):
        ax = axes[k // ncol, k % ncol]
        if labels.any():
            m, lo, hi = _mean_ci(stack[labels, :, k])
            ax.plot(x, m, color="seagreen", label="pass")
            ax.fill_between(x, lo, hi, color="seagreen", alpha=0.2)
        if (~labels).any():
            m, lo, hi = _mean_ci(stack[~labels, :, k])
            ax.plot(x, m, color="indianred", label="fail")
            ax.fill_between(x, lo, hi, color="indianred", alpha=0.2)
        ax.set_title(emo, fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7)

    for k in range(K, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")

    fig.suptitle(f"Decile-level probe score by outcome (section={section})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
