"""Visualization: heatmaps, scatter plots, and logit lens tables."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)


def plot_cosine_heatmap(
    sim_matrix: np.ndarray,
    emotion_names: list[str],
    output_path: Path,
    figsize: tuple[int, int] = (20, 18),
) -> None:
    """Plot a clustered cosine similarity heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g = sns.clustermap(
        sim_matrix,
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        figsize=figsize,
        linewidths=0,
        dendrogram_ratio=(0.1, 0.1),
    )
    g.ax_heatmap.tick_params(axis="both", labelsize=5)
    g.fig.suptitle("Emotion Vector Cosine Similarity", y=1.01, fontsize=14)
    g.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    logger.info("Saved cosine heatmap to %s", output_path)


def plot_pca_scatter(
    coords: np.ndarray,
    emotion_names: list[str],
    cluster_labels: np.ndarray | None,
    output_path: Path,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot PC1 vs PC2 scatter with emotion labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    if cluster_labels is not None:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=cluster_labels, cmap="tab10", s=30, alpha=0.8,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8)

    for i, name in enumerate(emotion_names):
        ax.annotate(
            name, (coords[i, 0], coords[i, 1]),
            fontsize=5, alpha=0.7,
            textcoords="offset points", xytext=(3, 3),
        )

    ax.set_xlabel("PC1 (Valence)")
    ax.set_ylabel("PC2 (Arousal)")
    ax.set_title("Emotion Vectors — PCA Projection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PCA scatter to %s", output_path)


def plot_umap_scatter(
    coords: np.ndarray,
    emotion_names: list[str],
    cluster_labels: np.ndarray | None,
    output_path: Path,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot UMAP 2D scatter with emotion labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    if cluster_labels is not None:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=cluster_labels, cmap="tab10", s=30, alpha=0.8,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8)

    for i, name in enumerate(emotion_names):
        ax.annotate(
            name, (coords[i, 0], coords[i, 1]),
            fontsize=5, alpha=0.7,
            textcoords="offset points", xytext=(3, 3),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Emotion Vectors — UMAP Projection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved UMAP scatter to %s", output_path)


def plot_logit_lens_table(
    logit_lens_results: dict[str, dict[str, list[tuple[str, float]]]],
    emotions_subset: list[str],
    output_path: Path,
    top_k: int = 5,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render a table of top/bottom tokens from logit lens analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for emotion in emotions_subset:
        if emotion not in logit_lens_results:
            continue
        data = logit_lens_results[emotion]
        top_str = ", ".join(f"{tok} ({sc:.1f})" for tok, sc in data["top"][:top_k])
        bottom_str = ", ".join(f"{tok} ({sc:.1f})" for tok, sc in data["bottom"][:top_k])
        rows.append([emotion, top_str, bottom_str])

    if not rows:
        logger.warning("No logit lens data to plot")
        return

    n_rows = len(rows)
    if figsize is None:
        figsize = (18, max(3, 0.4 * n_rows + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Emotion", "Top tokens (score)", "Bottom tokens (score)"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#d4e6f1")
        table[0, j].set_text_props(weight="bold")

    fig.suptitle("Logit Lens — Top/Bottom Tokens per Emotion Vector", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved logit lens table to %s", output_path)


def plot_rsa_heatmap(
    rsa_matrix: np.ndarray,
    layer_indices: list[int],
    output_path: Path,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot RSA correlation matrix across layers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(rsa_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    # Tick labels at regular intervals
    n = len(layer_indices)
    if n <= 20:
        tick_idx = list(range(n))
    else:
        tick_idx = list(range(0, n, max(1, n // 15)))
        if tick_idx[-1] != n - 1:
            tick_idx.append(n - 1)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(layer_indices[i]) for i in tick_idx], fontsize=7)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(layer_indices[i]) for i in tick_idx], fontsize=7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Representational Similarity Analysis — Cross-Layer Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved RSA heatmap to %s", output_path)


def plot_valence_arousal_correlation(
    pca_coords: np.ndarray,
    emotion_names: list[str],
    norms: dict[str, tuple[float, float]],
    correlation_result: dict[str, float],
    output_path: Path,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Plot PC1 vs valence and PC2 vs arousal scatter with regression lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    matched_idx = []
    valences, arousals = [], []
    for i, name in enumerate(emotion_names):
        if name in norms:
            matched_idx.append(i)
            v, a = norms[name]
            valences.append(v)
            arousals.append(a)

    pc1 = [pca_coords[i, 0] for i in matched_idx]
    pc2 = [pca_coords[i, 1] for i in matched_idx]
    names = [emotion_names[i] for i in matched_idx]

    # PC1 vs Valence
    ax1.scatter(valences, pc1, s=30, alpha=0.8)
    for j, name in enumerate(names):
        ax1.annotate(name, (valences[j], pc1[j]), fontsize=6, alpha=0.7,
                      textcoords="offset points", xytext=(3, 3))
    z = np.polyfit(valences, pc1, 1)
    xs = np.linspace(min(valences), max(valences), 50)
    ax1.plot(xs, np.polyval(z, xs), "r--", alpha=0.5)
    ax1.set_xlabel("Human Valence Rating (Warriner et al.)")
    ax1.set_ylabel("PC1 Score")
    ax1.set_title(f"PC1 vs Valence (r = {correlation_result['valence_r']:.3f})")

    # PC2 vs Arousal
    ax2.scatter(arousals, pc2, s=30, alpha=0.8)
    for j, name in enumerate(names):
        ax2.annotate(name, (arousals[j], pc2[j]), fontsize=6, alpha=0.7,
                      textcoords="offset points", xytext=(3, 3))
    z = np.polyfit(arousals, pc2, 1)
    xs = np.linspace(min(arousals), max(arousals), 50)
    ax2.plot(xs, np.polyval(z, xs), "r--", alpha=0.5)
    ax2.set_xlabel("Human Arousal Rating (Warriner et al.)")
    ax2.set_ylabel("PC2 Score")
    ax2.set_title(f"PC2 vs Arousal (r = {correlation_result['arousal_r']:.3f})")

    fig.suptitle("Emotion Vectors vs Human Affective Norms", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved valence/arousal correlation plot to %s", output_path)


def plot_variance_explained(
    explained_variance_ratio: np.ndarray,
    output_path: Path,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Scree plot of PCA variance explained."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    ax.bar(range(1, n + 1), explained_variance_ratio, alpha=0.7, label="Individual")
    ax.plot(range(1, n + 1), cumulative, "r-o", markersize=3, label="Cumulative")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Variance Explained — Emotion Vectors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved variance plot to %s", output_path)
