"""Geometric analysis of emotion vectors: cosine similarity, PCA, k-means, UMAP."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between emotion vectors.

    Args:
        vectors: (N, D) array of emotion vectors.

    Returns:
        (N, N) symmetric similarity matrix.
    """
    return cosine_similarity(vectors).astype(np.float32)


def compute_pca(
    vectors: np.ndarray,
    n_components: int = 10,
) -> tuple[np.ndarray, PCA]:
    """PCA on emotion vectors.

    Returns:
        (projected_coords of shape (N, n_components), fitted PCA object)
    """
    pca = PCA(n_components=min(n_components, vectors.shape[0], vectors.shape[1]))
    coords = pca.fit_transform(vectors)
    logger.info(
        "PCA variance explained: %s",
        ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(pca.explained_variance_ratio_[:5])),
    )
    return coords, pca


def compute_kmeans(vectors: np.ndarray, k: int = 10, seed: int = 42) -> np.ndarray:
    """K-means clustering of emotion vectors.

    Returns:
        Cluster labels array of shape (N,).
    """
    kmeans = KMeans(n_clusters=min(k, vectors.shape[0]), random_state=seed, n_init=10)
    labels = kmeans.fit_predict(vectors)
    return labels


def compute_umap(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """UMAP 2D projection of emotion vectors.

    Returns:
        2D coordinates array of shape (N, 2).
    """
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, vectors.shape[0] - 1),
        min_dist=min_dist,
        random_state=seed,
    )
    coords = reducer.fit_transform(vectors)
    return coords


def compute_rsa(
    vectors_by_layer: dict[int, np.ndarray],
) -> np.ndarray:
    """Representational Similarity Analysis across layers.

    Computes the cosine similarity matrix at each layer, then correlates
    those matrices (upper triangle) across all layer pairs.

    Args:
        vectors_by_layer: {layer_index: (N, D) emotion vectors}

    Returns:
        (L, L) matrix of Pearson correlations between layers' similarity structures.
    """
    layers = sorted(vectors_by_layer.keys())
    # Compute upper-triangle of cosine similarity for each layer
    sim_vecs = {}
    for layer in layers:
        sim = cosine_similarity(vectors_by_layer[layer])
        # Extract upper triangle (excluding diagonal)
        triu_idx = np.triu_indices_from(sim, k=1)
        sim_vecs[layer] = sim[triu_idx]

    n = len(layers)
    rsa_matrix = np.zeros((n, n), dtype=np.float32)
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            r = np.corrcoef(sim_vecs[li], sim_vecs[lj])[0, 1]
            rsa_matrix[i, j] = 0.0 if np.isnan(r) else float(r)

    logger.info("RSA computed across %d layers", n)
    return rsa_matrix


# Published human valence/arousal norms (Warriner et al., 2013, scale 1-9)
VALENCE_AROUSAL_NORMS: dict[str, tuple[float, float]] = {
    "afraid": (2.25, 5.12),
    "alarmed": (3.52, 6.50),
    "alert": (5.38, 5.14),
    "amazed": (7.55, 5.95),
    "amused": (7.05, 4.27),
    "angry": (2.53, 6.20),
    "annoyed": (2.80, 5.29),
    "anxious": (3.80, 6.20),
    "aroused": (5.95, 7.30),
    "ashamed": (2.52, 5.65),
    "astonished": (6.42, 5.73),
    "at ease": (6.15, 2.45),
    "awestruck": (6.05, 4.21),
    "bewildered": (4.32, 4.57),
    "bitter": (3.63, 4.64),
    "blissful": (7.67, 3.74),
    "bored": (2.95, 3.65),
    "brooding": (3.29, 4.00),
    "calm": (6.89, 1.67),
    "cheerful": (8.00, 5.76),
    "content": (7.16, 2.38),
    "desperate": (2.40, 6.27),
    "disgusted": (2.42, 5.42),
    "excited": (7.90, 6.26),
    "grateful": (7.64, 4.12),
    "guilty": (2.63, 5.28),
    "happy": (8.21, 6.49),
    "hopeful": (7.24, 4.07),
    "jealous": (2.85, 5.77),
    "lonely": (2.17, 3.90),
    "loving": (7.77, 5.13),
    "nostalgic": (5.73, 3.62),
    "proud": (7.32, 5.38),
    "sad": (1.61, 4.13),
    "surprised": (6.33, 6.45),
    "triumphant": (7.48, 5.64),
}


def correlate_with_human_norms(
    pca_coords: np.ndarray,
    emotion_names: list[str],
) -> dict[str, float]:
    """Correlate PC1/PC2 with published human valence/arousal ratings.

    Args:
        pca_coords: (N, >=2) PCA-projected emotion vectors.
        emotion_names: emotion labels aligned with rows of pca_coords.

    Returns:
        Dict with keys 'valence_r', 'arousal_r', and matched emotion count.
    """
    from scipy.stats import pearsonr

    valences, arousals, pc1s, pc2s = [], [], [], []
    matched = []
    for i, name in enumerate(emotion_names):
        if name in VALENCE_AROUSAL_NORMS:
            v, a = VALENCE_AROUSAL_NORMS[name]
            valences.append(v)
            arousals.append(a)
            pc1s.append(pca_coords[i, 0])
            pc2s.append(pca_coords[i, 1])
            matched.append(name)

    if len(matched) < 5:
        logger.warning("Too few emotions matched norms (%d), need at least 5", len(matched))
        return {"valence_r": 0.0, "arousal_r": 0.0, "n_matched": len(matched)}

    r_val, _ = pearsonr(pc1s, valences)
    r_aro, _ = pearsonr(pc2s, arousals)

    logger.info(
        "Human norm correlation (n=%d): PC1↔valence r=%.3f, PC2↔arousal r=%.3f",
        len(matched), r_val, r_aro,
    )
    return {
        "valence_r": float(r_val),
        "arousal_r": float(r_aro),
        "n_matched": len(matched),
        "matched_emotions": matched,
    }


def interpret_pca_components(
    pca: PCA,
    emotion_names: list[str],
    vectors: np.ndarray,
    top_k: int = 10,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """For each PC, list the top and bottom emotions by projection.

    Returns:
        {
            "PC1": {"top": [("happy", 0.32), ...], "bottom": [("sad", -0.28), ...]},
            "PC2": {...},
            ...
        }
    """
    projected = pca.transform(vectors)  # (N, n_components)
    result = {}
    for i in range(min(5, projected.shape[1])):
        scores = projected[:, i]
        sorted_indices = np.argsort(scores)
        top = [(emotion_names[idx], float(scores[idx])) for idx in sorted_indices[-top_k:][::-1]]
        bottom = [(emotion_names[idx], float(scores[idx])) for idx in sorted_indices[:top_k]]
        result[f"PC{i+1}"] = {"top": top, "bottom": bottom}
    return result
