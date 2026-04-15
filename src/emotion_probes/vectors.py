"""Emotion vector computation: mean subtraction and PCA denoising."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from emotion_probes.data import load_tensors, save_metadata, save_tensors

logger = logging.getLogger(__name__)


def compute_emotion_vectors(
    activations_dir: Path,
    output_dir: Path,
    layers: list[int],
    pca_variance_threshold: float = 0.5,
) -> dict[int, np.ndarray]:
    """Compute denoised emotion vectors from raw activations.

    Pipeline per layer:
      1. Load per-emotion mean activations (N_emotions, hidden_dim)
      2. Subtract global mean across emotions
      3. Denoise: project out top PCA components of neutral data
         that explain >= pca_variance_threshold of variance

    Returns:
        Dict mapping layer index to emotion vectors array (N_emotions, hidden_dim).
    """
    results = {}

    for layer in layers:
        logger.info("Computing emotion vectors for layer %d", layer)

        # Step 1: Load emotion means and global mean
        means_path = activations_dir / f"emotion_means_layer_{layer}.safetensors"
        global_path = activations_dir / f"global_mean_layer_{layer}.safetensors"

        means_data = load_tensors(means_path)
        emotion_means = means_data["emotion_means"]  # (N, hidden_dim)
        global_mean = load_tensors(global_path)["global_mean"]  # (hidden_dim,)

        # Load emotion names from metadata
        from safetensors import safe_open
        with safe_open(str(means_path), framework="numpy") as f:
            meta = f.metadata()
        emotion_names = json.loads(meta["emotions"])

        # Step 2: Subtract global mean
        raw_vectors = emotion_means - global_mean[np.newaxis, :]

        # Step 3: Denoise with neutral data PCA
        neutral_path = activations_dir / f"neutral_layer_{layer}.safetensors"
        if neutral_path.exists():
            neutral_data = load_tensors(neutral_path)["activations"]  # (M, hidden_dim)
            vectors, denoise_info = _denoise_with_neutral_pca(
                raw_vectors, neutral_data, pca_variance_threshold,
            )
            logger.info(
                "Layer %d: projected out %d PCA components (%.1f%% variance)",
                layer, denoise_info["n_components"], denoise_info["variance_explained"] * 100,
            )
        else:
            logger.warning(
                "No neutral activations found for layer %d; skipping denoising", layer,
            )
            vectors = raw_vectors
            denoise_info = {"n_components": 0, "variance_explained": 0.0}

        # Save
        vectors_dir = output_dir / "vectors"
        save_tensors(
            {"emotion_vectors": vectors},
            vectors_dir / f"emotion_vectors_layer_{layer}.safetensors",
            metadata={
                "layer": str(layer),
                "emotions": json.dumps(emotion_names),
                "n_emotions": str(len(emotion_names)),
                "denoised": "true",
                "pca_components_removed": str(denoise_info["n_components"]),
                "pca_variance_explained": f"{denoise_info['variance_explained']:.4f}",
            },
        )
        save_tensors(
            {"raw_vectors": raw_vectors},
            vectors_dir / f"raw_vectors_layer_{layer}.safetensors",
            metadata={
                "layer": str(layer),
                "emotions": json.dumps(emotion_names),
            },
        )
        save_metadata(
            {
                "layer": layer,
                "emotions": emotion_names,
                "denoise_info": denoise_info,
            },
            vectors_dir / f"vectors_metadata_layer_{layer}.json",
        )

        results[layer] = vectors
        logger.info("Saved emotion vectors for layer %d: shape %s", layer, vectors.shape)

    return results


def _denoise_with_neutral_pca(
    vectors: np.ndarray,
    neutral_activations: np.ndarray,
    variance_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Project out top PCA components of neutral data from emotion vectors.

    Args:
        vectors: (N, D) raw emotion vectors.
        neutral_activations: (M, D) activations from neutral dialogues.
        variance_threshold: project out enough components to explain this
            fraction of variance in neutral data.

    Returns:
        (denoised_vectors, info_dict)
    """
    # Center neutral data
    neutral_centered = neutral_activations - neutral_activations.mean(axis=0)

    # Fit PCA to find how many components explain >= threshold variance
    max_components = min(neutral_centered.shape[0], neutral_centered.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(neutral_centered)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # Find the smallest number of components whose cumulative variance >= threshold.
    # argmax on a bool array returns the first True index (first index where cumvar >= threshold).
    above_threshold = cumulative_variance >= variance_threshold
    if not above_threshold.any():
        n_components = max_components
    else:
        n_components = int(np.argmax(above_threshold)) + 1
    n_components = max(1, min(n_components, max_components))

    # Get the top-k principal components (already unit-norm from sklearn PCA)
    components = pca.components_[:n_components]  # (k, D)

    # Project out these components: v' = v - v @ C^T @ C
    # where C is (k, D), C^T @ C is the projection matrix onto the subspace
    denoised = vectors - vectors @ components.T @ components  # (N, D)

    actual_variance = float(cumulative_variance[n_components - 1]) if n_components > 0 else 0.0

    return denoised, {
        "n_components": n_components,
        "variance_explained": actual_variance,
        "total_neutral_samples": neutral_activations.shape[0],
    }
