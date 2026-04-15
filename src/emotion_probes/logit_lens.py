"""Logit lens: project emotion vectors through the unembedding matrix."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_logit_lens(
    emotion_vectors: np.ndarray,
    unembedding: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    emotion_names: list[str],
    top_k: int = 10,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Project each emotion vector through the unembedding matrix.

    For each emotion, find the top-k tokens that are most upweighted
    and most downweighted by the corresponding emotion vector.

    Args:
        emotion_vectors: (N, D) numpy array.
        unembedding: (vocab_size, D) torch tensor (lm_head weight).
        tokenizer: for decoding token IDs.
        emotion_names: list of N emotion names.
        top_k: number of top/bottom tokens per emotion.

    Returns:
        {
            "happy": {
                "top": [("joy", 2.34), ("smile", 1.98), ...],
                "bottom": [("grief", -1.56), ...]
            },
            ...
        }
    """
    import torch

    vectors_t = torch.from_numpy(emotion_vectors).float()
    unembed = unembedding.float().cpu()

    # (N, vocab_size) = (N, D) @ (D, vocab_size)
    logits = vectors_t @ unembed.T

    results = {}
    for i, emotion in enumerate(emotion_names):
        scores = logits[i]

        top_values, top_indices = scores.topk(top_k)
        bottom_values, bottom_indices = (-scores).topk(top_k)

        top_tokens = [
            (tokenizer.decode([idx.item()]).strip(), float(val))
            for idx, val in zip(top_indices, top_values)
        ]
        bottom_tokens = [
            (tokenizer.decode([idx.item()]).strip(), float(-val))
            for idx, val in zip(bottom_indices, bottom_values)
        ]

        results[emotion] = {"top": top_tokens, "bottom": bottom_tokens}

    return results
