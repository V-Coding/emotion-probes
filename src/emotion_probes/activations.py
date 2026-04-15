"""Streaming activation extraction with running mean accumulators."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from emotion_probes.data import load_neutral_dialogues, load_stories, save_metadata, save_tensors, topic_hash

if TYPE_CHECKING:
    from emotion_probes.config import ActivationConfig
    from emotion_probes.model import EmotionProbeModel

logger = logging.getLogger(__name__)


class RunningMean:
    """Numerically stable running mean accumulator using float64."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._sum = np.zeros(shape, dtype=np.float64)
        self._count = 0

    def update(self, values: np.ndarray) -> None:
        """Add a batch of values. *values* shape: (batch, *self.shape)."""
        self._sum += values.astype(np.float64).sum(axis=0)
        self._count += values.shape[0]

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> np.ndarray:
        if self._count == 0:
            raise ValueError("No values accumulated yet")
        return (self._sum / self._count).astype(np.float32)


def _average_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_offset: int,
) -> np.ndarray | None:
    """Average hidden states over valid token positions from *token_offset* onward.

    Args:
        hidden_states: (batch, seq_len, hidden_dim) — already float, padding zeroed.
        attention_mask: (batch, seq_len).
        token_offset: skip the first N tokens.

    Returns:
        np.ndarray of shape (batch, hidden_dim), or None if all sequences are too short.
    """
    seq_len = hidden_states.shape[1]
    if seq_len <= token_offset:
        return None

    hs = hidden_states[:, token_offset:, :]  # (batch, remaining, hidden_dim)
    mask = attention_mask[:, token_offset:].unsqueeze(-1)  # (batch, remaining, 1)

    # Identify sequences with valid tokens after offset (before clamping)
    token_counts = mask.sum(dim=1)  # (batch, 1)
    valid = token_counts.squeeze(-1) > 0  # (batch,)
    if not valid.any():
        return None

    # Average over valid token positions, clamp to avoid division by zero
    avg = (hs * mask).sum(dim=1) / token_counts.clamp(min=1)  # (batch, hidden_dim)

    result = avg[valid].cpu().numpy()
    return result


class ActivationExtractor:
    """Extract and accumulate mean activations from stories or dialogues."""

    def __init__(self, model: EmotionProbeModel, config: ActivationConfig) -> None:
        self.model = model
        self.config = config

    def extract_story_activations(
        self,
        emotions: list[str],
        topics: list[str],
        output_dir: Path,
        layers: list[int] | None = None,
    ) -> None:
        """Extract mean activations for each emotion from generated stories.

        Saves per-layer files:
          output_dir/activations/emotion_means_layer_{L}.safetensors
          output_dir/activations/global_mean_layer_{L}.safetensors
          output_dir/activations/metadata.json
        """
        if layers is None:
            layers = [self.model.target_layer]

        # Initialize accumulators: {layer: {emotion: RunningMean}}
        hidden_dim = self.model.hidden_size
        accumulators: dict[int, dict[str, RunningMean]] = {
            layer: {emotion: RunningMean((hidden_dim,)) for emotion in emotions}
            for layer in layers
        }

        total_pairs = sum(
            1 for emotion in emotions for topic in topics
            if (output_dir / "stories" / emotion / f"{topic_hash(topic)}.json").exists()
        )

        with tqdm(total=total_pairs, desc="Extracting activations") as pbar:
            for emotion in emotions:
                for topic in topics:
                    stories_dir = output_dir / "stories" / emotion
                    topic_file = stories_dir / f"{topic_hash(topic)}.json"
                    if not topic_file.exists():
                        continue

                    stories = load_stories(output_dir, emotion, topic)
                    self._process_texts(stories, accumulators, layers, emotion)
                    pbar.update(1)

        self._save_results(accumulators, emotions, layers, output_dir / "activations")

    def extract_neutral_activations(
        self,
        topics: list[str],
        output_dir: Path,
        layers: list[int] | None = None,
    ) -> None:
        """Extract activations from neutral dialogues (for PCA denoising).

        Saves concatenated activations to:
          output_dir/activations/neutral_layer_{L}.safetensors
        """
        if layers is None:
            layers = [self.model.target_layer]

        hidden_dim = self.model.hidden_size
        # For neutral data, we store all individual averages (not per-emotion means)
        # because PCA needs the full distribution
        all_activations: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

        for topic in tqdm(topics, desc="Extracting neutral activations"):
            neutral_file = output_dir / "neutral" / f"{topic_hash(topic)}.json"
            if not neutral_file.exists():
                continue

            dialogues = load_neutral_dialogues(output_dir, topic)
            for batch_start in range(0, len(dialogues), self.config.batch_size):
                batch = dialogues[batch_start : batch_start + self.config.batch_size]
                hidden_states_dict, attention_mask = self.model.get_hidden_states(batch, layers)

                for layer in layers:
                    avg = _average_hidden_states(
                        hidden_states_dict[layer], attention_mask, self.config.token_offset,
                    )
                    if avg is not None:
                        all_activations[layer].append(avg)

        # Save concatenated neutral activations
        act_dir = output_dir / "activations"
        for layer in layers:
            if all_activations[layer]:
                concatenated = np.concatenate(all_activations[layer], axis=0)
                save_tensors(
                    {"activations": concatenated},
                    act_dir / f"neutral_layer_{layer}.safetensors",
                    metadata={"layer": str(layer), "n_samples": str(concatenated.shape[0])},
                )
                logger.info("Saved %d neutral activations for layer %d", concatenated.shape[0], layer)

    def _process_texts(
        self,
        texts: list[str],
        accumulators: dict[int, dict[str, RunningMean]],
        layers: list[int],
        emotion: str,
    ) -> None:
        """Forward-pass texts in batches and accumulate their mean activations."""
        batch_size = self.config.batch_size
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]

            hidden_states_dict, attention_mask = self.model.get_hidden_states(batch, layers)

            for layer in layers:
                avg = _average_hidden_states(
                    hidden_states_dict[layer], attention_mask, self.config.token_offset,
                )
                if avg is not None:
                    accumulators[layer][emotion].update(avg)

    def _save_results(
        self,
        accumulators: dict[int, dict[str, RunningMean]],
        emotions: list[str],
        layers: list[int],
        act_dir: Path,
    ) -> None:
        """Save per-emotion means and global means for each layer."""
        for layer in layers:
            # Stack per-emotion means: (num_emotions, hidden_dim)
            emotion_means = {}
            valid_emotions = []
            for emotion in emotions:
                acc = accumulators[layer][emotion]
                if acc.count > 0:
                    emotion_means[emotion] = acc.mean
                    valid_emotions.append(emotion)
                else:
                    logger.warning("No activations for emotion '%s' at layer %d", emotion, layer)

            if not emotion_means:
                logger.error("No activations extracted for layer %d", layer)
                continue

            stacked = np.stack([emotion_means[e] for e in valid_emotions])  # (N, hidden_dim)
            global_mean = stacked.mean(axis=0)  # (hidden_dim,)

            save_tensors(
                {"emotion_means": stacked},
                act_dir / f"emotion_means_layer_{layer}.safetensors",
                metadata={
                    "layer": str(layer),
                    "emotions": json.dumps(valid_emotions),
                    "n_emotions": str(len(valid_emotions)),
                },
            )
            save_tensors(
                {"global_mean": global_mean},
                act_dir / f"global_mean_layer_{layer}.safetensors",
                metadata={"layer": str(layer)},
            )
            logger.info(
                "Saved activations for layer %d: %d emotions, shape %s",
                layer, len(valid_emotions), stacked.shape,
            )

        # Save metadata
        save_metadata(
            {"emotions": emotions, "layers": layers, "model": self.model.config.name},
            act_dir / "metadata.json",
        )
