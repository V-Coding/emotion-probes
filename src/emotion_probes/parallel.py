"""Auto-detect multi-GPU and run data-parallel generation."""

from __future__ import annotations

import logging
import os
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_probes.config import PipelineConfig

logger = logging.getLogger(__name__)


def detect_parallel_gpus(model_size_gb: float) -> list[int]:
    """Return GPU indices that can each independently hold the model.

    Args:
        model_size_gb: Estimated model VRAM in GB (weights + overhead).

    Returns:
        List of GPU indices with enough memory. Empty if no CUDA GPUs.
    """
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    capable = []
    for i in range(torch.cuda.device_count()):
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        if total_gb >= model_size_gb:
            capable.append(i)

    return capable


def estimate_model_vram_gb(model_name: str, quantize: str | None) -> float:
    """Estimate VRAM needed for a model in GB.

    Fetches the model config from HuggingFace (lightweight, no weights downloaded)
    and estimates memory based on parameter count and precision.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Try to get num_params from config; estimate from architecture if not available
        text_cfg = getattr(config, "text_config", config)
        hidden = getattr(text_cfg, "hidden_size", 2048)
        layers = getattr(text_cfg, "num_hidden_layers", 32)
        intermediate = getattr(text_cfg, "intermediate_size", hidden * 4)
        vocab = getattr(text_cfg, "vocab_size", 256000)

        # Rough parameter estimate: embeddings + layers * (attn + ffn + norms)
        n_params = vocab * hidden + layers * (4 * hidden * hidden + 2 * hidden * intermediate + 4 * hidden)
        n_params_b = n_params / 1e9

        bytes_per_param = {"4bit": 0.5, "8bit": 1.0}.get(quantize, 2.0)  # bf16 = 2 bytes
        weight_gb = n_params_b * bytes_per_param
        overhead_gb = 2.0  # KV cache, activations, CUDA context
        total = weight_gb + overhead_gb

        logger.info("Estimated %.1fB params, %.1f GB VRAM (%s)", n_params_b, total, quantize or "bf16")
        return total
    except Exception:
        logger.warning("Could not estimate model size, assuming 16 GB")
        return 16.0


def _story_worker(
    gpu_id: int,
    emotions: list[str],
    topics: list[str],
    config_path: str,
    output_dir: str,
) -> dict[str, int]:
    """Worker process: load model on one GPU and generate stories for an emotion slice."""
    # Pin to a single GPU before any CUDA operations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from emotion_probes.config import load_config
    from emotion_probes.generate import StoryGenerator
    from emotion_probes.model import EmotionProbeModel

    cfg = load_config(config_path)
    model = EmotionProbeModel(cfg.model)
    generator = StoryGenerator(model, cfg.generation)
    return generator.generate_all(emotions, topics, Path(output_dir))


def _neutral_worker(
    gpu_id: int,
    topics: list[str],
    config_path: str,
    output_dir: str,
) -> dict[str, int]:
    """Worker process: load model on one GPU and generate neutral dialogues for a topic slice."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from emotion_probes.config import load_config
    from emotion_probes.generate import NeutralDialogueGenerator
    from emotion_probes.model import EmotionProbeModel

    cfg = load_config(config_path)
    model = EmotionProbeModel(cfg.model)
    generator = NeutralDialogueGenerator(model, cfg.generation)
    return generator.generate_all(topics, Path(output_dir))


def parallel_generate_stories(
    config_path: str,
    emotions: list[str],
    topics: list[str],
    output_dir: Path,
    gpu_ids: list[int],
) -> dict[str, int]:
    """Generate stories across multiple GPUs in parallel.

    Splits emotions evenly across GPUs. Each GPU loads its own model instance
    and processes its slice. Atomic file writes ensure safe concurrent output.
    """
    n_workers = len(gpu_ids)
    # Round-robin split: worker i gets emotions[i], emotions[i+n], emotions[i+2n], ...
    chunks = [emotions[i::n_workers] for i in range(n_workers)]

    logger.info("Parallel generation: %d workers across GPUs %s, emotions split %s",
                n_workers, sorted(set(gpu_ids)), [len(c) for c in chunks])

    tasks = [
        (gpu_ids[i], chunks[i], topics, config_path, str(output_dir))
        for i in range(n_workers) if chunks[i]
    ]

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        async_results = pool.starmap_async(_story_worker, tasks)
        try:
            results = async_results.get(timeout=None)  # no timeout, but interruptible
        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating workers")
            pool.terminate()
            raise

    return _merge_summaries(results)


def parallel_generate_neutral(
    config_path: str,
    topics: list[str],
    output_dir: Path,
    gpu_ids: list[int],
) -> dict[str, int]:
    """Generate neutral dialogues across multiple GPUs in parallel."""
    n_workers = len(gpu_ids)
    chunks = [topics[i::n_workers] for i in range(n_workers)]

    logger.info("Parallel neutral generation: %d workers across GPUs %s, topics split %s",
                n_workers, sorted(set(gpu_ids)), [len(c) for c in chunks])

    tasks = [
        (gpu_ids[i], chunks[i], config_path, str(output_dir))
        for i in range(n_workers) if chunks[i]
    ]

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        async_results = pool.starmap_async(_neutral_worker, tasks)
        try:
            results = async_results.get(timeout=None)
        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating workers")
            pool.terminate()
            raise

    return _merge_summaries(results)


def _merge_summaries(results: list[dict[str, int]]) -> dict[str, int]:
    """Merge summary dicts from parallel workers."""
    merged = {"generated": 0, "skipped": 0, "failed": 0}
    for r in results:
        for k in merged:
            merged[k] += r[k]
    return merged
