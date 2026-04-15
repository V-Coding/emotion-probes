"""CLI entry point with subcommands for each pipeline stage."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

from emotion_probes.config import PipelineConfig, load_config
from emotion_probes.prompts import load_emotions, load_topics

logger = logging.getLogger("emotion_probes")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_cfg(config: str) -> PipelineConfig:
    cfg = load_config(config)
    cfg.ensure_dirs()
    return cfg


def _resolve_gpus(cfg: PipelineConfig, gpus_override: int | None, workers_per_gpu: int = 1) -> list[int]:
    """Determine which GPUs to use for data-parallel generation.

    When workers_per_gpu > 1, each GPU ID is repeated that many times so the
    parallel module spawns multiple workers sharing the same device.
    """
    from emotion_probes.parallel import detect_parallel_gpus, estimate_model_vram_gb

    model_vram = estimate_model_vram_gb(cfg.model.name, cfg.model.quantize)
    capable = detect_parallel_gpus(model_vram * workers_per_gpu)

    if gpus_override is not None:
        n = min(gpus_override, len(capable))
        if n < gpus_override:
            click.echo(f"Requested {gpus_override} GPUs but only {len(capable)} can hold the model. Using {max(n, 1)}.")
        gpu_ids = capable[:max(n, 1)] if capable else []
    else:
        gpu_ids = capable

    # Expand: [0, 1] with workers_per_gpu=2 → [0, 0, 1, 1]
    if workers_per_gpu > 1 and gpu_ids:
        gpu_ids = [g for g in gpu_ids for _ in range(workers_per_gpu)]

    return gpu_ids


def _resolve_emotions_and_topics(cfg: PipelineConfig) -> tuple[list[str], list[str]]:
    emotions = load_emotions(cfg.emotions_file)
    topics = load_topics(cfg.topics_file)
    if cfg.generation.emotions_subset:
        emotions = emotions[: cfg.generation.emotions_subset]
    if cfg.generation.topics_subset:
        topics = topics[: cfg.generation.topics_subset]
    return emotions, topics


@click.group()
def main() -> None:
    """Emotion probes for open-source language models."""
    _setup_logging()


# ------------------------------------------------------------------
# generate-stories
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--gpus", default=None, type=int, help="Number of GPUs for data-parallel generation (default: auto-detect)")
@click.option("--workers-per-gpu", default=1, type=int, help="Number of worker processes per GPU (default: 1)")
def generate_stories(config: str, gpus: int | None, workers_per_gpu: int) -> None:
    """Generate emotion-labeled stories."""
    cfg = _load_cfg(config)
    emotions, topics = _resolve_emotions_and_topics(cfg)

    click.echo(f"Generating stories: {len(emotions)} emotions x {len(topics)} topics x {cfg.generation.stories_per_topic} stories/topic")

    gpu_ids = _resolve_gpus(cfg, gpus, workers_per_gpu)
    config_abs = str(Path(config).resolve())

    if len(gpu_ids) > 1:
        from emotion_probes.parallel import parallel_generate_stories
        click.echo(f"Using {len(gpu_ids)} workers across GPUs: {gpu_ids}")
        summary = parallel_generate_stories(config_abs, emotions, topics, cfg.output_dir, gpu_ids)
    else:
        from emotion_probes.model import EmotionProbeModel
        from emotion_probes.generate import StoryGenerator
        model = EmotionProbeModel(cfg.model)
        generator = StoryGenerator(model, cfg.generation)
        summary = generator.generate_all(emotions, topics, cfg.output_dir)

    click.echo(f"Done: {summary}")


# ------------------------------------------------------------------
# generate-neutral
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--gpus", default=None, type=int, help="Number of GPUs for data-parallel generation (default: auto-detect)")
@click.option("--workers-per-gpu", default=1, type=int, help="Number of worker processes per GPU (default: 1)")
def generate_neutral(config: str, gpus: int | None, workers_per_gpu: int) -> None:
    """Generate emotionally neutral dialogues."""
    cfg = _load_cfg(config)
    _, topics = _resolve_emotions_and_topics(cfg)

    click.echo(f"Generating neutral dialogues: {len(topics)} topics x {cfg.generation.neutral_dialogues_per_topic} dialogues/topic")

    gpu_ids = _resolve_gpus(cfg, gpus, workers_per_gpu)
    config_abs = str(Path(config).resolve())

    if len(gpu_ids) > 1:
        from emotion_probes.parallel import parallel_generate_neutral
        click.echo(f"Using {len(gpu_ids)} workers across GPUs: {gpu_ids}")
        summary = parallel_generate_neutral(config_abs, topics, cfg.output_dir, gpu_ids)
    else:
        from emotion_probes.model import EmotionProbeModel
        from emotion_probes.generate import NeutralDialogueGenerator
        model = EmotionProbeModel(cfg.model)
        generator = NeutralDialogueGenerator(model, cfg.generation)
        summary = generator.generate_all(topics, cfg.output_dir)

    click.echo(f"Done: {summary}")


# ------------------------------------------------------------------
# extract-activations
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--source", type=click.Choice(["stories", "neutral", "both"]), default="both", help="Which data to extract from")
def extract_activations(config: str, source: str) -> None:
    """Extract hidden state activations from generated text."""
    cfg = _load_cfg(config)
    emotions, topics = _resolve_emotions_and_topics(cfg)

    from emotion_probes.model import EmotionProbeModel
    from emotion_probes.activations import ActivationExtractor

    model = EmotionProbeModel(cfg.model)
    layers = list(range(model.num_layers + 1)) if cfg.model.all_layers else [model.target_layer]
    extractor = ActivationExtractor(model, cfg.activation)

    if source in ("stories", "both"):
        click.echo(f"Extracting story activations: {len(emotions)} emotions, layers={layers}")
        extractor.extract_story_activations(emotions, topics, cfg.output_dir, layers)

    if source in ("neutral", "both"):
        click.echo(f"Extracting neutral activations: {len(topics)} topics, layers={layers}")
        extractor.extract_neutral_activations(topics, cfg.output_dir, layers)

    click.echo("Activation extraction complete.")


# ------------------------------------------------------------------
# compute-vectors
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
def compute_vectors(config: str) -> None:
    """Compute denoised emotion vectors from activations."""
    cfg = _load_cfg(config)

    from emotion_probes.model import EmotionProbeModel
    from emotion_probes.vectors import compute_emotion_vectors

    model = EmotionProbeModel(cfg.model)
    layers = list(range(model.num_layers + 1)) if cfg.model.all_layers else [model.target_layer]

    # Check prerequisites
    act_dir = cfg.output_dir / "activations"
    for layer in layers:
        if not (act_dir / f"emotion_means_layer_{layer}.safetensors").exists():
            click.echo(f"Error: Missing activations for layer {layer}. Run extract-activations first.", err=True)
            sys.exit(1)

    vectors = compute_emotion_vectors(
        activations_dir=act_dir,
        output_dir=cfg.output_dir,
        layers=layers,
        pca_variance_threshold=cfg.analysis.pca_variance_threshold,
    )

    for layer, v in vectors.items():
        click.echo(f"Layer {layer}: {v.shape[0]} emotion vectors of dim {v.shape[1]}")


# ------------------------------------------------------------------
# analyze
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
def analyze(config: str) -> None:
    """Run geometric analysis on emotion vectors."""
    cfg = _load_cfg(config)

    from emotion_probes.analysis import (
        compute_cosine_similarity,
        compute_kmeans,
        compute_pca,
        compute_umap,
        interpret_pca_components,
    )
    from emotion_probes.data import load_vectors_with_metadata, save_metadata

    # Find available vector files
    vectors_dir = cfg.output_dir / "vectors"
    vector_files = sorted(vectors_dir.glob("emotion_vectors_layer_*.safetensors"))
    if not vector_files:
        click.echo("Error: No emotion vectors found. Run compute-vectors first.", err=True)
        sys.exit(1)

    for vf in vector_files:
        vectors, emotion_names, layer = load_vectors_with_metadata(vf)

        click.echo(f"\n--- Layer {layer}: {len(emotion_names)} emotions ---")

        # Cosine similarity
        sim = compute_cosine_similarity(vectors)
        np.save(cfg.output_dir / "analysis" / f"cosine_sim_layer_{layer}.npy", sim)

        # PCA
        pca_coords, pca = compute_pca(vectors, n_components=10)
        np.save(cfg.output_dir / "analysis" / f"pca_coords_layer_{layer}.npy", pca_coords)

        # Interpret PCA
        pca_interp = interpret_pca_components(pca, emotion_names, vectors)
        save_metadata(pca_interp, cfg.output_dir / "analysis" / f"pca_interpretation_layer_{layer}.json")
        for pc, data_pc in list(pca_interp.items())[:2]:
            top_str = ", ".join(f"{n}" for n, _ in data_pc["top"][:5])
            bottom_str = ", ".join(f"{n}" for n, _ in data_pc["bottom"][:5])
            click.echo(f"  {pc} top: {top_str}")
            click.echo(f"  {pc} bottom: {bottom_str}")

        # K-means
        labels = compute_kmeans(vectors, k=cfg.analysis.kmeans_k, seed=cfg.seed)
        np.save(cfg.output_dir / "analysis" / f"kmeans_labels_layer_{layer}.npy", labels)

        # UMAP
        if vectors.shape[0] > cfg.analysis.umap_n_neighbors + 1:
            umap_coords = compute_umap(
                vectors,
                n_neighbors=cfg.analysis.umap_n_neighbors,
                min_dist=cfg.analysis.umap_min_dist,
                seed=cfg.seed,
            )
            np.save(cfg.output_dir / "analysis" / f"umap_coords_layer_{layer}.npy", umap_coords)
        else:
            click.echo("  Skipping UMAP (too few emotions for n_neighbors)")

    click.echo("\nAnalysis complete.")


# ------------------------------------------------------------------
# visualize
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
def visualize(config: str) -> None:
    """Generate visualization plots from analysis results."""
    cfg = _load_cfg(config)

    from emotion_probes.visualize import (
        plot_cosine_heatmap,
        plot_pca_scatter,
        plot_umap_scatter,
        plot_variance_explained,
    )

    vectors_dir = cfg.output_dir / "vectors"
    analysis_dir = cfg.output_dir / "analysis"
    figures_dir = cfg.output_dir / "figures"

    vector_files = sorted(vectors_dir.glob("emotion_vectors_layer_*.safetensors"))
    if not vector_files:
        click.echo("Error: No emotion vectors found. Run compute-vectors first.", err=True)
        sys.exit(1)

    for vf in vector_files:
        from emotion_probes.data import load_vectors_with_metadata
        vectors_data, emotion_names, layer = load_vectors_with_metadata(vf)

        click.echo(f"Generating plots for layer {layer}...")

        # Cosine heatmap
        sim_path = analysis_dir / f"cosine_sim_layer_{layer}.npy"
        if sim_path.exists():
            sim = np.load(sim_path)
            plot_cosine_heatmap(sim, emotion_names, figures_dir / f"cosine_heatmap_layer_{layer}.png")

        # PCA scatter
        pca_path = analysis_dir / f"pca_coords_layer_{layer}.npy"
        labels_path = analysis_dir / f"kmeans_labels_layer_{layer}.npy"
        if pca_path.exists():
            pca_coords = np.load(pca_path)
            labels = np.load(labels_path) if labels_path.exists() else None
            plot_pca_scatter(pca_coords, emotion_names, labels, figures_dir / f"pca_scatter_layer_{layer}.png")

        # UMAP scatter
        umap_path = analysis_dir / f"umap_coords_layer_{layer}.npy"
        if umap_path.exists():
            umap_coords = np.load(umap_path)
            labels = np.load(labels_path) if labels_path.exists() else None
            plot_umap_scatter(umap_coords, emotion_names, labels, figures_dir / f"umap_scatter_layer_{layer}.png")

        # Variance explained (from PCA on the vectors themselves)
        from sklearn.decomposition import PCA as PCA_
        pca_full = PCA_().fit(vectors_data)
        plot_variance_explained(
            pca_full.explained_variance_ratio_[:20],
            figures_dir / f"variance_explained_layer_{layer}.png",
        )

    click.echo("Visualization complete.")


# ------------------------------------------------------------------
# logit-lens
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--emotions", default=None, help="Comma-separated subset of emotions to show")
def logit_lens(config: str, emotions: str | None) -> None:
    """Project emotion vectors through the unembedding matrix."""
    cfg = _load_cfg(config)

    from emotion_probes.data import save_metadata
    from emotion_probes.logit_lens import compute_logit_lens
    from emotion_probes.model import EmotionProbeModel
    from emotion_probes.visualize import plot_logit_lens_table

    model = EmotionProbeModel(cfg.model)

    vectors_dir = cfg.output_dir / "vectors"
    vector_files = sorted(vectors_dir.glob("emotion_vectors_layer_*.safetensors"))
    if not vector_files:
        click.echo("Error: No emotion vectors found. Run compute-vectors first.", err=True)
        sys.exit(1)

    unembedding = model.get_unembedding_matrix()

    for vf in vector_files:
        from emotion_probes.data import load_vectors_with_metadata
        vectors, emotion_names, layer = load_vectors_with_metadata(vf)

        click.echo(f"Computing logit lens for layer {layer}...")
        results = compute_logit_lens(
            vectors, unembedding, model.tokenizer,
            emotion_names, top_k=cfg.analysis.logit_lens_top_k,
        )

        save_metadata(results, cfg.output_dir / "analysis" / f"logit_lens_layer_{layer}.json")

        # Plot table for a subset
        show_emotions = [e.strip() for e in emotions.split(",")] if emotions else emotion_names[:12]
        plot_logit_lens_table(
            results, show_emotions,
            cfg.output_dir / "figures" / f"logit_lens_layer_{layer}.png",
            top_k=min(5, cfg.analysis.logit_lens_top_k),
        )

        # Print a few examples
        for emo in show_emotions[:3]:
            if emo in results:
                top_str = ", ".join(f"{t} ({sc:.1f})" for t, sc in results[emo]["top"][:5])
                click.echo(f"  {emo}: {top_str}")

    click.echo("Logit lens complete.")


# ------------------------------------------------------------------
# run-all
# ------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--gpus", default=None, type=int, help="Number of GPUs for parallel generation stages")
@click.option("--workers-per-gpu", default=1, type=int, help="Number of worker processes per GPU (default: 1)")
@click.pass_context
def run_all(ctx: click.Context, config: str, gpus: int | None, workers_per_gpu: int) -> None:
    """Run the full pipeline end-to-end."""
    click.echo("=" * 60)
    click.echo("EMOTION PROBES — Full Pipeline")
    click.echo("=" * 60)

    # Stages that accept --gpus
    gpu_stages = {"generate-stories", "generate-neutral"}

    stages = [
        ("generate-stories", generate_stories),
        ("generate-neutral", generate_neutral),
        ("extract-activations", extract_activations),
        ("compute-vectors", compute_vectors),
        ("analyze", analyze),
        ("visualize", visualize),
        ("logit-lens", logit_lens),
    ]

    for name, cmd in stages:
        click.echo(f"\n{'='*60}")
        click.echo(f"Stage: {name}")
        click.echo(f"{'='*60}")
        try:
            kwargs = {"config": config}
            if name in gpu_stages:
                kwargs["gpus"] = gpus
                kwargs["workers_per_gpu"] = workers_per_gpu
            ctx.invoke(cmd, **kwargs)
        except SystemExit as e:
            if e.code != 0:
                click.echo(f"Stage {name} failed with exit code {e.code}", err=True)
                sys.exit(e.code)
        except Exception:
            logger.exception("Stage %s failed", name)
            sys.exit(1)

    click.echo(f"\n{'='*60}")
    cfg = _load_cfg(config)
    click.echo(f"Pipeline complete! Results in: {cfg.output_dir}")
    click.echo(f"{'='*60}")
