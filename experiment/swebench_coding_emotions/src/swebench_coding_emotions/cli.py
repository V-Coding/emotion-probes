"""`swebench-emotions` CLI — fetch / replay / analyze / viz subcommands.

The probe-build phase is handled by the upstream ``emotion-probes`` CLI; see
``scripts/01_build_probes.sh``. This CLI covers phases 2-5.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

from emotion_probes.config import load_config

from . import analysis as A
from . import augment as AUG
from . import replay as RP
from . import trajectories as TR
from . import viz as VZ

logger = logging.getLogger("swebench_coding_emotions")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _experiment_root(config_path: Path) -> Path:
    """Return the experiment folder, i.e. two levels above ``config/deepswe.yaml``."""
    return Path(config_path).resolve().parent.parent


def _swebench_dir(config_path: Path) -> Path:
    return _experiment_root(config_path) / "output" / "swebench"


def _probes_dir(config_path: Path) -> Path:
    """Resolve ``cfg.output_dir`` against the experiment root if it's relative."""
    cfg = load_config(config_path)
    p = Path(cfg.output_dir)
    if p.is_absolute():
        return p
    return _experiment_root(config_path) / p


@click.group()
@click.option("--verbose", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    """Emotion probes on SWE-bench coding trajectories."""
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--n-tasks", default=24, type=int, help="Total tasks (divisible by 4).")
@click.option(
    "--submission",
    default="20250629_deepswerl_r2eagent",
    help="Submission folder under s3://swe-bench-submissions/verified/",
)
@click.option("--limit", default=None, type=int, help="Optionally cap how many instances to download.")
def fetch(config: str, n_tasks: int, submission: str, limit: int | None) -> None:
    """Download DeepSWE trajectories and write a stratified sample manifest."""
    swebench_dir = _swebench_dir(config)
    cache_dir = swebench_dir / "raw"
    sample_path = swebench_dir / "sample.json"

    click.echo(f"Cache dir: {cache_dir}")
    trajs = TR.fetch_all(cache_dir, submission=submission, limit=limit)
    click.echo(f"Total fetched: {len(trajs)}")

    sample = TR.stratified_sample(trajs, n=n_tasks)
    TR.save_sample(sample, sample_path)
    click.echo(f"Sample manifest ({len(sample)} tasks): {sample_path}")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--layer", default=None, type=int, help="Override cfg.model.target_layer.")
@click.option("--window-tokens", default=8192, type=int)
@click.option("--stride-tokens", default=7168, type=int)
@click.option(
    "--variant",
    type=click.Choice(sorted(RP.VARIANT_TO_FILENAME.keys())),
    default="denoised",
    show_default=True,
    help="Which probe vectors to load: denoised (upstream PCA), augmented (this experiment), or raw.",
)
@click.option("--dry-run", is_flag=True, help="Skip model load; just check prerequisites.")
def replay(
    config: str, layer: int | None, window_tokens: int, stride_tokens: int,
    variant: str, dry_run: bool,
) -> None:
    """Replay each sampled trajectory through the model and save probe scores."""
    swebench_dir = _swebench_dir(config)
    sample_path = swebench_dir / "sample.json"
    raw_dir = swebench_dir / "raw"
    out_dir = swebench_dir / "replay" / variant

    if not sample_path.exists():
        click.echo(f"Error: no sample at {sample_path}. Run `swebench-emotions fetch` first.", err=True)
        sys.exit(1)

    sample_records = json.loads(sample_path.read_text(encoding="utf-8"))
    click.echo(f"Replaying {len(sample_records)} tasks → {out_dir}")

    cfg = load_config(config)
    layer = layer if layer is not None else cfg.model.target_layer
    if layer is None:
        click.echo("Error: cfg.model.target_layer is null and no --layer override given.", err=True)
        sys.exit(1)

    probes_dir = _probes_dir(config)
    vectors, emotions, global_mean = RP.load_probes(probes_dir, layer=layer, variant=variant)
    click.echo(f"Loaded {len(emotions)} emotion probes from {probes_dir} (layer={layer}, variant={variant})")

    fingerprint = RP.compute_fingerprint(cfg.model.name, cfg.model.torch_dtype, cfg.model.quantize, layer)

    if dry_run:
        click.echo(f"Fingerprint: {fingerprint}")
        click.echo("Dry run complete — model not loaded.")
        return

    from emotion_probes.model import EmotionProbeModel

    model = EmotionProbeModel(cfg.model)
    click.echo(f"Model loaded: {cfg.model.name} (layers={model.num_layers}, hidden={model.hidden_size})")

    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in sample_records:
        iid = rec["instance_id"]
        out_path = out_dir / f"{iid}.safetensors"
        if out_path.exists():
            existing_variant = RP.read_replay_variant(out_path)
            if existing_variant is not None and existing_variant != variant:
                click.echo(
                    f"Error: {out_path} was written with variant={existing_variant!r} "
                    f"but this run requested variant={variant!r}. Refusing to silently mix variants. "
                    f"Delete the file or run with the matching variant.",
                    err=True,
                )
                sys.exit(1)
            click.echo(f"  skip {iid} (already replayed, variant={existing_variant or '?'})")
            continue
        traj_path = raw_dir / "trajs" / f"{iid}.txt"
        if not traj_path.exists():
            click.echo(f"  skip {iid} (no transcript cached)")
            continue
        text = traj_path.read_text(encoding="utf-8")
        click.echo(f"  replay {iid}  resolved={rec['resolved']}  difficulty={rec['difficulty']}  chars={len(text)}")
        result = RP.replay_trajectory(
            model,
            vectors,
            global_mean,
            emotions,
            text,
            instance_id=iid,
            resolved=rec["resolved"],
            difficulty=rec["difficulty"],
            layer=layer,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
            model_fingerprint=fingerprint,
            probe_variant=variant,
        )
        RP.save_replay(result, out_path)
        # Free any CUDA cache between tasks
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    click.echo("Replay complete.")


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--layer", default=None, type=int, help="Override cfg.model.target_layer.")
@click.option(
    "--max-length", default=2048, type=int,
    help="Max tokens per augmented neutral text (forward-pass cap).",
)
@click.option("--dry-run", is_flag=True, help="Skip model load; just preview the texts.")
def augment(config: str, layer: int | None, max_length: int, dry_run: bool) -> None:
    """Build the experiment-local augmented neutral set and refit PCA denoising.

    Reads ``output/probes/vectors/raw_vectors_layer_{L}.safetensors`` and
    ``output/probes/activations/neutral_layer_{L}.safetensors`` from the
    upstream probe build, generates structurally-relevant neutral stimuli
    (chat-template / tool-call / code / diff / <think> surfaces), extracts
    their activations, unions with the upstream neutrals, refits PCA, and
    writes ``emotion_vectors_augmented_layer_{L}.safetensors``.
    """
    cfg = load_config(config)
    layer = layer if layer is not None else cfg.model.target_layer
    if layer is None:
        click.echo("Error: cfg.model.target_layer is null and no --layer override given.", err=True)
        sys.exit(1)

    probes_dir = _probes_dir(config)
    cache_dir = probes_dir / "neutral_augmented"

    texts = AUG.build_augmented_neutral_texts()
    click.echo(f"Augmented neutral set: {len(texts)} texts")
    click.echo(f"  example: {texts[0][:80]!r}…")

    if dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            (cache_dir / f"{i:03d}.txt").write_text(text, encoding="utf-8")
        click.echo(f"Dry run: wrote {len(texts)} texts to {cache_dir}; model not loaded.")
        return

    from emotion_probes.model import EmotionProbeModel
    model = EmotionProbeModel(cfg.model)
    click.echo(f"Model loaded: {cfg.model.name} (layer={layer})")

    out_path = AUG.run_augment(
        probes_dir=probes_dir,
        layer=layer,
        token_offset=cfg.activation.token_offset,
        batch_size=cfg.activation.batch_size,
        max_length=max_length,
        pca_variance_threshold=cfg.analysis.pca_variance_threshold,
        model=model,
        cache_dir=cache_dir,
    )
    click.echo(f"Augmented vectors written to {out_path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option(
    "--variant",
    type=click.Choice(sorted(RP.VARIANT_TO_FILENAME.keys())),
    default="denoised",
    show_default=True,
    help="Read replay files from output/swebench/replay/{variant}/.",
)
@click.option(
    "--token-offset", default=A.DEFAULT_TOKEN_OFFSET, type=int, show_default=True,
    help="Drop the first N token positions from per-task aggregation.",
)
@click.option(
    "--keep-special-tokens", is_flag=True,
    help="Keep tokenizer special-token positions (default: drop them).",
)
def analyze(config: str, variant: str, token_offset: int, keep_special_tokens: bool) -> None:
    """Aggregate per-task scores, run MW / OLS / decile permutation tests."""
    swebench_dir = _swebench_dir(config)
    replay_dir = swebench_dir / "replay" / variant
    out_dir = swebench_dir / "analysis" / variant

    if not any(replay_dir.glob("*.safetensors")):
        click.echo(
            f"Error: no replay files in {replay_dir}. "
            f"Run `swebench-emotions replay --variant {variant}` first.",
            err=True,
        )
        sys.exit(1)

    A.run_full_analysis(
        replay_dir, out_dir,
        token_offset=token_offset,
        drop_special_tokens=not keep_special_tokens,
    )
    click.echo(
        f"Wrote per_task.csv, stats.csv, summary.json to {out_dir} "
        f"(variant={variant}, token_offset={token_offset}, drop_special={not keep_special_tokens})"
    )


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------
@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--section", default="all", type=click.Choice(list(A.SECTIONS)))
@click.option(
    "--variant",
    type=click.Choice(sorted(RP.VARIANT_TO_FILENAME.keys())),
    default="denoised",
    show_default=True,
    help="Read replay/analysis files for this variant.",
)
def viz(config: str, section: str, variant: str) -> None:
    """Generate per-task timelines + outcome/heatmap/decile plots."""
    import pandas as pd

    swebench_dir = _swebench_dir(config)
    replay_dir = swebench_dir / "replay" / variant
    analysis_dir = swebench_dir / "analysis" / variant
    figures_dir = swebench_dir / "figures" / variant

    per_task_path = analysis_dir / "per_task.csv"
    if not per_task_path.exists():
        click.echo(
            f"Error: {per_task_path} missing; "
            f"run `swebench-emotions analyze --variant {variant}` first.",
            err=True,
        )
        sys.exit(1)
    per_task_df = pd.read_csv(per_task_path)

    # Per-task timelines
    timelines_dir = figures_dir / "timelines"
    for rp in sorted(replay_dir.glob("*.safetensors")):
        iid = rp.stem
        click.echo(f"  timeline: {iid}")
        VZ.plot_timeline(rp, timelines_dir / f"{iid}.png")

    click.echo("  box_by_outcome.png")
    VZ.plot_box_by_outcome(per_task_df, figures_dir / f"box_by_outcome_{section}.png", section=section)

    click.echo("  heatmap_tasks.png")
    VZ.plot_task_heatmap(per_task_df, figures_dir / f"heatmap_tasks_{section}.png", section=section)

    click.echo("  decile_curves.png")
    VZ.plot_decile_curves(replay_dir, figures_dir / f"decile_curves_{section}.png", section=section)

    click.echo(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
