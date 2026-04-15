"""Configuration schema and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    name: str = "google/gemma-4-E4B-it"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    quantize: Literal["4bit", "8bit"] | None = None
    target_layer: int | None = None  # None = auto (2/3 of num_layers)
    all_layers: bool = False


class GenerationConfig(BaseModel):
    stories_per_topic: int = 12
    topics_subset: int | None = None
    emotions_subset: int | None = None
    batch_size: int = 4
    max_new_tokens: int = 2048
    temperature: float = 0.8
    neutral_dialogues_per_topic: int = 12


class ActivationConfig(BaseModel):
    token_offset: int = 50
    batch_size: int = 8
    storage_format: Literal["safetensors"] = "safetensors"


class AnalysisConfig(BaseModel):
    pca_variance_threshold: float = 0.5
    kmeans_k: int = 10
    logit_lens_top_k: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


class PipelineConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    activation: ActivationConfig = ActivationConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    emotions_file: Path | None = None  # None = default (config/emotions.txt)
    topics_file: Path | None = None  # None = default (config/topics.txt)
    seed: int = 42

    @field_validator("output_dir", "cache_dir", "emotions_file", "topics_file", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v)

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stories").mkdir(exist_ok=True)
        (self.output_dir / "neutral").mkdir(exist_ok=True)
        (self.output_dir / "activations").mkdir(exist_ok=True)
        (self.output_dir / "vectors").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file.

    Relative paths for emotions_file and topics_file are resolved
    relative to the config file's directory, so they work correctly
    in spawned subprocesses regardless of working directory.
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    cfg = PipelineConfig(**raw)

    # Resolve file paths relative to config directory
    config_dir = path.parent
    if cfg.emotions_file and not cfg.emotions_file.is_absolute():
        cfg.emotions_file = (config_dir / cfg.emotions_file).resolve()
    if cfg.topics_file and not cfg.topics_file.is_absolute():
        cfg.topics_file = (config_dir / cfg.topics_file).resolve()
    return cfg
