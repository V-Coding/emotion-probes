"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from emotion_probes.config import PipelineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.model.name == "google/gemma-4-E4B-it"
        assert cfg.model.torch_dtype == "bfloat16"
        assert cfg.generation.stories_per_topic == 12
        assert cfg.activation.token_offset == 50
        assert cfg.analysis.pca_variance_threshold == 0.5

    def test_ensure_dirs(self, tmp_path: Path):
        cfg = PipelineConfig(output_dir=tmp_path / "out", cache_dir=tmp_path / "out" / "cache")
        cfg.ensure_dirs()
        assert (tmp_path / "out" / "stories").is_dir()
        assert (tmp_path / "out" / "neutral").is_dir()
        assert (tmp_path / "out" / "activations").is_dir()
        assert (tmp_path / "out" / "vectors").is_dir()
        assert (tmp_path / "out" / "analysis").is_dir()
        assert (tmp_path / "out" / "figures").is_dir()

    def test_quantize_validation(self):
        cfg = PipelineConfig(model={"name": "test", "quantize": "4bit"})
        assert cfg.model.quantize == "4bit"

        with pytest.raises(Exception):
            PipelineConfig(model={"name": "test", "quantize": "3bit"})


class TestLoadConfig:
    def test_load_default_yaml(self):
        cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
        assert cfg.model.name == "google/gemma-4-E4B-it"
        assert cfg.generation.stories_per_topic == 12
        assert cfg.generation.topics_subset is None
        assert cfg.generation.emotions_subset is None
        assert cfg.seed == 42

    def test_load_medium_yaml(self):
        cfg = load_config(PROJECT_ROOT / "config" / "medium.yaml")
        assert cfg.model.quantize is None
        assert cfg.generation.stories_per_topic == 12
        assert cfg.emotions_file is not None
