"""Shared test fixtures."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir() -> Path:
    return PROJECT_ROOT / "config"


@pytest.fixture
def emotions_path(config_dir: Path) -> Path:
    return config_dir / "emotions.txt"


@pytest.fixture
def topics_path(config_dir: Path) -> Path:
    return config_dir / "topics.txt"
