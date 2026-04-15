"""Story and neutral dialogue generation with resumability."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from tqdm import tqdm

from emotion_probes.data import (
    neutral_exists,
    save_neutral_dialogues,
    save_stories,
    stories_exist,
)
from emotion_probes.prompts import format_neutral_dialogue_prompt, format_story_prompt

if TYPE_CHECKING:
    from pathlib import Path

    from emotion_probes.config import GenerationConfig
    from emotion_probes.model import EmotionProbeModel

logger = logging.getLogger(__name__)


def _parse_stories(text: str) -> list[str]:
    """Parse generated text into individual stories using <new story> delimiter."""
    # Split on <new story> (case-insensitive, with optional whitespace)
    parts = re.split(r"<\s*new\s+story\s*>", text, flags=re.IGNORECASE)
    stories = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            stories.append(cleaned)
    return stories


def _parse_dialogues(text: str) -> list[str]:
    """Parse generated text into individual dialogues using <new dialogue> delimiter."""
    parts = re.split(r"<\s*new\s+dialogue\s*>", text, flags=re.IGNORECASE)
    dialogues = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            # Post-hoc replacement per the paper
            cleaned = cleaned.replace("Person:", "Human:")
            cleaned = cleaned.replace("AI:", "Assistant:")
            dialogues.append(cleaned)
    return dialogues


class StoryGenerator:
    """Generates emotion-labeled stories with per-(emotion, topic) checkpointing."""

    def __init__(self, model: EmotionProbeModel, config: GenerationConfig) -> None:
        self.model = model
        self.config = config

    def generate_all(
        self,
        emotions: list[str],
        topics: list[str],
        output_dir: Path,
    ) -> dict[str, int]:
        """Generate stories for all (emotion, topic) pairs.

        Returns a summary dict with counts of generated, skipped, and failed pairs.
        """
        n_stories = self.config.stories_per_topic
        summary = {"generated": 0, "skipped": 0, "failed": 0}
        total = len(emotions) * len(topics)

        with tqdm(total=total, desc="Generating stories") as pbar:
            for emotion in emotions:
                for topic in topics:
                    pbar.set_postfix_str(f"{emotion}", refresh=False)

                    if stories_exist(output_dir, emotion, topic):
                        summary["skipped"] += 1
                        pbar.update(1)
                        continue

                    prompt = format_story_prompt(
                        n_stories=n_stories,
                        topic=topic,
                        emotion=emotion,
                    )
                    try:
                        responses = self.model.generate(
                            [prompt],
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                        )
                        stories = _parse_stories(responses[0])

                        if len(stories) < n_stories:
                            logger.warning(
                                "Parsed %d/%d stories for (%s, %s)",
                                len(stories), n_stories, emotion, topic,
                            )

                        if stories:
                            save_stories(output_dir, emotion, topic, stories, self.model.config.name)
                            summary["generated"] += 1
                        else:
                            logger.error("No stories parsed for (%s, %s)", emotion, topic)
                            summary["failed"] += 1
                    except (RuntimeError, ValueError, OSError):
                        logger.exception("Failed to generate stories for (%s, %s)", emotion, topic)
                        summary["failed"] += 1

                    pbar.update(1)

        return summary


class NeutralDialogueGenerator:
    """Generates emotionally neutral dialogues with per-topic checkpointing."""

    def __init__(self, model: EmotionProbeModel, config: GenerationConfig) -> None:
        self.model = model
        self.config = config

    def generate_all(
        self,
        topics: list[str],
        output_dir: Path,
    ) -> dict[str, int]:
        """Generate neutral dialogues for all topics.

        Returns a summary dict with counts.
        """
        n_dialogues = self.config.neutral_dialogues_per_topic
        summary = {"generated": 0, "skipped": 0, "failed": 0}

        for topic in tqdm(topics, desc="Generating neutral dialogues"):
            if neutral_exists(output_dir, topic):
                summary["skipped"] += 1
                continue

            prompt = format_neutral_dialogue_prompt(
                n_stories=n_dialogues,
                topic=topic,
            )
            try:
                responses = self.model.generate(
                    [prompt],
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                dialogues = _parse_dialogues(responses[0])

                if len(dialogues) < n_dialogues:
                    logger.warning(
                        "Parsed %d/%d dialogues for topic: %s",
                        len(dialogues), n_dialogues, topic[:60],
                    )

                if dialogues:
                    save_neutral_dialogues(output_dir, topic, dialogues, self.model.config.name)
                    summary["generated"] += 1
                else:
                    logger.error("No dialogues parsed for topic: %s", topic[:60])
                    summary["failed"] += 1
            except (RuntimeError, ValueError, OSError):
                logger.exception("Failed to generate dialogues for topic: %s", topic[:60])
                summary["failed"] += 1

        return summary
