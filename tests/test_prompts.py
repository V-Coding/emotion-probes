"""Tests for prompt templates and data loading."""

from pathlib import Path

from emotion_probes.prompts import (
    format_neutral_dialogue_prompt,
    format_story_prompt,
    load_emotions,
    load_topics,
)


class TestLoadEmotions:
    def test_count(self, emotions_path: Path):
        emotions = load_emotions(emotions_path)
        assert len(emotions) == 171

    def test_known_entries(self, emotions_path: Path):
        emotions = load_emotions(emotions_path)
        for word in ["happy", "sad", "afraid", "calm", "desperate", "angry"]:
            assert word in emotions, f"'{word}' missing from emotions list"

    def test_no_blanks(self, emotions_path: Path):
        emotions = load_emotions(emotions_path)
        for e in emotions:
            assert e == e.strip()
            assert len(e) > 0


class TestLoadTopics:
    def test_count(self, topics_path: Path):
        topics = load_topics(topics_path)
        assert len(topics) == 100

    def test_no_blanks(self, topics_path: Path):
        topics = load_topics(topics_path)
        for t in topics:
            assert t == t.strip()
            assert len(t) > 0


class TestFormatStoryPrompt:
    def test_basic_format(self):
        prompt = format_story_prompt(n_stories=3, topic="A lost dog", emotion="sad")
        assert "3 different stories" in prompt
        assert "A lost dog" in prompt
        assert "feeling sad" in prompt
        assert "NEVER use the word 'sad'" in prompt

    def test_no_leftover_placeholders(self):
        prompt = format_story_prompt(n_stories=12, topic="Test topic", emotion="happy")
        assert "{" not in prompt


class TestFormatNeutralDialoguePrompt:
    def test_basic_format(self):
        prompt = format_neutral_dialogue_prompt(n_stories=5, topic="Weather patterns")
        assert "5 different dialogues" in prompt
        assert "Weather patterns" in prompt
        assert "neutral and emotionless" in prompt

    def test_no_leftover_placeholders(self):
        prompt = format_neutral_dialogue_prompt(n_stories=3, topic="Test")
        assert "{" not in prompt
