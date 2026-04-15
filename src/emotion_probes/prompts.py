"""Prompt templates from the paper and data loading helpers."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
DEFAULT_EMOTIONS_PATH = _CONFIG_DIR / "emotions.txt"
DEFAULT_TOPICS_PATH = _CONFIG_DIR / "topics.txt"

# ---------------------------------------------------------------------------
# Prompt templates based on the paper's methodology
# ---------------------------------------------------------------------------

STORY_PROMPT = """\
Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

<new story>
[story 1]
<new story>
[story 2]
<new story>
[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

NEUTRAL_DIALOGUE_PROMPT = """\
Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

<new dialogue>
[optional system instructions]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]

<new dialogue>

[dialogue 2]

etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_story_prompt(n_stories: int, topic: str, emotion: str) -> str:
    """Format the story generation prompt with the given parameters."""
    return STORY_PROMPT.format(n_stories=n_stories, topic=topic, emotion=emotion)


def format_neutral_dialogue_prompt(n_stories: int, topic: str) -> str:
    """Format the neutral dialogue generation prompt."""
    return NEUTRAL_DIALOGUE_PROMPT.format(n_stories=n_stories, topic=topic)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_emotions(path: str | Path | None = None) -> list[str]:
    """Load emotion words from a text file (one per line)."""
    path = Path(path) if path else DEFAULT_EMOTIONS_PATH
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_topics(path: str | Path | None = None) -> list[str]:
    """Load topic premises from a text file (one per line)."""
    path = Path(path) if path else DEFAULT_TOPICS_PATH
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
