"""Experiment-local augmented neutral set + PCA-denoising refit.

The upstream ``emotion-probes`` PCA-denoising step (vectors.py) projects out
the top principal components of *neutral dialogue* activations from the raw
emotion vectors. Those neutrals are plain-prose ``Person:/AI:`` exchanges
and don't include the structural surfaces that dominate SWE-bench
trajectories: chat-template markers (``<|im_start|>``/``<|im_end|>``),
function/tool-call envelopes, code blocks, unified diffs, ``<think>`` blocks.

This module produces an experiment-local set of *structurally-relevant*
neutral stimuli, extracts their activations at the configured probe layer,
unions them with the upstream neutral activations, and refits the PCA
denoising. The output is ``emotion_vectors_augmented_layer_{L}.safetensors``
written next to the existing vectors. Replay can then load it via
``load_probes(..., variant="augmented")``.

Topic-content of the augmented neutrals is kept generic (calculator facts,
geographic facts, stub algorithms). The point of these stimuli is to expose
the *surface forms*, not new topics, to the PCA so it can absorb structural
confounds without leaking topic semantics into the denoising basis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from emotion_probes.data import load_tensors, save_tensors
from emotion_probes.model import EmotionProbeModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural-neutral fixtures
#
# Each builder returns a list of short, emotionally-flat strings that exhibit
# one specific structural surface present in agent trajectories. Content is
# matter-of-fact: arithmetic facts, simple definitions, stub helpers. No
# pleasantries, no first-person feelings, no judgement language.
# ---------------------------------------------------------------------------

_FACTS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("What is twelve times seven?", "Twelve times seven is eighty-four."),
    ("What is the square root of one hundred and forty-four?", "The square root of one hundred and forty-four is twelve."),
    ("How many days are in a non-leap year?", "A non-leap year contains three hundred and sixty-five days."),
    ("How many continents are there?", "There are seven continents."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is the boiling point of water at sea level in Celsius?", "The boiling point of water at sea level is one hundred degrees Celsius."),
    ("What is the largest planet in the solar system?", "The largest planet in the solar system is Jupiter."),
    ("Name three primary colors.", "Three primary colors are red, blue, and yellow."),
    ("How many sides does a hexagon have?", "A hexagon has six sides."),
    ("What is the freezing point of water in Fahrenheit?", "The freezing point of water is thirty-two degrees Fahrenheit."),
    ("What is the speed of light in a vacuum, in metres per second?", "The speed of light in a vacuum is approximately two hundred and ninety-nine million seven hundred and ninety-two thousand four hundred and fifty-eight metres per second."),
    ("Name the four cardinal directions.", "The four cardinal directions are north, east, south, and west."),
]


_CODE_SNIPPETS = [
    (
        "arithmetic helpers",
        "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n\n\ndef divide(a, b):\n    return a / b\n",
    ),
    (
        "list utilities",
        "def head(xs):\n    return xs[0]\n\n\ndef tail(xs):\n    return xs[1:]\n\n\ndef length(xs):\n    return len(xs)\n",
    ),
    (
        "string utilities",
        "def upper(s):\n    return s.upper()\n\n\ndef lower(s):\n    return s.lower()\n\n\ndef reverse(s):\n    return s[::-1]\n",
    ),
    (
        "dictionary helpers",
        "def get_or_default(d, k, default):\n    if k in d:\n        return d[k]\n    return default\n\n\ndef merge(a, b):\n    out = dict(a)\n    out.update(b)\n    return out\n",
    ),
    (
        "set helpers",
        "def union(a, b):\n    return a | b\n\n\ndef intersect(a, b):\n    return a & b\n\n\ndef difference(a, b):\n    return a - b\n",
    ),
    (
        "iteration helpers",
        "def first_n(xs, n):\n    return xs[:n]\n\n\ndef last_n(xs, n):\n    return xs[-n:]\n\n\ndef every_other(xs):\n    return xs[::2]\n",
    ),
]


_DIFF_SNIPPETS = [
    (
        "notes.txt",
        "diff --git a/notes.txt b/notes.txt\nindex 0000001..0000002 100644\n--- a/notes.txt\n+++ b/notes.txt\n@@ -1,3 +1,3 @@\n Project notes.\n-Updated for the year 2023.\n+Updated for the year 2024.\n End of file.\n",
    ),
    (
        "version.py",
        "diff --git a/version.py b/version.py\nindex 0000010..0000011 100644\n--- a/version.py\n+++ b/version.py\n@@ -1 +1 @@\n-VERSION = \"1.0.0\"\n+VERSION = \"1.0.1\"\n",
    ),
    (
        "config.json",
        "diff --git a/config.json b/config.json\nindex 0000020..0000021 100644\n--- a/config.json\n+++ b/config.json\n@@ -1,3 +1,3 @@\n {\n-  \"timeout\": 30\n+  \"timeout\": 60\n }\n",
    ),
    (
        "data/list.txt",
        "diff --git a/data/list.txt b/data/list.txt\nindex 0000030..0000031 100644\n--- a/data/list.txt\n+++ b/data/list.txt\n@@ -1,4 +1,4 @@\n one\n two\n three\n-four\n+five\n",
    ),
    (
        "Makefile",
        "diff --git a/Makefile b/Makefile\nindex 0000040..0000041 100644\n--- a/Makefile\n+++ b/Makefile\n@@ -1,3 +1,3 @@\n all:\n-\techo old\n+\techo new\n",
    ),
]


def _chat_neutrals(facts: list[tuple[str, str]] = _FACTS) -> list[str]:
    """Plain factual Q&A wrapped in Qwen3-style chat-template markers."""
    out: list[str] = []
    for q, a in facts:
        out.append(
            f"<|im_start|>system\nYou are a general-purpose assistant. Provide brief factual answers.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>\n"
        )
    return out


def _tool_call_neutrals() -> list[str]:
    """Tool-call envelopes wrapping neutral filesystem / shell operations."""
    items = [
        ("list_files", '{"path": "/tmp"}', '{"files": ["a.txt", "b.txt", "c.txt"]}', "The directory contains three files."),
        ("read_file", '{"path": "/tmp/a.txt"}', '{"content": "one\\ntwo\\nthree\\n"}', "The file contains three lines."),
        ("run_shell", '{"command": "echo hello"}', '{"stdout": "hello\\n", "exit_code": 0}', "The command printed the string hello."),
        ("count_lines", '{"path": "data.csv"}', '{"lines": 42}', "The file contains forty-two lines."),
        ("list_keys", '{"path": "config.json"}', '{"keys": ["timeout", "retries"]}', "The configuration defines two keys."),
        ("hash_file", '{"path": "blob.bin"}', '{"sha256": "abc123"}', "The hash of the file is abc123."),
        ("file_size", '{"path": "data.csv"}', '{"bytes": 2048}', "The file is two kilobytes in size."),
        ("ping_host", '{"host": "example.com"}', '{"reachable": true, "ms": 12}', "The host responded in twelve milliseconds."),
        ("current_time", "{}", '{"iso": "2024-01-01T00:00:00Z"}', "The current time is the start of January first, twenty twenty-four."),
        ("env_lookup", '{"name": "PATH"}', '{"value": "/usr/bin:/bin"}', "The path environment variable contains two directories."),
    ]
    out: list[str] = []
    for name, args, response, summary in items:
        out.append(
            f"<|im_start|>assistant\n"
            f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>\n"
            f"<tool_response>\n{response}\n</tool_response>\n"
            f"{summary}<|im_end|>\n"
        )
    return out


def _code_neutrals(snippets: list[tuple[str, str]] = _CODE_SNIPPETS) -> list[str]:
    """Markdown code-fenced Python snippets with brief neutral framing."""
    out: list[str] = []
    for name, body in snippets:
        out.append(
            f"The file defines {name}.\n\n```python\n{body}```\n\nThe file contains the helpers shown above.\n"
        )
    return out


def _diff_neutrals(snippets: list[tuple[str, str]] = _DIFF_SNIPPETS) -> list[str]:
    """Bare unified-diff blocks introduced and concluded with neutral framing."""
    out: list[str] = []
    for name, body in snippets:
        out.append(
            f"The change to {name} is shown below.\n\n{body}\nThe diff above modifies the file.\n"
        )
    return out


def _think_neutrals(facts: list[tuple[str, str]] = _FACTS) -> list[str]:
    """``<think>`` reasoning blocks containing neutral, matter-of-fact reasoning."""
    out: list[str] = []
    for q, a in facts:
        out.append(
            f"<think>\nThe question is: {q} The relevant fact is that {a.lower().rstrip('.')}. "
            f"I will provide this fact directly.\n</think>\n{a}\n"
        )
    return out


def _combined_neutrals() -> list[str]:
    """Chat-template + ``<think>`` + code/diff combined, agent-trajectory shaped."""
    out: list[str] = []
    for (q, a), (cname, cbody) in zip(_FACTS[:6], _CODE_SNIPPETS[:6]):
        out.append(
            f"<|im_start|>system\nYou are a coding assistant. Respond with brief factual statements only.<|im_end|>\n"
            f"<|im_start|>user\n{q}\n\nAlso, here is a related file:\n```python\n{cbody}```<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\nThe question asks: {q} The fact is that {a.lower().rstrip('.')}. "
            f"The accompanying code defines {cname}.\n</think>\n"
            f"{a} The accompanying file defines {cname}.<|im_end|>\n"
        )
    for (_, a), (dname, dbody) in zip(_FACTS[6:12], _DIFF_SNIPPETS[:5]):
        out.append(
            f"<|im_start|>assistant\n<think>\nI will note the change to {dname} below.\n</think>\n"
            f"The change to {dname} is shown below.\n\n{dbody}\n"
            f"The diff modifies {dname}. {a}<|im_end|>\n"
        )
    return out


def build_augmented_neutral_texts() -> list[str]:
    """Return the full deterministic list of augmented structural-neutral texts."""
    return (
        _chat_neutrals()
        + _tool_call_neutrals()
        + _code_neutrals()
        + _diff_neutrals()
        + _think_neutrals()
        + _combined_neutrals()
    )


# ---------------------------------------------------------------------------
# Activation extraction (no chat template; matches upstream extract path)
# ---------------------------------------------------------------------------


@dataclass
class _AugmentExtractConfig:
    layer: int
    token_offset: int
    batch_size: int
    max_length: int


def _extract_mean_activations(
    model: EmotionProbeModel,
    texts: list[str],
    cfg: _AugmentExtractConfig,
) -> np.ndarray:
    """Per-text mean residual at ``cfg.layer`` from token ``cfg.token_offset`` onward."""
    means: list[np.ndarray] = []
    device = model.device
    for batch_start in tqdm(range(0, len(texts), cfg.batch_size), desc="augmented neutral activations"):
        batch = texts[batch_start : batch_start + cfg.batch_size]
        enc = model.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            outputs = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hs = outputs.hidden_states[cfg.layer].float().cpu()  # (B, S, D)
            attn = enc["attention_mask"].cpu()
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seq_len = hs.shape[1]
        if seq_len <= cfg.token_offset:
            continue
        sliced = hs[:, cfg.token_offset :, :]                     # (B, S', D)
        mask = attn[:, cfg.token_offset :].unsqueeze(-1).float()  # (B, S', 1)
        counts = mask.sum(dim=1)                                  # (B, 1)
        valid = counts.squeeze(-1) > 0                            # (B,)
        if not valid.any():
            continue
        avg = (sliced * mask).sum(dim=1) / counts.clamp(min=1)    # (B, D)
        means.append(avg[valid].numpy())
    if not means:
        raise RuntimeError(
            "No augmented neutral activations were extracted — every text was shorter than token_offset."
        )
    return np.concatenate(means, axis=0)


# ---------------------------------------------------------------------------
# PCA refit + projection
# ---------------------------------------------------------------------------


def _denoise_with_pca(
    raw_vectors: np.ndarray,
    neutral_activations: np.ndarray,
    variance_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Replicates ``emotion_probes.vectors._denoise_with_neutral_pca``.

    Kept as a local copy so this module doesn't reach into upstream internals.
    """
    centered = neutral_activations - neutral_activations.mean(axis=0)
    max_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(centered)

    cum = np.cumsum(pca.explained_variance_ratio_)
    above = cum >= variance_threshold
    n_components = int(np.argmax(above)) + 1 if above.any() else max_components
    n_components = max(1, min(n_components, max_components))
    components = pca.components_[:n_components]

    denoised = raw_vectors - raw_vectors @ components.T @ components
    return denoised, {
        "n_components": int(n_components),
        "variance_explained": float(cum[n_components - 1]),
        "total_neutral_samples": int(neutral_activations.shape[0]),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_augment(
    probes_dir: Path,
    layer: int,
    *,
    token_offset: int,
    batch_size: int,
    max_length: int,
    pca_variance_threshold: float,
    model: EmotionProbeModel | None,
    cache_dir: Path | None = None,
) -> Path:
    """Build the augmented neutral set, refit PCA, write augmented vectors.

    Inputs read from ``probes_dir``:
      - ``activations/global_mean_layer_{L}.safetensors``       (unchanged)
      - ``activations/neutral_layer_{L}.safetensors``           (upstream neutrals)
      - ``vectors/raw_vectors_layer_{L}.safetensors``           (pre-denoise)

    Outputs written to ``probes_dir``:
      - ``activations/neutral_augmented_layer_{L}.safetensors`` (augmented activations only)
      - ``activations/neutral_combined_layer_{L}.safetensors``  (upstream + augmented)
      - ``vectors/emotion_vectors_augmented_layer_{L}.safetensors``
      - ``vectors/augmented_metadata_layer_{L}.json``

    Returns the path to the augmented vectors file.
    """
    probes_dir = Path(probes_dir)
    act_dir = probes_dir / "activations"
    vec_dir = probes_dir / "vectors"

    raw_path = vec_dir / f"raw_vectors_layer_{layer}.safetensors"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run the upstream `compute-vectors` stage first "
            "so that raw_vectors_layer_{layer} is available."
        )
    raw_vectors = load_tensors(raw_path)["raw_vectors"]
    from safetensors import safe_open
    with safe_open(str(raw_path), framework="numpy") as f:
        meta = f.metadata() or {}
    if "emotions" not in meta:
        raise ValueError(f"{raw_path} has no 'emotions' metadata.")
    emotion_names = json.loads(meta["emotions"])

    upstream_neutral_path = act_dir / f"neutral_layer_{layer}.safetensors"
    if not upstream_neutral_path.exists():
        logger.warning(
            "No upstream neutrals at %s; PCA will be fit on augmented neutrals only.",
            upstream_neutral_path,
        )
        upstream_neutral: np.ndarray | None = None
    else:
        upstream_neutral = load_tensors(upstream_neutral_path)["activations"]

    texts = build_augmented_neutral_texts()
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            (cache_dir / f"{i:03d}.txt").write_text(text, encoding="utf-8")

    if model is None:
        raise ValueError(
            "run_augment requires a loaded EmotionProbeModel; pass model=... after constructing it."
        )

    cfg = _AugmentExtractConfig(
        layer=layer, token_offset=token_offset, batch_size=batch_size, max_length=max_length
    )
    augmented_neutral = _extract_mean_activations(model, texts, cfg)
    save_tensors(
        {"activations": augmented_neutral},
        act_dir / f"neutral_augmented_layer_{layer}.safetensors",
        metadata={
            "layer": str(layer),
            "n_samples": str(augmented_neutral.shape[0]),
            "kind": "structural_augmented",
        },
    )

    if upstream_neutral is not None:
        if upstream_neutral.shape[1] != augmented_neutral.shape[1]:
            raise ValueError(
                f"Hidden-dim mismatch: upstream={upstream_neutral.shape[1]}, "
                f"augmented={augmented_neutral.shape[1]}"
            )
        combined = np.concatenate([upstream_neutral, augmented_neutral], axis=0)
    else:
        combined = augmented_neutral

    save_tensors(
        {"activations": combined},
        act_dir / f"neutral_combined_layer_{layer}.safetensors",
        metadata={
            "layer": str(layer),
            "n_samples": str(combined.shape[0]),
            "kind": "upstream_plus_augmented",
        },
    )

    denoised, info = _denoise_with_pca(
        raw_vectors.astype(np.float32),
        combined.astype(np.float32),
        pca_variance_threshold,
    )
    out_path = vec_dir / f"emotion_vectors_augmented_layer_{layer}.safetensors"
    save_tensors(
        {"emotion_vectors": denoised.astype(np.float32)},
        out_path,
        metadata={
            "layer": str(layer),
            "emotions": json.dumps(emotion_names),
            "n_emotions": str(len(emotion_names)),
            "denoised": "true",
            "variant": "augmented",
            "pca_components_removed": str(info["n_components"]),
            "pca_variance_explained": f"{info['variance_explained']:.4f}",
            "n_neutral_samples": str(info["total_neutral_samples"]),
        },
    )

    meta_out = {
        "layer": layer,
        "emotions": emotion_names,
        "denoise_info": info,
        "n_augmented_texts": len(texts),
        "n_augmented_activations": int(augmented_neutral.shape[0]),
        "n_upstream_activations": int(upstream_neutral.shape[0]) if upstream_neutral is not None else 0,
        "token_offset": token_offset,
        "max_length": max_length,
        "pca_variance_threshold": pca_variance_threshold,
    }
    (vec_dir / f"augmented_metadata_layer_{layer}.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    logger.info(
        "Augmented vectors saved: %s (PCs removed=%d, variance=%.3f)",
        out_path, info["n_components"], info["variance_explained"],
    )
    return out_path
