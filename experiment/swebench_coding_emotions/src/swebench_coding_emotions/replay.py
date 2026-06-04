"""Teacher-forced activation capture + emotion-probe projection.

We load a trajectory (already R2E-agent-templated), tokenize it with offset
mapping, and run sliding-window forward passes. For each window we extract
one hidden-state layer, mean-center against the probe-build global mean,
and project onto the 15 emotion direction vectors.

Notes:
- We bypass ``EmotionProbeModel.get_hidden_states`` because it truncates at
  4096 tokens — SWE-bench trajectories routinely exceed 30k tokens.
- We do not re-apply ``tokenizer.apply_chat_template``. The trajectory text
  from S3 is already the exact string the agent saw.
- We persist fp16 scores to keep per-task files small (~1 MB for 25k tokens).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from emotion_probes.data import load_tensors, load_vectors_with_metadata, save_tensors
from emotion_probes.model import EmotionProbeModel

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
_PATCH_RE = re.compile(r"diff --git ")

# Span-based markers for multi-turn R2E-agent transcripts. Each turn looks like
#   Thought:\n\n<think>...reasoning...</think>\nAction:\n\n<function=NAME>...</function>\nObservation:...
# but the model frequently leaves </think> unclosed (e.g. 52 opens / 11 closes
# in one task; some tasks never close it), so the single "last </think>" marker
# above mislabels almost the whole trajectory. We instead bound each turn's
# reasoning by the reliable Thought: → Action: delimiters.
_ACTION_RE = re.compile(r"\nAction:")
_THOUGHT_MARK = "Thought:"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# A patch span is a file_editor tool call whose command actually edits a file
# (the model authoring the fix), as opposed to read-only view/search calls.
_FILE_EDIT_RE = re.compile(r"<function=file_editor>([\s\S]*?)</function>")
_EDIT_COMMAND_RE = re.compile(r"<parameter=command>\s*([a-zA-Z_]+)")
_EDIT_COMMANDS = frozenset({"str_replace", "create", "insert"})


@dataclass
class ReplayResult:
    instance_id: str
    scores: np.ndarray          # (T, 15) fp16, dot product of centered resid with emotion dirs
    cosine: np.ndarray          # (T, 15) fp16, cosine similarity version
    token_ids: np.ndarray       # (T,) int32
    char_starts: np.ndarray     # (T,) int32
    char_ends: np.ndarray       # (T,) int32
    layer: int
    emotions: list[str]
    resolved: bool
    difficulty: str
    thinking_end_char: int      # -1 if no </think> found  (legacy single-marker)
    patch_start_char: int       # -1 if no diff found      (legacy single-marker)
    model_fingerprint: str
    special_token_ids: list[int]  # tokenizer.all_special_ids at replay time
    probe_variant: str            # "denoised" | "augmented" | "raw"
    # Span-based section markers (char ranges). ``None`` marks a pre-spans replay
    # file so the analyzer falls back to the legacy single-marker logic above.
    thinking_spans: list[tuple[int, int]] | None = None
    patch_spans: list[tuple[int, int]] | None = None


# ---------------------------------------------------------------------------
# Fingerprint — lock probe-build config to replay config
# ---------------------------------------------------------------------------

def compute_fingerprint(model_name: str, torch_dtype: str, quantize: str | None, layer: int) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}|{torch_dtype}|{quantize}|{layer}".encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Probe asset loading
# ---------------------------------------------------------------------------

VARIANT_TO_FILENAME = {
    "denoised": "emotion_vectors",
    "augmented": "emotion_vectors_augmented",
    "raw": "raw_vectors",
}


def load_probes(
    probes_output_dir: Path,
    layer: int,
    variant: str = "denoised",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load emotion vectors + emotion names + global mean for a given layer.

    ``probes_output_dir`` is the same directory used as ``output_dir`` during
    the probe-build phase (i.e. ``experiment/.../output/probes``).

    ``variant`` selects which vectors file to load:
      - ``"denoised"``: upstream PCA-denoised vectors (default)
      - ``"augmented"``: PCA refit including this experiment's structural neutrals
      - ``"raw"``: pre-denoising emotion_means - global_mean
    """
    if variant not in VARIANT_TO_FILENAME:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {sorted(VARIANT_TO_FILENAME)}.")
    vectors_dir = probes_output_dir / "vectors"
    act_dir = probes_output_dir / "activations"

    name = VARIANT_TO_FILENAME[variant]
    vec_path = vectors_dir / f"{name}_layer_{layer}.safetensors"
    if not vec_path.exists():
        raise FileNotFoundError(
            f"Missing vectors file {vec_path} for variant={variant!r}. "
            "If variant='augmented', run `swebench-emotions augment` first."
        )

    if variant == "raw":
        tensors = load_tensors(vec_path)
        vectors = tensors["raw_vectors"]
        from safetensors import safe_open
        with safe_open(str(vec_path), framework="numpy") as f:
            meta = f.metadata() or {}
        emotions = json.loads(meta["emotions"])
        if "layer" in meta and int(meta["layer"]) != layer:
            raise ValueError(
                f"Layer mismatch in {vec_path}: file says {meta['layer']}, expected {layer}"
            )
    else:
        vectors, emotions, stored_layer = load_vectors_with_metadata(vec_path)
        if stored_layer != layer:
            raise ValueError(f"Layer mismatch: file says {stored_layer}, expected {layer}")

    global_mean = load_tensors(act_dir / f"global_mean_layer_{layer}.safetensors")["global_mean"]
    return vectors.astype(np.float32), emotions, global_mean.astype(np.float32)


# ---------------------------------------------------------------------------
# Section tagging
# ---------------------------------------------------------------------------

def _find_section_markers(text: str) -> tuple[int, int]:
    """Return (thinking_end_char, patch_start_char), -1 if absent.

    Legacy single-marker locator, kept for backward-compatibility metadata.
    Superseded by :func:`_find_section_spans` for the actual section masks.
    """
    think_end = -1
    think_matches = list(_THINK_RE.finditer(text))
    if think_matches:
        think_end = think_matches[-1].end()
    patch_start = -1
    pm = _PATCH_RE.search(text)
    if pm is not None:
        patch_start = pm.start()
    return think_end, patch_start


def _find_section_spans(
    text: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (thinking_spans, patch_spans) as lists of ``[start, end)`` char ranges.

    R2E-agent transcripts interleave many turns of
    ``Thought:\\n<think>...</think>\\nAction:\\n<function=...>...</function>\\nObservation:...``.
    Two facts make the legacy single-marker split unusable here:

    * The model often forgets to close ``</think>`` (one task: 52 opens / 11
      closes; two tasks never close it). "Span up to the last ``</think>``"
      then covers ~the whole trajectory, and how much it covers correlates
      with the outcome being tested — a confound. We instead bound each turn's
      reasoning by the reliable ``Thought:`` → ``Action:`` delimiters (closing
      at ``</think>`` when the model did emit it).
    * There is no final unified diff (``diff --git`` occurs nowhere); the model
      edits via ``file_editor`` tool calls. A patch span is each ``file_editor``
      call whose command actually edits a file (``str_replace`` / ``create`` /
      ``insert``), i.e. the model authoring the fix.

    ``agent`` is then defined (in analysis) as everything outside both span sets.
    """
    thinking: list[tuple[int, int]] = []
    prev = 0
    for m in _ACTION_RE.finditer(text):
        action_start = m.start()
        seg = text[prev:action_start]
        tj = seg.rfind(_THOUGHT_MARK)  # this turn's Thought: (closest to Action:)
        if tj != -1:
            ti = seg.find(_THINK_OPEN, tj)
            start = prev + (ti + len(_THINK_OPEN) if ti != -1 else tj + len(_THOUGHT_MARK))
            ce = seg.rfind(_THINK_CLOSE)  # closed this turn? trim the tag if so
            end = prev + ce if (ce != -1 and prev + ce > start) else action_start
            if end > start:
                thinking.append((start, end))
        prev = action_start

    patch: list[tuple[int, int]] = []
    for m in _FILE_EDIT_RE.finditer(text):
        cmd = _EDIT_COMMAND_RE.search(m.group(1))
        if cmd is not None and cmd.group(1) in _EDIT_COMMANDS:
            patch.append((m.start(), m.end()))

    return thinking, patch


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_trajectory(
    model: EmotionProbeModel,
    emotion_vectors: np.ndarray,    # (K, D) float32
    global_mean: np.ndarray,        # (D,)   float32
    emotion_names: list[str],
    text: str,
    *,
    instance_id: str,
    resolved: bool,
    difficulty: str,
    layer: int,
    window_tokens: int = 8192,
    stride_tokens: int = 7168,
    model_fingerprint: str | None = None,
    probe_variant: str = "denoised",
) -> ReplayResult:
    """Run sliding-window forward passes and return per-token probe scores.

    Returns one score per token of the original (un-windowed) sequence. When
    windows overlap, scores from the later window overwrite earlier scores
    past the stride boundary (they're contextually richer).
    """
    if stride_tokens >= window_tokens:
        raise ValueError("stride_tokens must be less than window_tokens")

    # Tokenize once with offsets; window over the token axis.
    enc = model.tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = enc["input_ids"][0]               # (T,)
    offsets = enc["offset_mapping"][0].numpy()    # (T, 2)
    T = input_ids.shape[0]
    D = emotion_vectors.shape[1]
    K = emotion_vectors.shape[0]

    if T == 0:
        raise ValueError(f"Empty tokenization for {instance_id}")

    # Precompute emotion norms for cosine
    emo_t = torch.from_numpy(emotion_vectors).to(torch.float32)  # (K, D)
    emo_norms = emo_t.norm(dim=1, keepdim=True).clamp_min(1e-6)  # (K, 1)
    gm_t = torch.from_numpy(global_mean).to(torch.float32)       # (D,)

    scores = np.zeros((T, K), dtype=np.float32)
    cosine = np.zeros((T, K), dtype=np.float32)
    filled = np.zeros(T, dtype=bool)

    device = model.device

    start = 0
    while start < T:
        end = min(start + window_tokens, T)
        window_ids = input_ids[start:end].unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model.model(
                input_ids=window_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            # (1, W, D) → (W, D) fp32 on CPU; drop the rest immediately
            hidden = outputs.hidden_states[layer][0].float().cpu()
        del outputs

        centered = hidden - gm_t                      # (W, D)
        win_scores = centered @ emo_t.T               # (W, K)
        win_cos = win_scores / (centered.norm(dim=1, keepdim=True).clamp_min(1e-6) * emo_norms.T)

        # Decide which tokens in this window to write.
        # For the first window, write everything.
        # For subsequent windows, only write tokens past the stride boundary
        # (i.e. new context past the overlap region).
        if start == 0:
            write_from = start
        else:
            write_from = start + (window_tokens - stride_tokens)
        write_to = end

        # Map window-local indices to global indices.
        local_lo = write_from - start
        local_hi = write_to - start
        scores[write_from:write_to] = win_scores[local_lo:local_hi].numpy()
        cosine[write_from:write_to] = win_cos[local_lo:local_hi].numpy()
        filled[write_from:write_to] = True

        if end == T:
            break
        start += stride_tokens

    if not filled.all():
        raise RuntimeError(f"Not all tokens scored for {instance_id} (filled={filled.sum()}/{T})")

    think_end, patch_start = _find_section_markers(text)
    thinking_spans, patch_spans = _find_section_spans(text)
    fingerprint = model_fingerprint or compute_fingerprint(
        model.config.name, model.config.torch_dtype, model.config.quantize, layer
    )

    special_ids = sorted({int(i) for i in (model.tokenizer.all_special_ids or [])})

    return ReplayResult(
        instance_id=instance_id,
        scores=scores.astype(np.float16),
        cosine=cosine.astype(np.float16),
        token_ids=input_ids.numpy().astype(np.int32),
        char_starts=offsets[:, 0].astype(np.int32),
        char_ends=offsets[:, 1].astype(np.int32),
        layer=layer,
        emotions=emotion_names,
        resolved=resolved,
        difficulty=difficulty,
        thinking_end_char=think_end,
        patch_start_char=patch_start,
        model_fingerprint=fingerprint,
        special_token_ids=special_ids,
        probe_variant=probe_variant,
        thinking_spans=thinking_spans,
        patch_spans=patch_spans,
    )


# ---------------------------------------------------------------------------
# Save / load per-task replay results
# ---------------------------------------------------------------------------

def save_replay(result: ReplayResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensors(
        {
            "scores": result.scores,
            "cosine": result.cosine,
            "token_ids": result.token_ids,
            "char_starts": result.char_starts,
            "char_ends": result.char_ends,
        },
        path,
        metadata={
            "instance_id": result.instance_id,
            "layer": str(result.layer),
            "emotions": json.dumps(result.emotions),
            "resolved": "1" if result.resolved else "0",
            "difficulty": result.difficulty,
            "thinking_end_char": str(result.thinking_end_char),
            "patch_start_char": str(result.patch_start_char),
            "model_fingerprint": result.model_fingerprint,
            "special_token_ids": json.dumps(result.special_token_ids),
            "probe_variant": result.probe_variant,
            "thinking_spans": json.dumps(
                [list(s) for s in (result.thinking_spans or [])]
            ),
            "patch_spans": json.dumps([list(s) for s in (result.patch_spans or [])]),
        },
    )


def read_replay_variant(path: Path) -> str | None:
    """Return the ``probe_variant`` metadata of a saved replay file, or None.

    Cheap: reads metadata only, not tensors. Used by the replay CLI to refuse
    silently overwriting a file written for a different probe variant.
    """
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
    except (OSError, ValueError):
        return None
    return meta.get("probe_variant")


def _load_spans(meta: dict, key: str) -> list[tuple[int, int]] | None:
    """Parse a JSON span list from metadata, or None if the key is absent.

    Absent (None) => pre-spans replay file => analysis uses the legacy
    single-marker logic. Present-but-empty ([]) => span logic with no spans.
    """
    raw = meta.get(key)
    if raw is None:
        return None
    return [tuple(s) for s in json.loads(raw)]


def load_replay(path: Path) -> ReplayResult:
    tensors = load_tensors(path)
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
    special_raw = meta.get("special_token_ids")
    special_ids: list[int] = json.loads(special_raw) if special_raw else []
    probe_variant = meta.get("probe_variant", "denoised")
    return ReplayResult(
        instance_id=meta["instance_id"],
        scores=tensors["scores"],
        cosine=tensors["cosine"],
        token_ids=tensors["token_ids"],
        char_starts=tensors["char_starts"],
        char_ends=tensors["char_ends"],
        layer=int(meta["layer"]),
        emotions=json.loads(meta["emotions"]),
        resolved=meta["resolved"] == "1",
        difficulty=meta["difficulty"],
        thinking_end_char=int(meta["thinking_end_char"]),
        patch_start_char=int(meta["patch_start_char"]),
        model_fingerprint=meta["model_fingerprint"],
        special_token_ids=special_ids,
        probe_variant=probe_variant,
        thinking_spans=_load_spans(meta, "thinking_spans"),
        patch_spans=_load_spans(meta, "patch_spans"),
    )
