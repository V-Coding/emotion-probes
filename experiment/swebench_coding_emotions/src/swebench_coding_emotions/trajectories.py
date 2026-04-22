"""Fetch, parse, and stratify DeepSWE-Preview SWE-bench Verified trajectories.

The submission at `s3://swe-bench-submissions/verified/20250629_deepswerl_r2eagent/`
is mirrored over HTTPS at `https://swe-bench-submissions.s3.amazonaws.com/...`.
For every instance listed in `results/results.json` we download:

- `trajs/<instance_id>.txt`          — full agent transcript (already templated)
- `logs/<instance_id>/report.json`   — per-task resolved=true/false

Difficulty labels come from the upstream `princeton-nlp/SWE-bench_Verified`
HuggingFace dataset, joined on `instance_id`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com/verified"
_DEFAULT_SUBMISSION = "20250629_deepswerl_r2eagent"

EASY_LABELS = {"<15 min fix", "15 min - 1 hour"}
HARD_LABELS = {"1-4 hours", ">4 hours"}


@dataclass
class Trajectory:
    instance_id: str
    prompt: str
    full_text: str
    patch: str
    resolved: bool
    difficulty: str

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# HTTP fetch with simple on-disk cache
# ---------------------------------------------------------------------------

def _http_get(url: str, cache_path: Path, timeout: int = 60) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    return text


def _results_url(submission: str) -> str:
    return f"{_S3_BASE}/{submission}/results/results.json"


def _traj_url(submission: str, instance_id: str) -> str:
    return f"{_S3_BASE}/{submission}/trajs/{instance_id}.txt"


def _report_url(submission: str, instance_id: str) -> str:
    return f"{_S3_BASE}/{submission}/logs/{instance_id}/report.json"


# ---------------------------------------------------------------------------
# Parsing a raw transcript .txt
# ---------------------------------------------------------------------------

_PROMPT_SPLIT_RE = re.compile(r"<\|im_start\|>assistant\n", re.MULTILINE)
_PATCH_RE = re.compile(r"(diff --git [\s\S]+?)(?=\n(?:<\|im_end\|>|$))", re.MULTILINE)


def _split_prompt(full_text: str) -> str:
    """Return the portion of the transcript before the first assistant turn.

    The R2E-agent template uses Qwen3 chat markers (`<|im_start|>assistant`).
    If none is found, fall back to the whole string — rare, but safer than
    crashing.
    """
    m = _PROMPT_SPLIT_RE.search(full_text)
    if m is None:
        return full_text
    return full_text[: m.start()]


def _extract_patch(full_text: str) -> str:
    """Pull the final unified-diff block out of the transcript, or empty."""
    matches = list(_PATCH_RE.finditer(full_text))
    if not matches:
        return ""
    return matches[-1].group(1)


# ---------------------------------------------------------------------------
# Difficulty join
# ---------------------------------------------------------------------------

def load_difficulty_map() -> dict[str, str]:
    """Load {instance_id: difficulty} from princeton-nlp/SWE-bench_Verified."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {row["instance_id"]: row["difficulty"] for row in ds}


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_all(
    cache_dir: Path,
    submission: str = _DEFAULT_SUBMISSION,
    limit: int | None = None,
) -> list[Trajectory]:
    """Download (or load cached) all trajectories for a submission.

    Caches to `<cache_dir>/trajs/<id>.txt`, `<cache_dir>/reports/<id>.json`,
    and `<cache_dir>/results.json`. Skips instances whose report is missing
    or cannot be parsed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results_raw = _http_get(_results_url(submission), cache_dir / "results.json")
    results = json.loads(results_raw)
    instance_ids: list[str] = results.get("submitted_instances") or list(results.get("resolved_ids", []))
    if not instance_ids:
        # Fallback: some submissions store a flat list under "instance_ids"
        instance_ids = list(results.get("instance_ids", []))
    if not instance_ids:
        raise RuntimeError(
            f"Could not find instance list in results.json for {submission}. "
            f"Keys present: {sorted(results.keys())}"
        )

    if limit is not None:
        instance_ids = instance_ids[:limit]

    difficulty = load_difficulty_map()

    trajs: list[Trajectory] = []
    for iid in tqdm(instance_ids, desc=f"fetch[{submission}]"):
        try:
            text = _http_get(_traj_url(submission, iid), cache_dir / "trajs" / f"{iid}.txt")
            report_raw = _http_get(_report_url(submission, iid), cache_dir / "reports" / f"{iid}.json")
        except requests.HTTPError as e:
            logger.warning("Skipping %s: %s", iid, e)
            continue

        try:
            report = json.loads(report_raw)
        except json.JSONDecodeError:
            logger.warning("Skipping %s: malformed report.json", iid)
            continue

        # Some submissions wrap the report as {<instance_id>: {...}}; handle both.
        if iid in report and isinstance(report[iid], dict):
            report = report[iid]
        resolved = bool(report.get("resolved", False))

        trajs.append(
            Trajectory(
                instance_id=iid,
                prompt=_split_prompt(text),
                full_text=text,
                patch=_extract_patch(text),
                resolved=resolved,
                difficulty=difficulty.get(iid, "unknown"),
            )
        )

    logger.info("Fetched %d / %d trajectories", len(trajs), len(instance_ids))
    return trajs


# ---------------------------------------------------------------------------
# Stratified sampling: 6 per cell of {pass,fail} x {easy,hard} = 24
# ---------------------------------------------------------------------------

def stratified_sample(
    trajs: list[Trajectory],
    n: int = 24,
    seed: int = 42,
) -> list[Trajectory]:
    """Pick ``n/4`` trajectories from each of {pass,fail} x {easy,hard}.

    Unknown-difficulty tasks are dropped. If a cell is short, all of it is
    taken and a warning logged.
    """
    if n % 4 != 0:
        raise ValueError(f"n must be divisible by 4 (got {n})")
    per_cell = n // 4
    rng = np.random.default_rng(seed)

    cells: dict[tuple[bool, str], list[Trajectory]] = {
        (True, "easy"): [],
        (True, "hard"): [],
        (False, "easy"): [],
        (False, "hard"): [],
    }
    for t in trajs:
        if t.difficulty in EASY_LABELS:
            bucket = "easy"
        elif t.difficulty in HARD_LABELS:
            bucket = "hard"
        else:
            continue
        cells[(t.resolved, bucket)].append(t)

    chosen: list[Trajectory] = []
    for key, candidates in cells.items():
        if len(candidates) < per_cell:
            logger.warning(
                "Cell %s has only %d trajectories (wanted %d)", key, len(candidates), per_cell
            )
            chosen.extend(candidates)
            continue
        idx = rng.choice(len(candidates), size=per_cell, replace=False)
        chosen.extend(candidates[i] for i in idx)

    logger.info(
        "Stratified sample: pass-easy=%d pass-hard=%d fail-easy=%d fail-hard=%d",
        sum(1 for t in chosen if t.resolved and t.difficulty in EASY_LABELS),
        sum(1 for t in chosen if t.resolved and t.difficulty in HARD_LABELS),
        sum(1 for t in chosen if not t.resolved and t.difficulty in EASY_LABELS),
        sum(1 for t in chosen if not t.resolved and t.difficulty in HARD_LABELS),
    )
    return chosen


def save_sample(trajs: list[Trajectory], path: Path) -> None:
    """Persist a stratified sample as JSON (one metadata record per task)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "instance_id": t.instance_id,
            "resolved": t.resolved,
            "difficulty": t.difficulty,
            "n_chars_full": len(t.full_text),
            "n_chars_patch": len(t.patch),
        }
        for t in trajs
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sample_ids(path: Path) -> list[str]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r["instance_id"] for r in records]
