# Experiment: Emotion probes on SWE-bench coding trajectories

Do coding models express more frustration, anger, or overwhelm — as measured by
emotion-direction probes on residual-stream activations — during SWE-bench tasks
they find harder or fail to solve, compared to ones they solve easily?

This experiment is a **consumer** of the `emotion-probes` package at the repo
root. It does not modify that package. It:

1. Builds emotion probes on `agentica-org/DeepSWE-Preview` using the existing
   `emotion-probes` CLI with a coding-focused emotion list.
2. Downloads DeepSWE's public SWE-bench Verified trajectories from the
   `swe-bench-submissions` S3 bucket.
3. Replays each trajectory through the local model, capturing per-token
   residual activations at the chosen probe layer and projecting onto the
   emotion directions.
4. Tests whether probe scores differ between passing vs failing tasks and easy
   vs hard tasks.

## Layout

```
config/         deepswe.yaml + emotions_coding.txt
src/            swebench_coding_emotions Python package
scripts/        5 thin shell wrappers (one per phase) + run_all.sh
tests/          unit tests
output/         gitignored; all artefacts land here
```

## Install

From the repo root:

```
pip install -e .
pip install -e experiment/swebench_coding_emotions
```

The optional-dependency group `quantize` on the root package is needed for
4-bit loading:

```
pip install -e '.[quantize]'
```

## Run

Each script `cd`s into this folder so relative paths resolve correctly.

```
bash experiment/swebench_coding_emotions/scripts/run_all.sh
```

Or phase by phase:

```
bash scripts/01_build_probes.sh        # ~30h on 1xH100 at 4-bit
bash scripts/01b_augment_neutrals.sh   # refit PCA against trajectory-shaped neutrals
bash scripts/02_fetch_trajectories.sh
bash scripts/03_replay.sh --variant augmented
bash scripts/04_analyze.sh
bash scripts/05_visualize.sh
```

## Outputs

- `output/probes/` — emotion direction vectors, global mean, cosine heatmap.
- `output/swebench/raw/` — cached trajectories + pass/fail reports.
- `output/swebench/replay/{variant}/` — per-task probe scores (.safetensors), one
  subdirectory per probe variant (`denoised`, `augmented`, `raw`). The `replay`,
  `analyze`, and `viz` subcommands all take `--variant` and read/write the matching
  subtree, so multiple variants can coexist without clobbering each other.
- `output/swebench/analysis/{variant}/` — per_task.csv, stats.csv, summary.json.
  `stats.csv` includes `metric=valence` rows (the single-axis distress composite,
  one test per section) and `summary.json` carries `valence_pass_vs_fail`.
- `output/swebench/figures/{variant}/` — timelines, box plots, valence box plot,
  heatmap, decile curves.

## Decisions

- **Model**: `agentica-org/DeepSWE-Preview` (Qwen3-32B, 64 layers, 5120 hidden).
- **Probe layer**: 42 (≈ 2/3 through the stack).
- **Emotions**: 15 coding-focused concepts (see `config/emotions_coding.txt`).
- **Sample**: 24 tasks stratified 6 each across {pass, fail} × {easy, hard}.
- **Harness**: teacher-forced replay of DeepSWE's public trajectories — no
  fresh generation. Chat template is **not** re-applied; the trajectory text
  is the string the agent originally saw.

## Confound-mitigation choices specific to this experiment

The upstream `emotion-probes` package builds emotion directions on plain
prose stories and PCA-denoises against plain `Person:/AI:` neutral
dialogues. SWE-bench transcripts are not plain prose: they contain
chat-template markers, tool-call envelopes, code blocks, unified diffs, and
`<think>` blocks. Two extra steps mitigate the resulting confounds without
changing the emotion stimuli themselves:

1. **Augmented neutral set** (`scripts/01b_augment_neutrals.sh`,
   `src/swebench_coding_emotions/augment.py`). Generates a small
   experiment-local set of *emotionally-flat* texts that exhibit each
   trajectory-time structural surface, extracts their activations at the
   probe layer, unions them with the upstream neutral activations, and
   refits the PCA denoising. Output:
   `output/probes/vectors/emotion_vectors_augmented_layer_{L}.safetensors`.
   Loaded by `replay --variant augmented`. The augmented stimuli are kept
   topic-generic so the PCA absorbs the structural directions without
   leaking topic semantics into the denoising basis.

2. **Token-axis filtering at aggregation time** (`analysis.aggregate`).
   Each per-task summary drops:
   - The first `--token-offset` positions (default 50, matches the
     upstream extract-activations offset; the residual stream there is
     dominated by template/system-prompt boilerplate the probe never saw).
   - Tokenizer special-token positions (`<|im_start|>`, `</think>`, BOS,
     etc.). The `<think>...</think>` content *between* the markers is kept
     — only the marker positions themselves are dropped. Override with
     `analyze --keep-special-tokens` to disable.

   The `auc` (mean × T) metric was removed: within a task it is perfectly
   correlated with `mean`, and across tasks it is dominated by trajectory
   length, which correlates with pass/fail and would confound the
   group-comparison tests. `mean`, `max`, and `p90` remain.

   Visualization (`viz.py`) intentionally applies only the *section* mask,
   not this aggregation mask, so plots show the underlying per-token series
   (useful for spotting probe spikes at template markers); statistics are
   computed on the masked tokens.

   `run_full_analysis` records any `(instance, section)` cells that ended
   up empty after filtering, both as a `WARNING` log line and under
   `summary.json["empty_section_cells"]`. This protects against
   silently-unequal sample sizes per emotion in the Mann-Whitney / OLS
   tests.

3. **Span-based section markers** (`replay._find_section_spans`,
   `analysis._section_mask`). The R2E-agent trajectories are multi-turn
   (`Thought:` → `<think>…</think>` → `Action:` → `<function=…>` →
   `Observation:`, ×dozens) and the model routinely leaves `</think>`
   unclosed (one task: 52 opens / 11 closes; two tasks never close it). A
   single "last `</think>`" marker therefore stretches `thinking` over
   almost the whole trajectory, and *how much* it covers correlates with
   pass/fail — a confound. The transcripts also contain no final
   `diff --git`; the model edits via `file_editor` tool calls. So sections
   are now defined per span: `thinking` = union of per-turn `Thought:` →
   `Action:` reasoning spans, `patch` = union of file-editing `file_editor`
   calls (`str_replace` / `create` / `insert`), `agent` = everything
   outside both. Spans are computed and persisted at replay time;
   pre-spans replay files fall back to the legacy single-marker logic, so
   **the new definitions take effect only after re-running `replay`**.

4. **Valence composite** (`analysis.compute_valence` / `valence_tests`,
   `metric=valence` in `stats.csv`). The 15 emotion probes are highly
   correlated — PC1 of the emotion directions is essentially a valence
   axis — so 15 BH-corrected per-emotion tests waste power and under-report
   one coherent effect. The composite z-scores each valenced emotion's
   per-task `mean` across tasks, flips the positive-valence emotions
   (signs in `analysis.VALENCE`; `curious`/`bored` excluded as
   engagement-not-distress), and averages to a single per-task distress
   score (higher = more negative affect). Each section then gets one
   pass-vs-fail test (Mann-Whitney + label-shuffle permutation), no BH
   correction needed.
