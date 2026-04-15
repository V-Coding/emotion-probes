# EmotionProbes

Generate emotion probes for open-source language models, based on the methodology from Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2026/emotions/index.html).

Extracts internal linear representations of 171 emotion concepts from model activations, validates them, and produces analysis (cosine similarity, PCA, UMAP, logit lens).

## Installation

```bash
pip install -e .
```

For 4-bit/8-bit quantization (recommended for limited GPU memory):

```bash
pip install -e ".[quantize]"
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Quick Start

Run the full pipeline with the medium config (20 curated emotions, 100 topics, 12 stories each):

```bash
emotion-probes run-all --config config/medium.yaml
```

This produces:
- Emotion vectors in `output/vectors/`
- Cosine similarity heatmap, PCA/UMAP scatter plots in `output/figures/`
- Logit lens tables in `output/analysis/`

## Full Run

For the complete reproduction (171 emotions, 100 topics, 12 stories each = 205,200 stories):

```bash
emotion-probes run-all --config config/default.yaml
```

Story generation is the bottleneck. All stages checkpoint automatically — if interrupted, re-running the same command resumes where it left off.

## Running Individual Stages

Each pipeline stage can be run independently:

```bash
# 1. Generate emotion-labeled stories
emotion-probes generate-stories --config config/medium.yaml

# 2. Generate emotionally neutral dialogues (for denoising)
emotion-probes generate-neutral --config config/medium.yaml

# 3. Extract hidden state activations
emotion-probes extract-activations --config config/medium.yaml

# 4. Compute denoised emotion vectors
emotion-probes compute-vectors --config config/medium.yaml

# 5. Run geometric analysis (cosine sim, PCA, k-means, UMAP)
emotion-probes analyze --config config/medium.yaml

# 6. Generate plots
emotion-probes visualize --config config/medium.yaml

# 7. Project vectors through unembedding (logit lens)
emotion-probes logit-lens --config config/medium.yaml --emotions happy,sad,angry,calm
```

Stages must be run in order — each depends on the previous stage's output.

## Configuration

Two configs are provided:

| Config | Emotions | Topics | Stories/topic | Total stories | Use case |
|--------|----------|--------|---------------|---------------|----------|
| `config/medium.yaml` | 20 | 100 | 12 | 24,000 | Curated emotion subset |
| `config/default.yaml` | 171 | 100 | 12 | 205,200 | Full reproduction |

Key settings in the YAML:

```yaml
model:
  name: google/gemma-4-E4B-it   # or google/gemma-4-31B-it
  quantize: 4bit                 # null, 4bit, or 8bit
  target_layer: null             # null = auto (2/3 of model depth)

generation:
  stories_per_topic: 12
  emotions_subset: null          # null = all 171, or integer to limit
  topics_subset: null            # null = all 100, or integer to limit
```

## Supported Models

- **google/gemma-4-E4B-it** (default)
- **google/gemma-4-31B-it**

Any HuggingFace causal LM should work by changing `model.name` in the config.

## Output Structure

```
output/
  stories/{emotion}/{topic_hash}.json   # Generated stories
  neutral/{topic_hash}.json             # Neutral dialogues
  activations/                          # Per-layer mean activations
  vectors/                              # Final emotion vectors (safetensors)
  analysis/                             # Cosine sim, PCA, k-means, logit lens
  figures/                              # Plots (PNG)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## How It Works

1. **Story generation** — The model writes short stories for each of 171 emotions across 100 topics, conveying each emotion indirectly (never naming it)
2. **Activation extraction** — Stories are fed back through the model; residual stream activations are averaged per emotion (from token 50 onward)
3. **Mean subtraction** — The global mean across all emotions is subtracted, isolating emotion-specific directions
4. **PCA denoising** — Top principal components of neutral (non-emotional) text activations are projected out to remove confounds
5. **Analysis** — Cosine similarity reveals emotion clustering; PCA shows valence (PC1) and arousal (PC2) axes; logit lens shows which tokens each emotion vector upweights

Based on: [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) (Anthropic, 2026)
