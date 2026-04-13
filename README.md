<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Website][website-shield]][website-url]
[![Discord][discord-shield]][discord-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://videodb.io/">
    <img src="https://codaio.imgix.net/docs/_s5lUnUCIU/blobs/bl-RgjcFrrJjj/d3cbc44f8584ecd42f2a97d981a144dce6a66d83ddd5864f723b7808c7d1dfbc25034f2f25e1b2188e78f78f37bcb79d3c34ca937cbb08ca8b3da1526c29da9a897ab38eb39d084fd715028b7cc60eb595c68ecfa6fa0bb125ec2b09da65664a4f172c2f" alt="VideoDB" width="300">
  </a>

  <h3 align="center">Do Thought Streams Matter?</h3>

  <p align="center">
    Evaluating Reasoning in Gemini Vision-Language Models for Video Scene Understanding
    <br />
    <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>Read the Paper »</strong></a> -->
  </p>
</p>

---

## Overview

Large vision-language models like Gemini 2.5 Flash support "extended thinking" - an internal reasoning trace before producing the final answer. But does thinking more actually help? Where do gains stop? What do models actually think about?

We benchmark four Gemini 2.5 Flash and Flash Lite configurations across **100 hours of video** (93,000+ scene-level results) to answer these questions. This repo provides the complete pipeline to reproduce our evaluation on your own video data.

<!-- Read the full paper: [Do Thought Streams Matter? (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) -->

### Evaluation Metrics

- **Contentfulness** -- How much of the thought stream is actual scene content vs. meta-commentary?
- **Thought Coverage (Recall)** -- Of everything the model thought about, how much made it into the final output?
- **Output Grounding (Precision)** -- Of everything in the final output, how much was actually in the thought stream?
- **F1** -- Harmonic mean of Recall and Precision
- **Dominant Entity Analysis** -- Most prominent subject, action, and setting per scene

## Key Findings

| Variant | Contentfulness | Recall | Precision | F1 |
|---------|----------------|--------|-----------|-----|
| Flash – 128 | 0.323 | 0.853 | 0.767 | 0.830 |
| Flash – Dynamic | 0.594 | 0.953 | 0.964 | 0.957 |
| Lite – 512 | 0.520 | 0.940 | 0.948 | 0.942 |
| **Lite – 1024** | **0.582** | **0.954** | **0.966** | **0.959** |

- **Diminishing returns** -- Most quality improvement happens in the first ~300 thinking tokens, then plateaus sharply
- **Lite 1024 wins** -- Leads on every metric while using 30% fewer thought tokens than Flash Dynamic
- **Compression-step hallucination** -- Flash 128 outputs ~25% content not grounded in its own thought stream (Precision = 0.767)
- **Cross-tier similarity** -- Flash and Lite produce 88-90% similar thought streams despite being different model tiers
- **Subject specificity** -- Lower budgets default to generic labels ("person" at 15%) vs. specific identities ("chef", "streamer" at 8%) with higher budgets

> **Note on thought text visibility:** Gemini only returns thought text in the response when `includeThoughts=True` is set in `ThinkingConfig`. Without this flag, the model uses thinking tokens internally but the thought stream is not accessible. This pipeline sets it correctly for all variants.

## Dataset

Our benchmark uses approximately **100 hours of video** spanning:

- **37 visual styles** -- animation, cinematic, documentary, gameplay, live concert, surveillance, vlogs, and more
- **38 content domains** -- entertainment, sports, news, education, culinary, music, drama, gaming, corporate, travel
- **93,000+ scene-level results** across all four model variants

Scenes are extracted using VideoDB's visual and semantic boundary detection. Each scene is processed with 1-10 frames at 1 FPS.

## Prerequisites

**Python 3.12+** is required.

You will need API keys from three services:

| Key | Required For | Get It At |
|-----|-------------|-----------|
| `VIDEO_DB_API_KEY` | Video upload + scene extraction | [console.videodb.io](https://console.videodb.io) |
| `GOOGLE_API_KEY` | Gemini VLM inference | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | LLM judge for metrics | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

## Quick Start

```bash
git clone https://github.com/video-db/gemini-reasoning-eval.git
cd gemini-reasoning-eval

pip install -r requirements.txt

cp .env.sample .env
# Fill in your API keys in .env
```

Add your videos to `configs/default.yaml` (see [Video Sources](#video-sources) below), then run the full pipeline:

```bash
python run.py --config configs/default.yaml
```

## Video Sources

Edit the `videos.sources` list in your config file. You can use local files, URLs, or both.

**Local files** -- provide absolute paths to video files on your machine:

```yaml
videos:
  sources:
    - "/home/user/videos/clip1.mp4"
    - "/home/user/videos/clip2.mov"
```

**URLs** -- YouTube links, direct video URLs, or any publicly accessible video:

```yaml
videos:
  sources:
    - "https://www.youtube.com/watch?v=your_video_id"
    - "https://example.com/video.mp4"
```

Local paths and URLs can be mixed in the same list. The pipeline detects each type automatically.

## Configuration

All options are set in a YAML config file. The default is `configs/default.yaml`.

**Gemini variants** -- specify model + thinking budgets per family (`-1` = dynamic/unlimited):

```yaml
gemini:
  variants:
    - model: "gemini-2.5-flash"
      thinking_budgets: [128, -1]
    - model: "gemini-2.5-flash-lite"
      thinking_budgets: [512, 1024]
  max_concurrent: 10   # parallel Gemini API calls per variant
```

**LLM judge** -- OpenAI model used to evaluate coverage metrics:

```yaml
judge:
  model: "gpt-5"
  batch_size: 1
  max_concurrent: 5    # parallel judge batches
  tokens_per_minute: 2000000
```

**Visualization** -- output format and resolution for figures:

```yaml
visualization:
  enabled: true
  format: "png"
  dpi: 300
```

## Usage

### Full Pipeline

Runs all five steps in sequence: upload, extract, infer, evaluate, visualize.

```bash
python run.py --config configs/default.yaml
```

### Run Individual Steps

Run or re-run a specific step independently:

```bash
python run.py --config configs/default.yaml --step upload      # Upload videos to VideoDB
python run.py --config configs/default.yaml --step extract     # Extract scenes and frame URLs
python run.py --config configs/default.yaml --step infer       # Run Gemini VLM inference
python run.py --config configs/default.yaml --step evaluate    # Compute evaluation metrics
python run.py --config configs/default.yaml --step visualize   # Generate publication figures
```

### Resume an Interrupted Run

```bash
python run.py --config configs/default.yaml --resume
```

Already-completed steps are skipped automatically. Safe to re-run after any interruption.

## Pipeline

```
1. Upload        Local files / URLs -> VideoDB collection
      |
2. Extract       Videos -> scene boundaries + frame URLs
      |
3. Infer         Frames + prompt -> Gemini VLM (per model x budget, parallel)
      |                              -> thought stream + structured JSON
4. Evaluate      LLM judge -> Recall, Precision (parallel batches)
                 NLTK -> Contentfulness
                 -> F1, Dominant Entity
      |
5. Visualize     Aggregated metrics -> publication figures
```

## Metrics

### Contentfulness

Fraction of thought stream words that are scene content (noun/verb phrases) vs. meta-commentary ("let me analyze", "I will format").

```
Contentfulness = content_words / total_words
```

### Recall (Thought Coverage)

Of all items the model reasoned about, how many appear in the final output?

```
Recall = |thought_items matching final_items| / |thought_items|
```

### Precision (Output Grounding)

Of all items in the final output, how many were grounded in the thought stream?

```
Precision = |final_items matching thought_items| / |final_items|
```

Low Precision indicates **compression-step hallucination** -- the model added content not present in its reasoning.

### F1

```
F1 = 2 * Recall * Precision / (Recall + Precision)
```

### Matching

Items are matched using cascaded fuzzy matching:
1. Exact match
2. Token-sort ratio >= 75
3. Partial ratio >= 75 (handles morphological variants like "speaks" / "speaking")

## Output Structure

```
outputs/
  upload_manifest.json           # source -> media_id mapping
  scenes/{media_id}.json         # scene boundaries + frame URLs
  raw/{variant}/{media_id}.json  # raw VLM responses per variant
  metrics/{variant}.json         # per-scene evaluation metrics
  metrics/aggregated.json        # variant-level summary
  figures/                       # publication-quality PNGs
```

## Project Structure

```
gemini-reasoning-eval/
  run.py              # CLI entry point
  src/
    config.py         # YAML config + env loading
    pipeline.py       # Pipeline orchestrator (upload/extract/infer/evaluate/visualize)
    gemini.py         # Gemini VLM caller with async + thinking budget support
    judge.py          # OpenAI LLM judge with caching and rate limiting
    metrics.py        # All evaluation metrics (contentfulness, coverage, dominant)
    schemas.py        # Pydantic data models
    plots.py          # Publication-quality figure generation
  configs/
    default.yaml      # Pipeline configuration
```

## About VideoDB

[VideoDB](https://videodb.io) is the perception, memory, and action layer for AI agents operating on video and audio. It turns continuous media (files, live streams, and desktop capture) into structured context, searchable moments with playable evidence, and event triggers that can drive automations and editing.

## Community & Support
- **Docs**: [docs.videodb.io](https://docs.videodb.io)
- **Issues**: [GitHub Issues](https://github.com/video-db/gemini-reasoning-eval/issues)
- **Discord**: [Join community](https://discord.gg/py9P639jGz)

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/video-db/gemini-reasoning-eval.svg?style=flat
[stars-url]: https://github.com/video-db/gemini-reasoning-eval/stargazers
[issues-shield]: https://img.shields.io/github/issues/video-db/gemini-reasoning-eval.svg?style=flat
[issues-url]: https://github.com/video-db/gemini-reasoning-eval/issues
[website-shield]: https://img.shields.io/website?url=https%3A%2F%2Fvideodb.io&style=flat
[website-url]: https://videodb.io
[discord-shield]: https://img.shields.io/badge/discord-join-7289da?style=flat&logo=discord
[discord-url]: https://discord.gg/py9P639jGz
