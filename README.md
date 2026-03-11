# Hyper-Language Tokenizer Benchmark

[![WandB](https://img.shields.io/badge/WandB-hyper--language-blue.svg)](https://wandb.ai/wanwasing/hyper-language)

## Overview

Benchmarking a custom **Hyper-Language (HL) Tokenizer** for multi-lingual text compression efficiency.

- **Languages**: English (en), Chinese (zh), Japanese (ja), French (fr) from `allenai/c4`.
- **Models**: Qwen/Qwen2-1.5B (FP16, CUDA).
- **Metrics**:
  - Token length savings vs. baseline tokenizer.
  - Tokenization time (build + encode + model tokenize).
  - Perplexity (PPL) degradation.
- **Dataset**: 10k mixed samples (~2.5k per lang), max 128 tokens.
- **Vocab**: 10k for HL.

Compares HL against:
- Qwen tokenizer (baseline).
- Tiktoken cl100k_base.
- GPT-2 tokenizer (approx).

## Installation

```bash
pip install -r requirements.txt
# torch, transformers, datasets, wandb, tqdm, numpy, tiktoken, accelerate
```

**CUDA Required** (tested on WSL-Ubuntu with CUDA 12.1).

## Usage

### 1. Baseline HL vs Qwen (Multi-lang)
```bash
python train_baseline_hl.py
```

### 2. Multi Baseline (Qwen + Tiktoken + GPT2)
```bash
python train_multi_base.py
# or fixed version
python train_multi_base_fixed.py
```

Both log to [WandB](https://wandb.ai/wanwasing/hyper-language).

## Original Ideas (HL Innovation)

**Core Original Concept**: *Abstract Shared IDs* – Cross-language words cluster into universal \"hyper-concepts\" (e.g., [HL001] = apple/蘋果/pomme/りんご). 

- **Multi-Lingual Merging**: Manual + auto (future: clustering). Beats per-lang tokenizers.
- **Hybrid Tokenize**: jieba (zh) + regex (others) → global freq top + concepts first.
- **Hash Fallback**: Rare words → short [Hxxxx] MD5.
- **Abstract Decode**: Tokens → concept reps (not orig word), enables \"semantic compression\".

Vocab build prioritizes shared concepts → massive savings on samples.

See `hl_tokenizer.py` for implementation (custom vocab builder + encode).

## Notes

- Batch size: 64 (OOM-safe for 1.5B on typical GPU).
- Streaming dataset load.
- Empty CUDA cache between batches.

## License

MIT