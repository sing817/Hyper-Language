# Hyper-Language 分詞器基準測試

[![WandB](https://img.shields.io/badge/WandB-hyper--language-blue.svg)](https://wandb.ai/wanwasing/hyper-language)

## 概述

基準測試自訂 **Hyper-Language (HL) 分詞器** 在多語言文字壓縮效率。

- **語言**：英文 (en)、中文 (zh)、日文 (ja)、法文 (fr)，來自 `allenai/c4`。
- **模型**：Qwen/Qwen2-1.5B (FP16, CUDA)。
- **指標**：
  - 相對於基準分詞器的 token 長度節省。
  - 分詞時間 (建置 + 編碼 + 模型分詞)。
  - Perplexity (PPL) 退化。
- **資料集**：10k 混合樣本 (~2.5k 每語言)，最大 128 tokens。
- **詞彙**：HL 10k。

比較 HL 對：
- Qwen 分詞器 (基準)。
- Tiktoken cl100k_base。
- GPT-2 分詞器 (近似)。

## 安裝

```bash
pip install -r requirements.txt
# torch, transformers, datasets, wandb, tqdm, numpy, tiktoken, accelerate
```

**需要 CUDA** (WSL-Ubuntu CUDA 12.1 測試)。

## 使用

### 1. HL vs Qwen 基準 (多語言)
```bash
python train_baseline_hl.py
```

### 2. 多基準 (Qwen + Tiktoken + GPT2)
```bash
python train_multi_base.py
# 或修復版
python train_multi_base_fixed.py
```

兩者皆記錄至 [WandB](https://wandb.ai/wanwasing/hyper-language)。

## 原創想法 (HL 創新)

**核心原創概念**：*抽象共享 ID* – 跨語言詞彙聚類成通用「超概念」（例：[HL001] = apple/蘋果/pomme/りんご）。

- **多語言合併**：手動 + 自動（未來：聚類）。超越單語言分詞器。
- **混合分詞**：jieba (zh) + regex (其他) → 全球頻次 top + 概念優先。
- **Hash 備案**：罕見詞 → 短 [Hxxxx] MD5。
- **抽象解碼**：Token → 概念代表（非原詞），實現「語義壓縮」。

詞彙建置優先共享概念 → 樣本大幅節省。

詳見 `hl_tokenizer.py` 實作 (自訂詞彙建置 + 編碼)。

## 注意

- Batch size: 64 (1.5B 典型 GPU OOM-safe)。
- 串流資料集載入。
- 批次間清空 CUDA 快取。

## 授權

MIT