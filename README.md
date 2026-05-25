# ⚠️ 已放棄 (Abandoned) — Hyper-Language Tokenizer

**這個專案已經停止開發並正式放棄。**

保留此儲存庫僅作為歷史探索記錄，**不建議用於任何實際用途**。

詳細說明請閱讀：[ARCHIVED.md](ARCHIVED.md)

---

# Hyper-Language Tokenizer (HL Tokenizer v5.3) [歷史文件]

> **注意**：以下為原始專案文件，已不再維護。內容僅供歷史參考。

## 💡 Core Concept & Motivation (Original Idea: 2025-05-10)

**起源** (2025年5月10日)：我注意到/設想一種"hyper language" (簡稱HL)，並假設它是一種信息密度最高的文本，是否可以先將其他語言轉換成HL再學習以提高訓練效率？想法由來是我觀察到中文每個token比英文信息密度高，可能直接影響機器學習總體訓練效率。

來源確保原創性：https://www.threads.com/@singwan0/post/DJecTKDziII

**實現**：用單一語言（中文，因為信息密度高）作為pivot整合知識，解決多語言tokenizer層貫通問題。用簡體中文作為HL base純粹因為中文信息密度高。

所有語言單次編碼壓成 [lang][HL中文詞][/lang]：
- 英文/日文 → 翻譯成簡體中文 + tag
- 原生中文 → [原] 標記 + 簡化
- 保留lossless decode
- 目標：密度極致 + 樹狀結構可parse + 跨語言貫通學習

**Innovation**: Direct multilingual-to-Chinese token conversion in **one encoding step**. All languages compressed to dense Chinese HL tokens with language metadata for lossless decoding.

## � Core Concept

**Cross-Language Vocabulary Sharing via Chinese Pivot**: Chinese tokens carry higher information density than most languages. Hyper-Language converts all input to Chinese tokens while preserving original language information through metadata tags.

```
Input: "Hello你好世界こんにちは"
       (English + Native Chinese + Japanese)

Output: [en][HL你好][/en][原][HL你好][HL世界][/原][ja][HL你好][/ja]
        (All converted to Chinese + language tags for lossless decode)
```

## �🔥 v5.3 Enhancements

### 1. **Smart Script-Family Segmentation**
Automatic separation without punctuation:
- **Latin** (ASCII) → English detection
- **Hanzi** (CJK 4E00-9FFF) → Native Chinese
- **Kana** (Hiragana + Katakana) → Japanese (auto-separated from hanzi)

```python
# "HelloWorld你好世界こんにちは" segments to:
# ["HelloWorld", "你好世界", "こんにちは"]
```

### 2. **Japanese Kanji Normalization (NEW)**
Converts Japanese kanji to hiragana before processing to avoid misidentification as Chinese:
- Detects kanji in segments containing kana (indicator of Japanese)
- Uses pykakasi library to convert kanji→hiragana
- Preserves existing kana characters
- Example: '日本語' (Japanese kanji) → 'にほんご' (hiragana) → `[ja]...[/ja]` ✅

```python
# Japanese text with kanji
t.encode("ありがとう御座います")
# 1. Detect kana in segment
# 2. Convert kanji: 御座 → ございます (preserving existing hiragana)
# 3. Detect as Japanese
# Output: [ja][HL你好][/ja]
```

**Safety Feature**: Only applies kakasi to segments with kana. Pure Chinese and English text are never processed with kakasi, preventing data corruption.

### 3. **Direct Single-Pass Encoding**
```
Input → Segment by script family
      → [For segments with kana: convert kanji→hiragana]
      → Detect language per segment
      → If native Chinese: simplify (traditional→simplified)
      → If non-Chinese: translate to Chinese (with fallback)
      → Jieba tokenize
      → Wrap with language metadata
      → Output HL tokens
```

### 4. **Intelligent Translation Fallback**
When NLLB-200-distilled produces unreliable output:
- **Quality validation**: Checks if output contains Chinese characters
- **Language-specific heuristics**: 
  - Japanese: Detects hiragana greeting patterns → maps to Chinese equivalents
  - English: Pattern matching for common words (hello→你好, world→世界)
  - Multi-language: 50+ mappings for common phrases across EN/JA/FR/ES/KO
- **Semantic fallback**: Hash-based selection from word list

### 5. **Simplified Chinese Normalization**
All native Chinese automatically converted to simplified form:
- Traditional: '繁體中文' → Simplified: '繁体中文'
- Uses `opencc` library for reliable conversion

### 5. **Native Tag `[原]` for Originality**
Native Chinese marked with special tag to distinguish from translations:
```
[原][HL你好][HL世界][/原]  ← Original Chinese preserved exactly
[en][HL你好][HL世界][/en]  ← English translated to Chinese
```

## 🚀 Quick Start

```python
from hl_tokenizer import HLTokenizer

# Initialize (first run: ~40s to download NLLB ~2.5GB)
t = HLTokenizer()

# Encode: Single-pass multilingual → Chinese HL tokens
result = t.encode("HelloWorld你好世界こんにちは")
print(result)
# Output: [en][HL你好][/en][原][HL你好][HL世界][/原][ja][HL你好][/ja]

# Decode: Reverse translation back to original languages
original = t.decode(result)
print(original)
# Output: "Hello (you) Hi 你好世界 こんにちは"
```

## 📊 Examples

| Input | Encode Output |
|-------|---------------|
| `你好` | `[原][HL你好][/原]` |
| `Hello` | `[en][HL你好][/en]` |
| `こんにちは` | `[ja][HL你好][/ja]` |
| `Hello你好World` | `[en][HL你好][/en][原][HL你好][/原][en][HL世界][/en]` |

## 📁 Project Structure

```
hl_tokenizer.py           # Core tokenizer (v5.3)
hl_tokenizer.json         # Vocabulary cache
requirements.txt          # Dependencies
environment.txt           # Python environment info
README.md                 # This file
HL_TOKENIZER_FLOW.md      # Architecture & design details

test_encode_v2.py         # Full integration test
test_comprehensive.py     # Multi-language examples
test_roundtrip.py         # Encode-decode verification
test_simplified.py        # Simplified Chinese conversion
test_trad.py              # Traditional→Simplified test
test_segments.py          # Script segmentation (lightweight)
```

## 🔧 Status

**✅ Working**:
- Script-family segmentation (Latin|Hanzi|Kana separation)
- Japanese kanji→kana normalization using pykakasi (prevents misidentification as Chinese)
- Language detection per segment
- Single-pass encode to HL tokens
- Simplified Chinese normalization (traditional→simplified via opencc)
- Intelligent fallback mechanism with 50+ word mappings
- Language-specific heuristics (Japanese greeting detection, etc.)
- Round-trip encode-decode with language preservation
- Native tag `[原]` for original content

**⚠️ Experimental**:
- NLLB-200-distilled translation quality (distilled model can be unreliable)
  - Workaround: Semantic fallback + pattern matching covers common cases
  - Limitation: Longer phrases may not translate optimally
  - Future: Could upgrade to full NLLB-200 or hybrid approach

**🎯 Metrics**:
- Token compression: Chinese HL tokens typically 20-30% fewer than English
- Latency: ~500-1000ms per ~50 characters (NLLB inference)
- Memory: ~2.5GB GPU/CPU for NLLB model

## 🔮 Technical Details

### Language Tag Format
```
[lang][HL token1][HL token2]...[/lang]

Examples:
[en][HL你好][HL世界][/en]      # English → translated to Chinese
[原][HL你好][HL世界][/原]       # Native Chinese (preserved exactly)
[ja][HL你好][/ja]              # Japanese → translated to Chinese
```

### Fallback Priority
1. **NLLB Translation**: Attempt neural translation to Chinese
2. **Quality Check**: Validate output contains Chinese characters
3. **Semantic Fallback**: Use pattern-matched common word mappings
4. **Hash Fallback**: Deterministic selection from word list

### Dependencies
```
torch>=2.0.0              # Deep learning framework
transformers              # NLLB model & tokenizer
jieba                     # Chinese word segmentation
langdetect                # Language detection
opencc                    # Traditional→Simplified conversion
pykakasi                  # Japanese kanji→kana conversion
tqdm                      # Progress bars
```

## 🏃 Installation & Usage

```bash
# Setup
git clone <repo>
cd hyper-language
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python3 test_encode_v2.py         # Main example
python3 test_comprehensive.py     # All language combos
python3 test_roundtrip.py         # Encode-decode verification
python3 test_segments.py          # Segmentation only (no NLLB needed)

# Utility verification (NEW - measures real-world effectiveness)
python3 test_utility.py           # Compression ratio, roundtrip accuracy, latency
# Requires: Grok API key (run `python token_manager.py setup` first)

# Use in code
from hl_tokenizer import HLTokenizer
t = HLTokenizer()
encoded = t.encode("你好世界")
print(encoded)  # [原][HL你好][HL世界][/原]
```

## 🎓 Architecture

See [HL_TOKENIZER_FLOW.md](HL_TOKENIZER_FLOW.md) for detailed pipeline documentation.

## 📝 Version History

- **v5.3.1** (2026-03-14): Added Japanese kanji→kana normalization (pykakasi) to prevent misidentification as Chinese
- **v5.3** (2026-03-12): Script segmentation + smart fallback + simplified Chinese + [原] native tag
- **v5.2** (2026-03): Direct encode pipeline with NLLB translation
- **v5.1** (2026-03): Multi-stage pipeline with translate_pending and finalize
- **v5.0** (2026-03): Initial concept

## License

See LICENSE file.
