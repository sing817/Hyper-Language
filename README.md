# ⚠️ 已放棄 (Abandoned) — hyper-language-abandoned

**這個儲存庫已正式放棄，不再維護。**

---

## 為什麼放棄？

### 原始動機
這個專案原本想改善**小模型在多語言場景下的表現**。

核心想法是：同一個概念（例如「蘋果」）在不同語言中存在大量重複內容。如果能找到方法有效壓縮這些跨語言的重複，或許能讓小模型用更少的資料和運算，學到更好的多語言能力。

最初的假設是：中文資訊密度較高，或許可以把多語言輸入先轉換成以中文為主的「Hyper Language」中間表示，再進行訓練，以達到壓縮與對齊的效果。

### 實際嘗試的做法
後續實現成一個複雜的預處理 tokenizer，主要包含：
- 使用 LLM（Grok）進行語言標記與切分
- 使用 NLLB-200-distilled 大型翻譯模型將非中文內容翻譯成中文
- 大量啟發式 fallback 機制
- 加入語言標籤（`[en]`、`[原]`、`[ja]` 等）試圖保留可逆性

目標是「單次編碼、多語言轉中文、資訊密度最大化」。

### 後來發現的主要問題

經過實際實作與測試後，這個方向存在以下根本限制：

1. **效率目標自相矛盾**  
   為了壓縮重複，卻引入 Grok API + NLLB 1.3B 兩個重量級模型，單句延遲經常達到數秒到十幾秒，完全違背小模型追求效率的初衷。

2. **壓縮效果多為幻覺，伴隨資訊損失**  
   許多「高壓縮」案例來自翻譯失敗後的粗暴 fallback 映射，大量語義細節被抹除，對模型學習反而有害。

3. **把問題轉移而非解決**  
   沒有真正減少模型需要學習的內容，而是把跨語言對齊的難題，轉嫁到「高品質翻譯」這個更困難的子問題上。

4. **在表示層的改進空間有限**  
   小模型多語言能力不足，更主要的瓶頸在資料品質、訓練目標設計，以及模型架構如何共享跨語言知識，而非單純的輸入表示轉換。

5. **難以規模化使用**  
   整個 pipeline 過度依賴外部付費 API 和大型本地模型，穩定性與實用性都很差。

### 最終結論
這個「以中文為高密度樞紐語言進行預處理壓縮」的技術路線，經過驗證後被認為**不適合繼續發展**。

雖然「減少跨語言概念重複、提升小模型多語言效率」這個問題本身仍然值得研究，但更好的方向應該放在資料策展、合成資料、跨語言對齊目標，以及模型架構層面，而不是在 tokenizer 層做高成本、有損的轉換。

---

**保留此儲存庫僅作為歷史記錄**，讓後來對類似想法有興趣的人，可以看到實際嘗試後遇到的具體問題，避免重蹈覆轍。

**本專案已不再維護，也不會再有更新。**

---

# Hyper-Language Tokenizer (HL Tokenizer v5.3) [歷史文件]

> **注意**：以下為原始專案文件，已不再維護。僅供歷史參考。

## 💡 Core Concept & Motivation (Original Idea: 2025-05-10)

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
