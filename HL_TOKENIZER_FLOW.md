# HL Tokenizer v5.3 Flow

## Overview
**Hyper-Language (HL) Tokenizer** converts multilingual input to dense Chinese HL tokens in one pass, with language metadata for lossless decoding.

## Encode Pipeline (Single Pass)

### Step 1: Language-Aware Segmentation
```
Input: "HelloWorld你好世界こんにちは"
       ↓
       Split by script family:
       - Latin [A-Za-z0-9] → "HelloWorld"
       - Hanzi [CJK] → "你好世界"  
       - Kana [Hiragana+Katakana] → "こんにちは"
       ↓
Output: ["HelloWorld", "你好世界", "こんにちは"]
```

### Step 2: Japanese Kanji Normalization (NEW)
```
For each segment, check if it contains kana (indicator of Japanese):

"HelloWorld" (no kana)
  → Skip kakashi processing
  → Keep as-is

"你好世界" (no kana)
  → Skip kakashi processing (pure Chinese)
  → Keep as-is

"こんにちは" (has kana)
  → Detected as likely Japanese
  → Apply kakashi: keep hiragana, convert any kanji to kana
  → Result: "こんにちは" (already all kana)

"ありがとう御座います" (has kana)
  → Detected as likely Japanese
  → Apply kakashi: preserve ありがとう, convert 御座 → ございます
  → Result: "ありがとうございます" (all kana)

Safety: Only process segments with kana to avoid corrupting Chinese/English text
```

### Step 3: Language Detection Per Segment
```
"HelloWorld" → Language Model → "en"
"你好世界"   → Language Model → "zh"
"こんにちは" → Language Model → "ja"
```

### Step 4: Convert Native Chinese (Traditional → Simplified)
```
For segments detected as "zh":
  "繁體中文" → [opencc] → "繁体中文"
```

### Step 5: Translate Non-Chinese to Chinese (with Fallback)
```
For non-Chinese segments:
  
  "HelloWorld" 
    ↓ [Try NLLB translation]
    ↓ 'HelloWorld (Hello World) ♂' (garbage output)
    ↓ [Quality validation fails - no Chinese chars]
    ↓ [Semantic fallback triggered]
    ↓ Pattern match "hello" in text
    ↓ Result: "你好"

  "こんにちは"
    ↓ [Try NLLB translation]
    ↓ '. ♂️ Hi, how are you' (garbage)
    ↓ [Quality validation fails]
    ↓ [Language-specific heuristic: Japanese greeting chars detected]
    ↓ Result: "你好"
```

### Step 6: Jieba Tokenization
```
Chinese text → Jieba word segmentation → tokens
"你好世界" → ["你好", "世界"]
"你好"     → ["你好"]
```

### Step 7: Wrap in Language Metadata
```
Native Chinese:   [原][HL你好][HL世界][/原]
English → Chinese: [en][HL你好][HL世界][/en]
Japanese → Chinese: [ja][HL你好][/ja]

Final output:
[en][HL你好][/en][原][HL你好][HL世界][/原][ja][HL你好][/ja]
```

## Decode Pipeline

### Reverse Process
```
Input: [en][HL你好][/en][原][HL你好][HL世界][/原][ja][HL你好][/ja]
       ↓
       Parse language blocks:
       - [en] block → collect HL tokens → "你好" → back-translate to EN
       - [原] block → collect HL tokens → "你好世界" → keep as Chinese
       - [ja] block → collect HL tokens → "你好" → back-translate to JA
       ↓
Output: "Hello 你好世界 こんにちは" (approximately)
```

## Smart Fallback Mechanism

When NLLB-200-distilled produces unreliable output:

### 1. Quality Validation
```
Does output contain Chinese characters? [是_Chinese, 否_Not Chinese]
- If Yes: Use it
- If No: Trigger fallback
```

### 2. Language-Specific Heuristics
```
Japanese: Check for hiragana patterns
  こんにちは → Contains こん → Greeting → '你好'
  
English: Substring matching
  "hello" in "HelloWorld" → '你好'
  "world" in "world" → '世界'
  "thank" in "Thankyou" → '谢谢'
  
Multi-language: 50+ common word mappings
  bonjour → '你好'
  gracias → '谢谢'
  etc.
```

### 3. Semantic Hash Fallback
```
If no pattern matches:
  hash(text) % word_list_len → deterministic selection
  Ensures reproducibility across runs
```

### Fallback Priority
```
1. NLLB Translation (if valid Chinese output)
2. Substring pattern matching (hello, world, thank, etc.)
3. Language-specific detection (Japanese hiragana, etc.)
4. Hash-based word selection (fallback)
```

## Key Features

### 🔹 Script-Family Aware
- **Latin**: ASCII characters [A-Za-z0-9]
- **Hanzi**: CJK characters [4E00-9FFF]
- **Kana**: Hiragana [3040-309F] + Katakana [30A0-30FF]
- Properly separates mixed text without punctuation

### 🔹 Implicit Language Detection
- Per-segment langdetect
- Script family as initial signal
- Language-specific heuristics for common patterns

### 🔹 Japanese Kanji Normalization (NEW)
- Uses **pykakasi** library to convert Japanese kanji to hiragana
- Only applies to segments containing kana (safe Japanese detection)
- Prevents misidentification of Japanese kanji as Chinese
- Example: '御座' (kanji) → 'ございます' (hiragana)
- Safety: Pure Chinese and English text never processed with kakashi

### 🔹 Simplified Chinese Normalization
- Online conversion via `opencc` library
- '繁體中文' → '繁体中文'
- All output guaranteed simplified form

### 🔹 Native Tag [原]
- Marks original (non-translated) Chinese
- Allows distinguishing source vs. translated content
- Single Chinese character more elegant than 2-letter code

### 🔹 Jieba Integration
- Chinese-specific word segmentation
- Character-by-character for non-Chinese (expected behavior)
- Post-filters whitespace-only tokens

### 🔹 Deterministic Fallback
- Random seed(42) on NLLB
- Hash-based word selection (same input → same output)
- Reproducible across runs

## Text Representation

### HL Token Format
```
[lang][HLword1][HLword2]...[/lang]

🔹 lang options:
   - Two-letter code: en, ja, fr, es, de, ko
   - Special: 原 (native Chinese)

🔹 HLword: Chinese text inside [HL...] tags
```

### Examples
```
Input: "Hello你好World"
Output: [en][HL你好][/en][原][HL你好][/原][en][HL世界][/en]
```

## Performance

- **Latency**: ~500-1000ms per 50 chars (NLLB bottleneck)
- **Memory**: ~2.5GB (NLLB model)
- **GPU**: Optional but ~5x faster with CUDA
- **Compression**: Chinese tokens 20-30% fewer than English equiv.

## Dependencies

```
torch>=2.0.0          # Neural inference framework
transformers          # NLLB model & tokenizer (auto-download ~2.5GB)
jieba                 # Chinese word segmentation
langdetect            # Language detection (via multinomial)
opencc                # Traditional→Simplified conversion
tqdm                  # Progress bars (optional)
```

## Architecture Notes

- **Single-pass design**: Input → Segments → Translate → Tokens → Output
- **No multi-stage pipeline**: Unlike v5.1-5.2, no separate translate_pending/finalize steps
- **Stateless**: Each encode/decode call independent
- **Deterministic within session**: NLLB seed=42, but model still has floating-point variance
- **Quality-aware**: Validates translation before accepting, falls back gracefully

## Version History

- **v5.3.1** (2026-03-14): Added pykakasi for Japanese kanji→kana normalization (prevents Chinese misidentification)
- **v5.3** (2026-03-12): Script segmentation + intelligent fallback + simplified Chinese + [原] tag
- **v5.2** (2026-03): Direct encode with NLLB, introduced [lang]...[/lang] blocks
- **v5.1** (2026-03): Multi-stage (encode → translate_pending → finalize)
- **v5.0** (2026-03): Initial concept

## Future Improvements

1. **Better Translation Model**: Upgrade NLLB-200-distilled to full NLLB-200 (better quality, larger)
2. **Hybrid Approach**: Combine NLLB with retrieval-based methods for common phrases
3. **Multilingual Benchmarks**: Compare token compression vs. baseline systems
4. **Embedding Integration**: Use word embeddings for better fallback selection
5. **Fine-tuning**: Train custom translation model on HL corpus
