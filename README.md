# Hyper-Language Tokenizer (HL Tokenizer v2.5)

Cross-language semantic compression via Chinese-pivot shared vocabulary. Boost LLM training efficiency by converting to high-density tokens.

## 💡 Idea Origin
 Observed Chinese tokens carry higher information density than English, directly impacting LLM training time (fewer tokens = faster). Proposal: Convert multilingual data to \"Hyper Language (HL)\"—the densest representation—for pre-training efficiency gains. 

## 🔥 Design Essence
**Cross-Language Vocabulary Sharing** (中文樞紐共享):
- **Zh Pivot Vocab**: Chinese → `[HL你好]` (lossless, zero-cost).
- **Variants**: Non-Zh → `[HL你好:en]` (shared pivot, decode to orig/fallback).

**Decode Table**:
| Token         | Decode (Variant / Zh Fallback)     |
|---------------|-----------------------------------|
| `[HL你好]`    | `你好` (base)                     |
| `[HL你好:en]` | `hello` / 你好                   |
| `[HL苹果:en]` | `apple` / 苹果                    |
| `[HL狗:es]`   | `perro` / 狗                      |

**Core Compression**: Shared HL IDs (Zh pivot >50% efficiency), low PPL via semantic abstraction.

**Example**:
```
Input: 你好 hello 蘋果 apple
Encode: [HL你好] [HL你好:en] [HL苹果] [HL苹果:en]
Decode: 你好 hello 蘋果 apple  ✓ lossless
```

## 🚀 Quick Start
### Tokenizer
```python
from hl_tokenizer import HLTokenizer
t = HLTokenizer(vocab_size=20000)
t.build_vocab(your_texts)  # Builds zh_pivot + variants
encoded = t.encode("Hello world 你好世界")
print(t.decode(encoded))  # Hello world 你好 世界 ✓
```

## 📁 Files
- `hl_tokenizer.py`: Core (v2.5: jieba + lang detect + variant mapping).
- `train_*.py`: Benchmark scripts.
- `HL_TOKENIZER_FLOW.md`: Architecture diagram.
- `data/`: Sample datasets (CLUE/TNews).
- `requirements.txt`

## 🔮 Roadmap
- Qwen batch pivot (drop googletrans).
- Embedding-based auto-clustering for variants.
- Full multilingual benchmarks (add ja/ko/es).

**v3.2 (2026-03-12)**: Tagged preprocess flow [:lang:待trans] → batch trans → finalize [word:lang]. Separates heavy NLLB from tokenization for pipeline efficiency.