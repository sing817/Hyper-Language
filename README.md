# Hyper-Language Tokenizer (HL Tokenizer v2.4) / 超語言分詞器 v2.4

## 🔥 Design Essence / 設計精髓
**Chinese self-ID + variant mapping**:
- **Zh direct self-ID**: Chinese words → `[HL你好]` (zero-cost lossless).
- **Non-zh variants**: Non-Chinese → `[HL你好:en]` (shared zh pivot concepts).
- **Decode strategy**:
  | Token          | Decode (variant orig / fallback zh) |
  |----------------|-------------------------------------|
  | `[HL你好]`     | `你好` (zh base)                    |
  | `[HL你好:en]`  | `hello` (orig lang back-trans)      |
  | `[HL狗:es]`    | `perro`                             |

**Compression core**: Multi-lang share HL ID, zh-heavy >50% ratio, low PPL (semantic abstraction).

Example / 示例:
```
Input: 你好 hello apple / Hello world 蘋果
Encode: [HL你好] [HL你好:en] [HL苹果:en]
Decode: 你好 hello 苹果
```

## 🚀 Usage / 使用
```python
from hl_tokenizer import HLTokenizer
t = HLTokenizer(vocab_size=20000)
t.build_vocab(your_texts)  # Manual + data → zh_bases
encoded = t.encode(\"你好世界 Hello world\")  # [HL你好] [HL世界] [HL世界:en]...
print(t.decode(encoded))  # 你好 世界 Hello world ✓ lossless
```

## 📁 Files / 檔案
- `hl_tokenizer.py`: Core (per-segment lang tokenize v2.4, jieba boost)
- `train_*.py`: Benchmarks (Qwen/Tiktoken PPL)
- `HL_TOKENIZER_FLOW.md`: Flowchart
- `data/`: CLUE/TNews
- `wandb`: [wanwaising/hyper-language](https://wandb.ai/wanwasing/hyper-language)

## 🔮 Next / 下一步
- LLM pivot (Qwen batch, drop googletrans)
- Embedding cluster for variants
- Multi-lang benchmarks

**v2.4 (2026-03-12 Claw)**: Fixed pangram glue; bilingual docs.