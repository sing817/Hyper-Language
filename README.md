# Hyper-Language Tokenizer (HL Tokenizer v5.1)

Cross-language semantic compression via Chinese-pivot shared vocabulary. Boost LLM training efficiency by converting to high-density tokens.

## 💡 Idea Origin
 Observed Chinese tokens carry higher information density than English, directly impacting LLM training time (fewer tokens = faster). Proposal: Convert multilingual data to \"Hyper Language (HL)\"—the densest representation—for pre-training efficiency gains. 

## 🔥 Design Essence
**Cross-Language Vocabulary Sharing** (中文樞紐共享):

**Decode Table v5.1** (lang-block bare HL):
| Format                       | Decode (orig lang / zh fallback) |
|------------------------------|----------------------------------|
| `[HL你好][HL世界]`           | `你好世界` (zh bare)            |
| `[en][HL你好][HL世界][/en]`  | `Hello World` / 你好世界        |
| `[ja][HL你好][HL世界][/ja]`  | `こんにちは世界` / 你好世界     |


**Example v5.1**:
```
Input: Hello World,你好世界,こんにちは世界
Pending: [en]Hello World[/en][HL你好][HL世界][ja]こんにちは世界[/ja]
Trans: [en]你好世界[/en]...[ja]你好世界[/ja]
Final: [en][HL你好][HL世界][/en][HL你好][HL世界][ja][HL你好][HL世界][/ja]
Decode: Hello World, 你好世界, こんにちは世界 ✓
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