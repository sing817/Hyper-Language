# Hyper-Language Tokenizer (HL Tokenizer v2)

## Core Idea 核心想法
**Zh self-ID + lang variants**：中文詞 direct [HL你好] (0 overhead lossless)，non-zh variants [HL你好:en]=hello，共享概念壓縮多 lang。

- Encode: 你好 hello apple → [HL你好] [HL你好:en] [HL苹果:en]
- Decode: 你好 hello apple (orig lang) or fallback zh 你好 苹果
- Semantics preserved: manual concepts 'dog': ['狗', 'dog', 'perro'] → 同 base [HL狗]

**壓縮**：zh heavy text ratio >0.5+，ML PPL low (共享 ID)。

## Usage 使用
```python
from hl_tokenizer import HLTokenizer
t = HLTokenizer(vocab_size=20000)
t.build_vocab(your_texts)
encoded = t.encode(text)
decoded = t.decode(encoded)
```

## Files 檔案
- `hl_tokenizer.py`: core class
- `train_*.py`: baseline/multi train
- data/: clue/tnews samples
- wandb: wanwaising/hyper-language

## Next 後續
- LLM auto-map new non-zh → zh pivot
- Embedding sim cluster variants
- Benchmark PPL on cluecorp/tnews

(2026-03-12 v2 by Claw)