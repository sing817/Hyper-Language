import json
import os
from collections import Counter
from hl_tokenizer import HLTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Params
MAX_SAMPLES = 100000  # 100k mixed
SAMPLES_PER_LANG = MAX_SAMPLES // 4  # 25k each
HL_VOCAB_SIZE = 100000  # 標準 100k vocab
OUTPUT_DIR = 'tokenizer'

print(f"Training HL Tokenizer: {MAX_SAMPLES} samples, vocab_size={HL_VOCAB_SIZE}")

# Load 100k texts (en/zh/ja/fr C4)
texts = []
dataset_configs = [
    ('allenai/c4', 'en', 'text'),
    ('allenai/c4', 'zh', 'text'),
    ('allenai/c4', 'ja', 'text'),
    ('allenai/c4', 'fr', 'text')
]

for ds_name, subds, field in dataset_configs:
    print(f"Loading {subds}...")
    ds = load_dataset(ds_name, subds, split='train', streaming=True)
    lang_texts = []
    for ex in tqdm(ds, total=SAMPLES_PER_LANG, desc=f"{subds} samples"):
        text = ex[field].strip()
        if len(text) > 20:  # filter short
            lang_texts.append(text)
        if len(lang_texts) >= SAMPLES_PER_LANG:
            break
    texts.extend(lang_texts)
    print(f"Loaded {len(lang_texts)} {subds} samples")

print(f"Total texts: {len(texts)}")

# Build HL
hl_tok = HLTokenizer(vocab_size=HL_VOCAB_SIZE)
hl_tok.build_vocab(texts)

# Stats
print(f"Final vocab size: {len(hl_tok.hl_vocab)}")
print(f"Concepts used: {len(hl_tok.concepts)}")
print("Sample vocab:", list(hl_tok.hl_vocab.items())[:20])

# Save JSONs (HF-compatible)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f'{OUTPUT_DIR}/hl_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(hl_tok.hl_vocab, f, ensure_ascii=False, indent=2)
with open(f'{OUTPUT_DIR}/reverse_hl.json', 'w', encoding='utf-8') as f:
    json.dump(hl_tok.reverse_hl, f, ensure_ascii=False, indent=2)
with open(f'{OUTPUT_DIR}/concepts.json', 'w', encoding='utf-8') as f:
    json.dump(hl_tok.concepts, f, ensure_ascii=False, indent=2)
with open(f'{OUTPUT_DIR}/tokenizer.json', 'w', encoding='utf-8') as f:  # summary
    summary = {
        'vocab_size': len(hl_tok.hl_vocab),
        'samples_used': len(texts),
        'langs': ['en', 'zh', 'ja', 'fr']
    }
    json.dump(summary, f, indent=2)

print(f"Saved to {OUTPUT_DIR}/")
print("Next: pip install huggingface_hub; huggingface-cli upload /path/to/tokenizer wanwasing/hyper-language-tokenizer")