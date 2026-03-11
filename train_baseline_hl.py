import wandb
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from hl_tokenizer import HLTokenizer
from tqdm import tqdm
import random

max_samples = 10000
batch_size = 64
hl_vocab_size = 10000

wandb.init(project="hyper-language", config={"dataset": "allenai/c4-multi (zh/en/ja/fr)", "max_samples": max_samples, "batch_size": batch_size, "hl_vocab": hl_vocab_size,"multi_lang": True})

model_sizes = ['Qwen/Qwen2-1.5B']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

results = {}

for size in model_sizes:
    print(f"\n=== {size.upper()} OOM-safe GPU Benchmark ===")
    tokenizer = AutoTokenizer.from_pretrained(size)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        size, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)  # Load once

    samples_per_lang = max_samples // 4
    dataset_configs = [
        ('allenai/c4', 'en', 'text', 'en'),
        ('allenai/c4', 'zh', 'text', 'zh'),  # 中文 C4
        ('allenai/c4', 'ja', 'text', 'ja'),  # 日文 C4
        ('allenai/c4', 'fr', 'text', 'fr')   # 法文 C4
    ]

    texts = []
    for dataset_name, subdataset, field, lang in dataset_configs:
        print(f"Loading {lang}...")
        ds = load_dataset(dataset_name, subdataset, split='train', streaming=True)
        lang_texts = []
        for ex in ds:
            text = ex[field].strip()
            if len(text) > 20:
                lang_texts.append(text)
            if len(lang_texts) >= samples_per_lang:
                break
        texts.extend(lang_texts)
        print(f"Loaded {len(lang_texts)} {lang} samples")

    random.shuffle(texts)
    texts = texts[:max_samples]
    print(f"Total mixed texts: {len(texts)}")


    # Baseline tokenize time
    start = time.time()
    encodings_base_full = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors='pt')
    base_tokenize_time = time.time() - start
    base_tokens_avg = encodings_base_full['input_ids'].shape[1]

    # Baseline PPL batches
    print("Baseline PPL batches...")
    ppl_base_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Base batches"):
        batch_texts = texts[i:i+batch_size]
        encodings_b = tokenizer(batch_texts, truncation=True, max_length=128, padding=True, return_tensors='pt')
        encodings_b = {k: v.to(device) for k, v in encodings_b.items()}
        with torch.no_grad():
            outputs = model(**encodings_b, labels=encodings_b['input_ids'])
            ppl_base_list.append(torch.exp(outputs.loss).item())
        torch.cuda.empty_cache()
    ppl_base = np.mean(ppl_base_list)

    # HL
    print("HL vocab...")
    hl_tok = HLTokenizer(vocab_size=hl_vocab_size)
    hl_build_start = time.time()
    hl_tok.build_vocab(texts)
    hl_build_time = time.time() - hl_build_start

    print("HL encode...")
    hl_encode_start = time.time()
    hl_texts = [hl_tok.encode(t) for t in tqdm(texts, desc="encode")]
    hl_encode_time = time.time() - hl_encode_start

    print("HL tokenize time...")
    hl_tokenize_start = time.time()
    encodings_hl_full = tokenizer(hl_texts, truncation=True, max_length=128, padding=True, return_tensors='pt')
    hl_tokenize_time = time.time() - hl_tokenize_start
    hl_tokens_avg = encodings_hl_full['input_ids'].shape[1]

    # HL PPL batches
    print("HL PPL batches...")
    ppl_hl_list = []
    for i in tqdm(range(0, len(hl_texts), batch_size), desc="HL batches"):
        batch_hl = hl_texts[i:i+batch_size]
        encodings_h = tokenizer(batch_hl, truncation=True, max_length=128, padding=True, return_tensors='pt')
        encodings_h = {k: v.to(device) for k, v in encodings_h.items()}
        with torch.no_grad():
            outputs = model(**encodings_h, labels=encodings_h['input_ids'])
            ppl_hl_list.append(torch.exp(outputs.loss).item())
        torch.cuda.empty_cache()
    ppl_hl = np.mean(ppl_hl_list)

    res = {
        'base_tokens_avg': base_tokens_avg,
        'hl_tokens_avg': hl_tokens_avg,
        'savings_pct': (1 - hl_tokens_avg / base_tokens_avg) * 100,
        'base_tokenize_s': base_tokenize_time,
        'hl_build_s': hl_build_time,
        'hl_encode_s': hl_encode_time,
        'hl_tokenize_s': hl_tokenize_time,
        'base_ppl': ppl_base,
        'hl_ppl': ppl_hl,
        'ppl_change_pct': (ppl_hl - ppl_base) / ppl_base * 100 if ppl_base > 0 else 0
    }
    results[size] = res
    wandb.log(res)
    print("Results:", res)

wandb.finish()
print("\nDone! wandb.ai/wanwaising/hyper-language")
