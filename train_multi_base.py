import wandb
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from hl_tokenizer import HLTokenizer
from tqdm import tqdm
import random
import tiktoken

max_samples = 1000
batch_size = 64  # your safe
hl_vocab_size = 1000

wandb.init(project="hyper-language", config={"mode": "multi_base", "dataset": "allenai/c4-multi", "max_samples": max_samples, "batch_size": batch_size, "hl_vocab": hl_vocab_size})

model_size = 'Qwen/Qwen2-1.5B'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_size)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_size, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

samples_per_lang = max_samples // 4
dataset_configs = [
    ('allenai/c4', 'en', 'text', 'en'),
    ('allenai/c4', 'zh', 'text', 'zh'),
    ('allenai/c4', 'ja', 'text', 'ja'),
    ('allenai/c4', 'fr', 'text', 'fr')
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
print(f"Total texts: {len(texts)}")

# Qwen base
start = time.time()
encodings_base = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors='pt')
base_tokenize_s = time.time() - start
base_tokens_avg = encodings_base['input_ids'].shape[1]

# Tiktoken GPT cl100k
tik_enc = tiktoken.get_encoding("cl100k_base")
tik_lens = [len(tik_enc.encode(t)) for t in tqdm(texts, desc="tik")]
tik_avg = np.mean(tik_lens)

# GPT2 tokenizer
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
gpt2_lens = [len(gpt2_tok.encode(t, truncation=True, max_length=128)) for t in tqdm(texts[::10], desc="gpt2")]  # sample to speed
gpt2_avg = np.mean(gpt2_lens) * 10 / len(texts) * len(texts)  # approx

# HL
print("HL...")
hl_tok = HLTokenizer(vocab_size=hl_vocab_size)
hl_build_start = time.time()
hl_tok.build_vocab(texts)
hl_build_s = time.time() - hl_build_start

hl_encode_start = time.time()
hl_texts = [hl_tok.encode(t) for t in tqdm(texts, desc="HL encode")]
hl_encode_s = time.time() - hl_encode_start

hl_tokenize_start = time.time()
encodings_hl = tokenizer(hl_texts, truncation=True, max_length=128, padding=True, return_tensors='pt')
hl_tokenize_s = time.time() - hl_tokenize_start
hl_tokens_avg = encodings_hl['input_ids'].shape[1]

# PPL base
print("Base PPL...")
ppl_base_list = []
for i in tqdm(range(0, len(texts), batch_size), desc="Base PPL"):
    batch = texts[i:i+batch_size]
    enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
        ppl_base_list.append(torch.exp(out.loss).item())
    torch.cuda.empty_cache()
ppl_base = np.mean(ppl_base_list)

# PPL HL
print("HL PPL...")
ppl_hl_list = []
for i in tqdm(range(0, len(hl_texts), batch_size), desc="HL PPL"):
    batch_hl = hl_texts[i:i+batch_size]
    enc_hl = tokenizer(batch_hl, truncation=True, max_length=128, padding=True, return_tensors='pt')
    enc_hl = {k: v.to(device) for k, v in enc_hl.items()}
    with torch.no_grad():
        out = model(**enc_hl, labels=enc_hl['input_ids'])
        ppl_hl_list.append(torch.exp(out.loss).item())
    torch.cuda.empty_cache()
ppl_hl = np.mean(ppl_hl_list)

res = {
    'qwen_base_tokens_avg': base_tokens_avg,
    'tik_avg': tik_avg,
    'gpt2_avg': gpt2_avg,
    'hl_tokens_avg': hl_tokens_avg,
    'savings_vs_qwen': (1 - hl_tokens_avg / base_tokens_avg) * 100,
    'savings_vs_tik': (1 - hl_tokens_avg / tik_avg) * 100,
    'base_tokenize_s': base_tokenize_s,
    'hl_build_s': hl_build_s,
    'hl_encode_s': hl_encode_s,
    'hl_tokenize_s': hl_tokenize_s,
    'base_ppl': ppl_base,
    'hl_ppl': ppl_hl,
    'ppl_change_pct': (ppl_hl - ppl_base) / ppl_base * 100
}
wandb.log(res)
print("Results:", res)

wandb.finish()
print("Multi baseline done! Check wandb for charts.")
