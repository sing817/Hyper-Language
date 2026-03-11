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
batch_size = 64
hl_vocab_size = 1000
MAX_LEN = 128  # 統一

wandb.init(project='hyper-language', name='multi_base_FIXED (trunc all)', config={'mode': 'multi_base_fixed', 'max_samples': max_samples, 'max_len': MAX_LEN})

model_size = 'Qwen/Qwen2-1.5B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

tokenizer = AutoTokenizer.from_pretrained(model_size)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_size, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

# Load texts (same)
samples_per_lang = max_samples // 4
dataset_configs = [
    ('allenai/c4', 'en', 'text', 'en'), ('allenai/c4', 'zh', 'text', 'zh'),
    ('allenai/c4', 'ja', 'text', 'ja'), ('allenai/c4', 'fr', 'text', 'fr')
]
texts = []
for ds_name, subds, field, lang in dataset_configs:
    print(f'Loading {lang}...')
    ds = load_dataset(ds_name, subds, split='train', streaming=True)
    lang_texts = [ex[field].strip() for ex in ds if (text := ex[field].strip()) and len(text) > 20][:samples_per_lang]
    texts.extend(lang_texts)
    print(f'Loaded {len(lang_texts)} {lang}')
random.shuffle(texts)
texts = texts[:max_samples]
print(f'Total: {len(texts)}')

# Qwen base
start = time.time()
encodings_base = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
base_tokenize_s = time.time() - start
qwen_avg = encodings_base['input_ids'].shape[1]

# Tik cl100k FIXED: trunc 128
tik_enc = tiktoken.get_encoding('cl100k_base')
tik_lens = [min(len(tik_enc.encode(t)), MAX_LEN) for t in tqdm(texts, desc='tik FIXED')]
tik_avg = np.mean(tik_lens)

# GPT2 FIXED: full trunc (no subsample approx bug)
gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
gpt2_lens = [len(gpt2_tok.encode(t, truncation=True, max_length=MAX_LEN)) for t in tqdm(texts, desc='gpt2 FIXED')]
gpt2_avg = np.mean(gpt2_lens)

# HL
hl_tok = HLTokenizer(vocab_size=hl_vocab_size)
hl_build_s = time.time()
hl_tok.build_vocab(texts)
hl_build_s = time.time() - hl_build_s
hl_texts = [hl_tok.encode(t) for t in tqdm(texts, desc='HL encode')]
hl_encode_s = time.time() - hl_encode_s  # wait, fix time

# HL Qwen tokenize
hl_tokenize_start = time.time()
encodings_hl = tokenizer(hl_texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
hl_tokenize_s = time.time() - hl_tokenize_start
hl_avg = encodings_hl['input_ids'].shape[1]

# PPL Qwen base
ppl_base_list = []
for i in tqdm(range(0, len(texts), batch_size), desc='Qwen PPL'):
    batch = texts[i:i+batch_size]
    enc = tokenizer(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
        ppl_base_list.append(torch.exp(out.loss).item())
    torch.cuda.empty_cache()
ppl_qwen = np.mean(ppl_base_list)

# PPL HL
ppl_hl_list = []
for i in tqdm(range(0, len(hl_texts), batch_size), desc='HL PPL'):
    batch_hl = hl_texts[i:i+batch_size]
    enc_hl = tokenizer(batch_hl, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
    enc_hl = {k: v.to(device) for k, v in enc_hl.items()}
    with torch.no_grad():
        out = model(**enc_hl, labels=enc_hl['input_ids'])
        ppl_hl_list.append(torch.exp(out.loss).item())
    torch.cuda.empty_cache()
ppl_hl = np.mean(ppl_hl_list)

res = {
    'qwen_avg': qwen_avg, 'tik_avg': tik_avg, 'gpt2_avg': gpt2_avg, 'hl_avg': hl_avg,
    'savings_vs_qwen': (1 - hl_avg / qwen_avg)*100,
    'savings_vs_tik': (1 - hl_avg / tik_avg)*100,
    'savings_vs_gpt2': (1 - hl_avg / gpt2_avg)*100,
    'base_tokenize_s': base_tokenize_s, 'hl_build_s': hl_build_s, 'hl_encode_s': hl_encode_s, 'hl_tokenize_s': hl_tokenize_s,
    'ppl_qwen': ppl_qwen, 'ppl_hl': ppl_hl, 'ppl_change_pct': (ppl_hl - ppl_qwen)/ppl_qwen *100
}
wandb.log(res)
print('FIXED Results:', res)
wandb.finish()
print('Fixed done! Tik/GPT2 trunc OK now.')