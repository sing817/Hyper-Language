#!/usr/bin/env python3
"""Test raw NLLB output."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-1.3B"
device = 0 if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.to(device)
model.eval()

lang_map = {
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'en': 'eng_Latn',
}

tests = [
    ("HelloWorld", "en", "zh"),
    ("Hello", "en", "zh"),
    ("こんにちは", "ja", "zh"),
    ("hello hello hello", "en", "zh"),  # Test repetition
]

for text, src, tgt in tests:
    src_code = lang_map[src]
    tgt_code = lang_map[tgt]
    tokenizer.src_lang = src_code
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            max_new_tokens=10,  # Very short
            num_beams=1,
            repetition_penalty=5.0,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f'{text:20} ({src}→{tgt}): {decoded!r}')
