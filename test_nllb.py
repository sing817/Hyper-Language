#!/usr/bin/env python3
"""Debug NLLB translation directly."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

# Test direct NLLB translation
test_cases = [
    ("HelloWorld", "en", "zh"),
    ("こんにちは", "ja", "zh"),
    ("你好", "zh", "en"),
]

print("Testing direct NLLB translation:")
for text, src_lang, tgt_lang in test_cases:
    try:
        result = t._translate([text], src_lang, tgt_lang)
        print(f'{text!r:20} ({src_lang}→{tgt_lang}): {result[0]!r}')
    except Exception as e:
        print(f'{text!r:20} ({src_lang}→{tgt_lang}): ERROR - {e}')
