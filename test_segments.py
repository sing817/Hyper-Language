#!/usr/bin/env python3
"""Test lang_segments function only."""
import sys
sys.path.insert(0, '.')

from hl_tokenizer import HLTokenizer
t = HLTokenizer()

# Test segmentation (no jieba/NLLB needed)
tests = [
    'こんにちは世界',           # Pure Japanese
    'Hello World',              # Pure English  
    '你好世界',                 # Pure Chinese
    'HelloWorld你好世界こんにちは',  # No spaces, mixed
    'Hello World 你好世界 こんにちは',  # With spaces
]

for test in tests:
    segments = t.lang_segments(test)
    print(f'Input: {test!r}')
    print(f'Segments: {segments}')
    print()
