#!/usr/bin/env python3
"""Test simplified Chinese conversion for native Chinese text."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    '你好世界',           # Pure simplified
    '你好世界',           # Simplified (same as above)
    'HelloWorld你好世界こんにちは',  # Mixed
    'Hello你好',          # English + simplified
]

print("=" * 70)
print("SIMPLIFIED CHINESE CONVERSION TEST")
print("=" * 70)

for test_input in test_cases:
    print(f'\nInput: {test_input!r}')
    result = t.encode(test_input)
    print(f'Output: {result}')
