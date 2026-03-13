#!/usr/bin/env python3
"""Test encode-decode round trip with consistent language wrappers."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    'HelloWorld',
    '你好世界',
    'こんにちは',
    'Hello你好',
]

print("=" * 70)
print("ENCODE-DECODE ROUND TRIP TEST")
print("=" * 70)

for test_input in test_cases:
    print(f'\n--- Input: {test_input!r} ---')
    encoded = t.encode(test_input)
    print(f'Encoded: {encoded}')
    
    try:
        decoded = t.decode(encoded)
        print(f'Decoded: {decoded!r}')
    except Exception as e:
        print(f'Decode ERROR: {e}')
