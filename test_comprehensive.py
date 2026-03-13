#!/usr/bin/env python3
"""Comprehensive test of multilingual encode."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    'HelloWorld你好世界こんにちは',
    'Hello world',
    'こんにちは',
    'Bonjour',
    'gracias',
    '你好',
    'Thankyou',
    'sayounara',
]

print("=" * 70)
print("COMPREHENSIVE MULTILINGUAL TOKENIZER TEST")
print("=" * 70)

for test_input in test_cases:
    print(f'\nInput: {test_input!r}')
    result = t.encode(test_input)
    print(f'Output: {result}')
    print()
