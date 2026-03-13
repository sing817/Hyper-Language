#!/usr/bin/env python3
"""Test Japanese kanji to kana conversion."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    'こんにちは',                      # Pure hiragana
    'こんにちは世界',                   # Hiragana + kanji
    '日本語',                          # Pure kanji (Japanese)
    'ありがとうございます',             # Pure hiragana
    'ありがとう御座います',            # Hiragana + kanji mix
    'Hello日本語World',               # Mixed English + Japanese kanji + Japanese kana
]

print("=" * 70)
print("JAPANESE KANJI TO KANA CONVERSION TEST")
print("=" * 70)

for test_input in test_cases:
    print(f'\nInput: {test_input!r}')
    result = t.encode(test_input)
    print(f'Output: {result}')
