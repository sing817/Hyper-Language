#!/usr/bin/env python3
"""Test traditional to simplified Chinese conversion."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    ('你好世界', '简体中文'),
    ('你好世界', '繁体中文'),
    ('你好', '单字繁体'),
    ('繁體中文', '纯繁体'),
    ('HelloWorld你好世界こんにちは', '混合含繁体'),
]

print("=" * 70)
print("TRADITIONAL TO SIMPLIFIED CHINESE CONVERSION TEST")
print("=" * 70)

for test_input, description in test_cases:
    print(f'\n{description}:')
    print(f'  Input:  {test_input!r}')
    result = t.encode(test_input)
    print(f'  Output: {result}')
