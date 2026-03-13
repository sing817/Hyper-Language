#!/usr/bin/env python3
"""Test improved encode function."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_input = 'HelloWorld你好世界こんにちは'
print(f'Input: {test_input!r}')
print()

result = t.encode(test_input)
print(f'Encode output:\n{result}')
print()
print(f'Pretty print:')
import re
# Split on language blocks
parts = re.findall(r'\[/?[a-z]{2}\]|(\[HL[^\]]*\])', result)
for p in parts:
    if p:
        print(f'  {p}')
