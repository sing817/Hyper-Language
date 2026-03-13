#!/usr/bin/env python3
"""Debug script family classification."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_chars = [
    ('H', 'latin'),
    ('你', 'hanzi'),
    ('こ', 'kana'),
]

for char, expected in test_chars:
    group = t._get_script_group(char)
    o = ord(char)
    ok = 'OK' if group.startswith(expected[0]) or expected in group else 'FAIL'
    print(f'{char}: {group:8} (U+{o:04X}) [{ok}]')
