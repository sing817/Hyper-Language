#!/usr/bin/env python3
"""Test is_chinese detection."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

tests = [
    '您的世界. (欢迎访问)',
    '你好,我没有. !',
    '您好,我们有个朋友.',
    '你好,你不错. 您',
]

for text in tests:
    result = t.is_chinese(text)
    print(f'{text:30} → is_chinese: {result}')
