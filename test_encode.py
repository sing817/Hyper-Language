#!/usr/bin/env python3
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

# Test 1: Just encoding (no NLLB)
print('=== Test Encode Only (No NLLB) ===')
ja = 'こんにちは世界'
en = 'Hello World'
zh = '你好世界'

print(f'JA encode: {t.encode(ja)}')
print(f'EN encode: {t.encode(en)}')
print(f'ZH encode: {t.encode(zh)}')

# Test 2: Continuous mixed encoding
print('\n=== Test Continuous Mixed Encode ===')
mixed = 'HelloWorld你好世界こんにちは'
print(f'Mixed encode: {t.encode(mixed)}')

# Test 3: With spaces
print('\n=== Test with spaces ===')
mixed_space = 'Hello World 你好世界 こんにちは世界'
print(f'With spaces: {t.encode(mixed_space)}')
