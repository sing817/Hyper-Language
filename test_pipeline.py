#!/usr/bin/env python3
"""Test full pipeline encode_full()"""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_input = 'HelloWorld你好世界こんにちは'
print(f'Input: {test_input!r}')

# Step 1: Encode  
pending = t.encode(test_input)
print(f'1. Encode (pending): {pending}')

# Step 2: Translate pending
print('\n2. Translate pending...')
try:
    transed = t.translate_pending(pending)
    print(f'   Trans: {transed}')
except Exception as e:
    print(f'   ERROR: {e}')
    transed = pending

# Step 3: Finalize
print('\n3. Finalize...')
try:
    final = t.finalize(transed)
    print(f'   Final: {final}')
except Exception as e:
    print(f'   ERROR: {e}')
