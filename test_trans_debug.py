#!/usr/bin/env python3
"""Debug translate_pending step."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

# Input with language blocks
test_input = "[en]HelloWorld[/en][HL你好][HL世界][ja]こんにちは[/ja]"
print(f'Input: {test_input}')
print('\nCalling translate_pending...\n')

result = t.translate_pending(test_input)
print(f'\nOutput: {result}')
