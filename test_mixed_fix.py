#!/usr/bin/env python3
"""Test that mixed languages work correctly."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

# Test cases that should previously fail
test_cases = [
    ('Hello你好世界こんにちは', 'Mixed: EN + ZH + JA'),
    ('你好Hello世界', 'Mixed: ZH + EN + ZH'),
    ('こんにちは你好', 'Mixed: JA + ZH'),
    ('あんにちは世界', 'JA hiragana + ZH kanji'),
]

print("=" * 70)
print("MIXED LANGUAGE HANDLING TEST (Fixed)")
print("=" * 70)

for test_input, description in test_cases:
    print(f'\n{description}')
    print(f'Input: {test_input!r}')
    
    try:
        result = t.encode(test_input)
        print(f'Output: {result}')
        
        # Verify structure
        import re
        langs_found = re.findall(r'\[([a-z原]+)\]', result)
        print(f'Languages detected: {langs_found}')
    except Exception as e:
        print(f'ERROR: {e}')
