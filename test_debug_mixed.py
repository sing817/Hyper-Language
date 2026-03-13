#!/usr/bin/env python3
"""Debug mixed language handling."""
from hl_tokenizer import HLTokenizer

t = HLTokenizer()

test_cases = [
    'Hello你好世界こんにちは',
    '你好Hello世界',
    'こんにちは你好',
    'Hello你好こんにちは',
]

print("=" * 70)
print("DEBUG MIXED LANGUAGE HANDLING")
print("=" * 70)

for test_input in test_cases:
    print(f'\n--- Input: {test_input!r} ---')
    
    # Debug segments
    print("Segments:")
    segments = t.lang_segments(test_input)
    for i, seg in enumerate(segments):
        print(f"  {i}: {seg!r}")
        # Check for kana/hanzi
        has_kana = any(0x3040 <= ord(c) <= 0x30FF for c in seg)
        has_hanzi = any(0x4E00 <= ord(c) <= 0x9FFF for c in seg)
        print(f"     has_kana={has_kana}, has_hanzi={has_hanzi}")
    
    print("\nEncoded:")
    result = t.encode(test_input)
    print(f"  {result}")
