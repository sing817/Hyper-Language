#!/usr/bin/env python3
"""Debug kakasi output format."""
import pykakasi

kakasi = pykakasi.kakasi()
result = kakasi.convert('日本語')

print("Result type:", type(result))
print("Raw result:", result)
print("\nIterating:")
for item in result:
    print("  Item:", item, "| Type:", type(item))
    if isinstance(item, dict):
        print("    Keys:", item.keys())
        print("    Content:", item)
