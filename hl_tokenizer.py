import re
from collections import Counter
from typing import Dict, List

class HLTokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab: Dict[str, str] = {}
        self.reverse_vocab: Dict[str, str] = {}
        self.vocab_size = vocab_size
        self.id_counter = 1

    def build_vocab(self, texts: List[str], top_k: int = None):
        words = re.findall(r'\\b\\w+\\b', ' '.join(texts), re.UNICODE)
        word_counts = Counter(words)
        candidates = word_counts.most_common(top_k or self.vocab_size)
        for word, _ in candidates:
            token = f'[ID:{self.id_counter:04d}]'
            self.vocab[word.lower()] = token
            self.reverse_vocab[token] = word
            self.id_counter += 1

    def encode(self, text: str) -> str:
        hl_text = re.sub(r'\\b\\w+\\b', lambda m: self.vocab.get(m.group().lower(), m.group()), text, flags=re.IGNORECASE)
        hl_text = re.sub(r'\\s+', ' ', hl_text.strip())
        return hl_text

    def decode(self, hl_text: str) -> str:
        for token, word in self.reverse_vocab.items():
            hl_text = hl_text.replace(token, word)
        return hl_text

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_tokens = len(re.findall(r'\\b\\w+\\b', orig))
        comp_tokens = len(re.findall(r'\\[ID:\\d+\\]|\\b\\w+\\b', compressed))
        return comp_tokens / orig_tokens if orig_tokens else 1.0

if __name__ == '__main__':
    tokenizer = HLTokenizer()
    samples = [
        "Hello world apple. 你好世界 蘋果。",
        "The quick brown fox jumps over the lazy dog."
    ]
    tokenizer.build_vocab(samples, top_k=10)
    print("Vocab sample:", list(tokenizer.vocab.items())[:5])
    for text in samples:
        orig_len = len(re.findall(r'\\b\\w+\\b', text))
        comp = tokenizer.encode(text)
        ratio = tokenizer.compress_ratio(text, comp)
        decoded = tokenizer.decode(comp)
        accuracy = "Perfect" if decoded.lower() == text.lower() else "Lossy"
        print(f"\nOrig ({orig_len} tokens): {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded} ({accuracy})")
