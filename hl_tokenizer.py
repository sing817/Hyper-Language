import re
from collections import Counter, defaultdict
from typing import Dict, List
import hashlib

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.hl_vocab: Dict[str, str] = {}  # lower_word → [HLxxx]
        self.reverse_hl: Dict[str, str] = {}  # [HLxxx] → rep_word (abstract rep)
        self.vocab_size = vocab_size
        self.hl_counter = 1
        self.concepts = self._init_concepts()  # 原創: 多 lang → 抽象概念 ID
        self.hash_fallback = True  # 罕見 hash ID

    def _init_concepts(self) -> Dict[str, List[str]]:
        # 原創共享: 概念 → multi-lang words (手動/後 auto cluster)
        return {
            'fruit_apple': ['apple', '蘋果', 'pomme', 'りんご', 'apfel'],
            'world': ['world', '世界', 'monde', '世界', 'welt'],
            'hello': ['hello', '你好', 'bonjour', 'こんにちは'],
            'dog': ['dog', '狗', 'chien', '犬', 'hund'],
            'quick': ['quick', '快速', 'rapide', '速い'],
            'brown': ['brown', '棕色', 'brun', '茶色'],
            'fox': ['fox', '狐狸', 'renard', '狐'],
            'lazy': ['lazy', '懶', 'paresseux', '怠惰'],
            'jump': ['jumps', '跳', 'saute', '飛び越える'],
            'garden': ['garden', '花園', 'jardin', '庭'],
            # 加更多 (WordNet + Translate, 1k 易)
        }

    def _tokenize_cross(self, text: str) -> List[str]:
        if re.search(r'[\u4e00-\u9fff]', text):
            import jieba
            return jieba.lcut(text)
        return re.findall(r'\b\w+\b', text)

    def build_vocab(self, texts: List[str]):
        # 1. 原創共享: concepts 先 assign
        for concept, words in self.concepts.items():
            token = f'[HL{self.hl_counter:03d}]'
            for w in words:
                self.hl_vocab[w.lower()] = token
            self.reverse_hl[token] = concept  # abstract rep
            self.hl_counter += 1
            print(f'Shared concept {concept}: {words} → {token}')

        # 2. 剩餘 freq top (global tokenize)
        all_words = []
        for text in texts:
            words = self._tokenize_cross(text)
            all_words.extend([w for w in words if w.lower() not in self.hl_vocab and len(w) > 1])
        counts = Counter(w.lower() for w in all_words)
        top_remaining = counts.most_common(self.vocab_size - self.hl_counter)

        for w_lower, _ in top_remaining:
            if self.hl_counter > self.vocab_size: break
            token = f'[HL{self.hl_counter:03d}]'
            self.hl_vocab[w_lower] = token
            self.reverse_hl[token] = all_words[all_words.index(w_lower)] if w_lower in all_words else w_lower
            self.hl_counter += 1

        # 3. 罕見 hash fallback (optional)
        print(f"Built {len(self.hl_vocab)} HL IDs (concepts + freq)")

    def encode(self, text: str) -> str:
        words = self._tokenize_cross(text)
        hl_words = []
        for w in words:
            w_lower = w.lower()
            token = self.hl_vocab.get(w_lower)
            if not token and self.hash_fallback:
                h = hashlib.md5(w_lower.encode()).hexdigest()[:4]
                token = f'[H{h}]'  # short hash
            hl_words.append(token or w)
        return ' '.join(hl_words)

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        decoded = []
        for w in words:
            if w.startswith('[HL') or w.startswith('[H'):
                rep = self.reverse_hl.get(w, w)
                decoded.append(rep)
            else:
                decoded.append(w)
        return ' '.join(decoded)  # space for readability

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_words = len(re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', orig))
        comp_hl = sum(1 for w in compressed.split() if w.startswith('[H'))
        return (orig_words - comp_hl) / orig_words if orig_words else 1.0  # HL count as save

if __name__ == '__main__':
    tokenizer = HLTokenizer(vocab_size=50)
    samples = [
        "Hello world apple. 你好世界 蘋果。",
        "The quick brown fox jumps over the lazy dog. 快速棕狐狸跳過懶狗。",
        "Pomme rouge dans le jardin. Bonjour chien.",
        "速い茶色の狐が怠惰な犬を飛び越える。りんご。"
    ]
    tokenizer.build_vocab(samples)
    print("Vocab sample:", list(tokenizer.hl_vocab.items())[:10])
    for text in samples:
        orig_len = len(re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', text))
        comp = tokenizer.encode(text)
        ratio = tokenizer.compress_ratio(text, comp)
        decoded = tokenizer.decode(comp)
        print(f"\nOrig ({orig_len}): {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded}")
