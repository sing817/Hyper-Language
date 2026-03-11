import re
import json
from collections import Counter, defaultdict
from typing import Dict, List
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import string

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.hl_vocab: Dict[str, str] = {}
        self.variant_rev: Dict[str, str] = {}
        self.zh_bases: set[str] = set()
        self.vocab_size = vocab_size
        self.concepts = self._init_concepts()
        self.hash_fallback = True
        self._load_models()

    def _load_models(self):
        print("Hyper-Language Tokenizer v2: zh self-ID [HL你好], variants [HL你好:en]")

    def _guess_lang(self, w: str) -> str:
        w = w.lower()
        if len(w) < 2:
            return 'unk'
        if re.match(r'^[a-z]+$', w):
            return 'en'
        if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', w):
            return 'ja'
        if re.search(r'[àâäéèêëîïôöùûüÿçñ]', w):
            return 'fr'
        if re.search(r'[áéíóúñ]', w):
            return 'es'
        if re.match(r'^[\u4e00-\u9fff]+$', w):
            return 'zh'
        return 'unk'

    def _is_non_zh(self, word: str) -> bool:
        return not bool(re.search(r'[\u4e00-\u9fff]', word))

    def _init_concepts(self) -> Dict[str, List[str]]:
        return {
            'apple': ['苹果', 'apple', 'pomme', 'りんご', 'apfel'],
            'world': ['世界', 'world', 'monde', 'welt'],
            'hello': ['你好', 'hello', 'bonjour', 'こんにちは', 'hola'],
            'dog': ['狗', 'dog', 'chien', '犬', 'hund', 'perro'],
            'fox': ['狐狸', 'fox'],
            'jump': ['跳', 'jumps', 'jump', '跳跃'],
            'brown': ['棕色', 'brown'],
            'quick': ['快速', 'quick'],
            'lazy': ['懒惰', 'lazy'],
        }

    def _tokenize_cross(self, text: str) -> List[str]:
        text = re.sub(r'[。！？，。!?;:\\' + string.punctuation + ']', ' ', text)
        if re.search(r'[\u4e00-\u9fff]', text):
            import jieba
            raw_words = jieba.lcut(text)
        else:
            raw_words = re.findall(r'\b\w+\b', text)
        words = []
        for w in raw_words:
            if len(w) <= 1 or w.isspace():
                continue
            if self._is_non_zh(w):
                words.append(w.lower())
            else:
                words.append(w)
        return words

    def _translate_batch(self, words: List[str]) -> Dict[str, str]:
        return {w: w for w in words}  # pivot self

    def build_vocab(self, texts: List[str]):
        print("Building v2 vocab...")
        # Manual
        for _, words in self.concepts.items():
            zh_words = [w for w in words if self._guess_lang(w) == 'zh']
            zh_rep = zh_words[0] if zh_words else words[0]
            token = f'[HL{zh_rep}]'
            self.zh_bases.add(zh_rep)
            self.hl_vocab[zh_rep] = token
            for w in words:
                if self._guess_lang(w) == 'zh':
                    continue
                lang = self._guess_lang(w)
                if lang == 'unk':
                    continue
                vtoken = f'[HL{zh_rep}:{lang}]'
                self.variant_rev[vtoken] = w
                self.hl_vocab[w] = vtoken

        all_words = []
        for text in tqdm(texts, desc="Candidates"):
            words = self._tokenize_cross(text)
            all_words.extend(words)

        unique_words = [w for w in set(all_words) if w not in self.hl_vocab]
        non_zh_words = [w for w in unique_words if self._is_non_zh(w)]
        print(f"New: {len(non_zh_words)} non-zh / {len(unique_words)}")
        zh_map = {w: w for w in unique_words if not self._is_non_zh(w)}
        for i in tqdm(range(0, len(non_zh_words), 5), desc="Pivot"):
            batch = non_zh_words[i:i+20]
            zh_map.update(self._translate_batch(batch))

        zh_counts = Counter(zh_map[w] for w in unique_words)
        top_zh = zh_counts.most_common(self.vocab_size)

        for zh_rep, _ in top_zh:
            if len(self.zh_bases) >= self.vocab_size:
                break
            if zh_rep in self.zh_bases:
                continue
            self.zh_bases.add(zh_rep)
            token = f'[HL{zh_rep}]'
            self.hl_vocab[zh_rep] = token
            matching_orig = [w for w in unique_words if zh_map[w] == zh_rep]
            for w in matching_orig:
                lang = self._guess_lang(w)
                if lang == 'zh' or lang == 'unk':
                    self.hl_vocab[w] = token
                else:
                    vtoken = f'[HL{zh_rep}:{lang}]'
                    self.variant_rev[vtoken] = w
                    self.hl_vocab[w] = vtoken

        print(f"Vocab: {len(self.zh_bases)} zh bases, {len(self.hl_vocab)} entries")

    def encode(self, text: str) -> str:
        words = self._tokenize_cross(text)
        hl_words = []
        for w in words:
            token = self.hl_vocab.get(w)
            if not token and self.hash_fallback:
                h = hashlib.md5(w.encode()).hexdigest()[:4]
                token = f'[H{h}]'
            hl_words.append(token or w)
        return ' '.join(hl_words)

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        decoded = []
        for w in words:
            m = re.match(r'\[HL([^\]:]+)(?::([a-z]{2}))?\]', w)
            if m:
                base = m.group(1)
                lang = m.group(2)
                if lang and w in self.variant_rev:
                    decoded.append(self.variant_rev[w])
                else:
                    decoded.append(base)
            elif w.startswith('[H'):
                decoded.append(w)
            else:
                decoded.append(w)
        return re.sub(r' +', ' ', ' '.join(decoded)).strip()

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_words = len(self._tokenize_cross(orig))
        comp_hl = sum(1 for w in compressed.split() if w.startswith('[HL'))
        return (orig_words - comp_hl) / orig_words if orig_words else 1.0

if __name__ == '__main__':
    tokenizer = HLTokenizer(vocab_size=200)
    samples = [
        "你好世界 蘋果。Hello world apple. ",
        "快速棕色狐狸跳過懶狗。The quick brown fox jumps over the lazy dog.",
        "Pomme bonjour chien.",
        "速い狐犬りんご。"
    ]
    tokenizer.build_vocab(samples)
    print("Vocab sample:", list(tokenizer.hl_vocab.items())[:10])
    for text in samples:
        orig_len = len(tokenizer._tokenize_cross(text))
        comp = tokenizer.encode(text)
        ratio = tokenizer.compress_ratio(text, comp)
        decoded = tokenizer.decode(comp)
        print(f"\\nOrig ({orig_len}): {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded}")
