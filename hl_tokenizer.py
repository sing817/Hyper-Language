import re
import jieba
from collections import Counter
from typing import Dict, List
from difflib import SequenceMatcher

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000, base_lang: str = 'zh'):
        self.vocab: Dict[str, str] = {}  # word -> token
        self.reverse_vocab: Dict[str, str] = {}  # token -> word (main rep)
        self.vocab_size = vocab_size
        self.base_lang = base_lang
        self.id_counter = 1
        self.zh_vocab = {}  # zh specific
        self.zh_synonyms = self._init_synonyms()  # en/fr/ja -> zh

    def _init_synonyms(self) -> Dict[str, str]:
        # 擴充此 dict (手動/API/embeddings)
        return {
            'apple': '蘋果', 'world': '世界', 'time': '時間', 'dog': '狗',
            'cat': '貓', 'book': '書', 'computer': '電腦', 'hello': '你好',
            'good': '好', 'big': '大', 'small': '小', 'house': '房子'
        }

    def _is_zh(self, text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def _tokenize_non_zh(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text, re.UNICODE)

    def build_vocab(self, texts: List[str], top_k: int = None):
        # 1. zh 優先: jieba + freq
        zh_texts = [t for t in texts if self._is_zh(t)]
        zh_words = []
        for text in zh_texts:
            zh_words.extend(jieba.cut(text))
        zh_counts = Counter(zh_words)
        top_zh = zh_counts.most_common(self.vocab_size // 2)
        for i, (w, _) in enumerate(top_zh):
            token = f'[Z{i+1:04d}]'
            self.zh_vocab[w] = token
            self.vocab[w] = token
            self.reverse_vocab[token] = w
            self.id_counter += 1

        # 2. 非 zh: 借 zh synonyms 或 新 ID (sim match)
        non_zh_words = []
        for text in texts:
            if self._is_zh(text): continue
            words = self._tokenize_non_zh(text)
            non_zh_words.extend(words)
        non_zh_counts = Counter(w.lower() for w in non_zh_words)
        candidates = non_zh_counts.most_common(top_k or self.vocab_size // 2)

        for word, _ in candidates:
            word_lower = word.lower()
            # 借 zh
            zh_eq = self.zh_synonyms.get(word_lower, None)
            if zh_eq and zh_eq in self.zh_vocab:
                token = self.zh_vocab[zh_eq]
            else:
                # sim match to zh_vocab
                best_match = max(self.zh_vocab.keys(), key=lambda zw: SequenceMatcher(None, word_lower, zw.lower()).ratio())
                sim = SequenceMatcher(None, word_lower, best_match.lower()).ratio()
                if sim > 0.7:  # threshold
                    token = self.zh_vocab[best_match]
                else:
                    token = f'[ID{self.id_counter:04d}]'
                    self.id_counter += 1
            self.vocab[word_lower] = token
            if token not in self.reverse_vocab:
                self.reverse_vocab[token] = word

        print(f"Built: {len(self.zh_vocab)} zh IDs, total vocab {len(self.vocab)}")

    def encode(self, text: str) -> str:
        if self._is_zh(text):
            words = jieba.cut(text)
            hl_words = [self.vocab.get(''.join(w), ''.join(w)) for w in words]  # jieba returns tuples?
        else:
            words = self._tokenize_non_zh(text)
            hl_words = [self.vocab.get(w.lower(), w) for w in words]
        hl_text = re.sub(r'\s+', ' ', ' '.join(hl_words).strip())
        return hl_text

    def decode(self, hl_text: str) -> str:
        for token, word in self.reverse_vocab.items():
            hl_text = hl_text.replace(token, word)
        return hl_text

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_tokens = len(re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', orig))  # better for CJK
        comp_tokens = len(re.findall(r'\[Z\d+\]|\[ID\d+\]|\b\w+\b|[\u4e00-\u9fff]+', compressed))
        return comp_tokens / orig_tokens if orig_tokens else 1.0

if __name__ == '__main__':
    tokenizer = HLTokenizer(vocab_size=50)
    samples = [
        "Hello world apple. 你好世界 蘋果。",  # mix
        "The quick brown fox jumps over the lazy dog. 快速棕狐狸。",
        "Pomme rouge dans le jardin."  # fr apple
    ]
    tokenizer.build_vocab(samples)
    print("Vocab sample:", list(tokenizer.vocab.items())[:10])
    for text in samples:
        orig_len = len(re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', text))
        comp = tokenizer.encode(text)
        ratio = tokenizer.compress_ratio(text, comp)
        decoded = tokenizer.decode(comp)
        print(f"\nOrig ({orig_len}): {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded}")
