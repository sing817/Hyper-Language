import re
import jieba
from collections import Counter
from typing import Dict, List
from difflib import SequenceMatcher
import string

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000, base_lang: str = 'zh'):
        self.vocab: Dict[str, str] = {}  # lower_word -> token
        self.reverse_vocab: Dict[str, str] = {}  # token -> orig_word sample
        self.vocab_size = vocab_size
        self.base_lang = base_lang
        self.id_counter = 1
        self.zh_vocab = {}  # zh_word -> token
        self.zh_synonyms = self._init_synonyms()
        self.stopwords_zh = set(['的', '了', '在', '是', '有', '和', '就', '不', '一', '了', '上', '也', '很', '到', '说', '去', '她', '他', '这', '个', '们', '来', '为', '会', '中', '能', '出', '要', '下', '并', '以', '自', '之', '最', '过', '后', '又'])
        self.punct = set(string.punctuation) | set('。 ，')

    def _init_synonyms(self) -> Dict[str, str]:
        return {
            # en -> zh
            'apple': '蘋果', 'world': '世界', 'time': '時間', 'dog': '狗', 'cat': '貓', 'book': '書', 
            'computer': '電腦', 'hello': '你好', 'good': '好', 'big': '大', 'small': '小', 'house': '房子',
            'quick': '快速', 'brown': '棕色', 'fox': '狐狸', 'jumps': '跳', 'over': '過', 'lazy': '懶',
            'the': '這', 'and': '和', 'of': '的', 'to': '到', 'a': '一', 'in': '在', 'that': '那',
            # fr -> zh
            'pomme': '蘋果', 'rouge': '紅', 'dans': '在', 'le': '這', 'jardin': '花園',
            # ja -> zh (音/義)
            '世界': '世界', '犬': '狗', '猫': '貓', '本': '書', '速い': '快速', '茶色': '棕色',
            '狐': '狐狸'
        }

    def _is_zh(self, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _tokenize_non_zh(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\w*\b', text, re.UNICODE)

    def build_vocab(self, texts: List[str], top_k: int = None):
        # 1. zh 優先
        zh_words = []
        for text in texts:
            if self._is_zh(text):
                words = jieba.lcut(text)
                zh_words.extend([w for w in words if len(w) > 1 and w not in self.stopwords_zh and w not in self.punct])
        zh_counts = Counter(zh_words)
        top_zh = zh_counts.most_common(self.vocab_size // 2)
        for i, (w, _) in enumerate(top_zh):
            token = f'[Z{i+1:04d}]'
            self.zh_vocab[w] = token
            self.vocab[w.lower()] = token
            self.reverse_vocab[token] = w
            self.id_counter += 1

        # 2. 非 zh borrow or new
        non_zh_words = []
        for text in texts:
            if not self._is_zh(text):
                words = self._tokenize_non_zh(text)
                non_zh_words.extend([w for w in words if len(w) > 1])
        non_zh_counts = Counter(w.lower() for w in non_zh_words)
        candidates = non_zh_counts.most_common((top_k or self.vocab_size) // 2)

        for word_lower, _ in candidates:
            orig_word = non_zh_words[non_zh_words.index(word_lower)] if word_lower in non_zh_words else word_lower
            # borrow zh
            zh_eq = self.zh_synonyms.get(word_lower, None)
            token = self.zh_vocab.get(zh_eq, None)
            if not token:
                # sim match
                best_zh = max(self.zh_vocab, key=lambda zw: SequenceMatcher(None, word_lower, zw.lower()).ratio(), default='')
                sim_score = SequenceMatcher(None, word_lower, best_zh.lower()).ratio()
                if sim_score > 0.6:
                    token = self.zh_vocab[best_zh]
                    print(f'Borrow: {word_lower} -> {best_zh} (sim {sim_score:.2f})')
            if not token:
                token = f'[ID{self.id_counter:04d}]'
                self.id_counter += 1
            self.vocab[word_lower] = token
            self.reverse_vocab[token] = orig_word

        print(f"Built: {len(self.zh_vocab)} zh IDs, total {len(self.vocab)} vocab")

    def encode(self, text: str) -> str:
        if self._is_zh(text):
            words = jieba.lcut(text)
            hl_words = [self.vocab.get(w.lower(), w) for w in words if len(w) > 1]
        else:
            words = self._tokenize_non_zh(text)
            hl_words = [self.vocab.get(w.lower(), w) for w in words if len(w) > 1]
        return ' '.join(hl_words)

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        decoded_words = [self.reverse_vocab.get(w, w) for w in words]
        return ''.join(decoded_words)  # no spaces in orig

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_words = len(re.findall(r'\b\w+\w*\b|[\u4e00-\u9fff]{1,}', orig))
        comp_words = len(compressed.split())
        return comp_words / orig_words if orig_words else 1.0

if __name__ == '__main__':
    tokenizer = HLTokenizer(vocab_size=100)
    samples = [
        "Hello world apple pie. 你好世界 蘋果派。",
        "The quick brown fox jumps over the lazy dog. 快速的棕狐狸跳過懶狗。",
        "Pomme rouge dans le jardin. 紅蘋果在花園。",
        "速い茶色の狐が怠惰な犬を飛び越える。"  # ja
    ]
    tokenizer.build_vocab(samples)
    print("Vocab sample:", list(tokenizer.vocab.items())[:10])
    for text in samples:
        orig_len = len(re.findall(r'\b\w+\w*\b|[\u4e00-\u9fff]{1,}', text))
        comp = tokenizer.encode(text)
        ratio = tokenizer.compress_ratio(text, comp)
        decoded = tokenizer.decode(comp)
        print(f"\nOrig ({orig_len}): {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded}")
