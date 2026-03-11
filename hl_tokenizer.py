import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import string
import jieba
from googletrans import Translator
import langdetect
from janome.tokenizer import Tokenizer

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.hl_vocab: Dict[str, str] = {}
        self.variant_rev: Dict[str, str] = {}
        self.zh_bases: set[str] = set()
        self.vocab_size = vocab_size
        self.concepts = self._init_concepts()
        # Boost concepts in jieba
        for word_list in self.concepts.values():
            for w in word_list:
                if re.match(r'^[\\u4e00-\\u9fff]+$', w):
                    jieba.add_word(w, 10000)
        self.hash_fallback = True
        self.translator = Translator()
        self.ja_tokenizer = Tokenizer()
        self.trans_cache: Dict[str, str] = {}
        self.rev_cache: Dict[tuple[str, str], str] = {}
        self._load_models()

    def _load_models(self):
        print("Hyper-Language Tokenizer v2: zh self-ID [HL你好], variants [HL你好:en]")

    def _detect_lang(self, text: str) -> str:
        text = text.strip()
        if len(text) < 6:
            return self._guess_lang(text)
        try:
            ld_lang = langdetect.detect(text)
            if ld_lang.startswith('zh'):
                return 'zh'
            return ld_lang
        except:
            return self._guess_lang(text)

    def _guess_lang(self, w: str) -> str:
        w = w.lower().strip()
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
        # Segment into language-specific candidates first
        candidates = re.findall(r'([\u4e00-\u9fff]+)|([a-zA-Z\u00C0-\u024F]+)|([\u3040-\u30ff\u31f0-\u31ff]+)|([^ \t\n\r\f\v\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]{1,})', text, re.UNICODE)
        words = []
        for groups in candidates:
            cand = ''.join([g for g in groups if g]).strip()
            if len(cand) <= 1 or re.match(r'^[^\w\u4e00-\u9fff]+$', cand):  # Skip short pure punct
                continue
            lang = self._guess_lang(cand)
            if lang == 'zh':
                subwords = jieba.lcut(cand)
                words.extend([sw.strip() for sw in subwords if len(sw) > 1])
            elif lang == 'ja':
                tokens = self.ja_tokenizer.tokenize(cand)
                words.extend([token.surface for token in tokens if len(token.surface) > 1 and token.part_of_speech.split(',')[0] not in ['記号', '助詞', '助動詞']])
            else:
                # Lower English-like, preserve others
                if re.match(r'^[a-zA-Z]+$', cand):
                    words.append(cand.lower())
                else:
                    words.append(cand)
        return words

    def _translate_batch(self, words: List[str]) -> Dict[str, str]:
        translations = {}
        for w in words:
            try:
                trans = self.translator.translate(w, dest='zh-CN')
                translations[w] = trans.text.strip()
            except Exception:
                translations[w] = w
        return translations

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

        for w in unique_words:
            if w in zh_map:
                self.trans_cache[w] = zh_map[w]

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
        zh_words = []
        non_zh_words = []
        for w in words:
            if self._is_non_zh(w):
                non_zh_words.append(w)
            else:
                zh_words.append(w)
        hl_words = [f'[HL{w}]' for w in zh_words]
        if non_zh_words:
            nonzh_text = ' '.join(non_zh_words)
            lang = self._detect_lang(nonzh_text)
            if lang == 'unk':
                hl_words += [f'[H{hashlib.md5(w.encode()).hexdigest()[:4]}]' if self.hash_fallback else w for w in non_zh_words]
            else:
                for w in non_zh_words:
                    if w in self.trans_cache:
                        zh_rep = self.trans_cache[w]
                    else:
                        try:
                            trans = self.translator.translate(w, dest='zh-CN')
                            zh_rep = trans.text.strip()
                            self.trans_cache[w] = zh_rep
                        except:
                            zh_rep = w
                    hl_words.append(f'[HL{zh_rep}:{lang}]')
        return ' '.join(hl_words)

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        decoded = []
        for w in words:
            m = re.match(r'\[HL([^\]:]+)(?::([a-z]{2}))?\]', w)
            if m:
                base = m.group(1)
                lang = m.group(2)
                if lang:
                    key = (base, lang)
                    if key in self.rev_cache:
                        orig = self.rev_cache[key]
                    else:
                        try:
                            orig = self.translator.translate(base, src='zh-CN', dest=lang).text.strip()
                            self.rev_cache[key] = orig
                        except:
                            orig = base
                    decoded.append(orig)
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
    tokenizer = HLTokenizer(vocab_size=1000)
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
