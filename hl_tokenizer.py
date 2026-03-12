import re
from collections import Counter
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import jieba
import langdetect

class HLTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.zh_bases: set[str] = set()
        self.hl_vocab: Dict[str, str] = {}
        self.variant_rev: Dict[str, str] = {}
        self.trans_cache: Dict[str, tuple[str, str]] = {}  # orig -> (zh, lang)
        self.rev_cache: Dict[tuple[str, str], str] = {}  # (zh, lang) -> orig approx
        self.concepts = self._init_concepts()
        
        # Boost concepts in jieba
        for word_list in self.concepts.values():
            for w in word_list:
                if re.match(r'^[\\u4e00-\\u9fff]+$', w):
                    jieba.add_word(w, 10000)
        
        self.translate_pipe = None  # Lazy init NLLB
        self.device = 0 if torch.cuda.is_available() else -1
        self.lang_map = {
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'en': 'eng_Latn',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'de': 'deu_Latn',
            'ko': 'kor_Hang',
            # Add more as needed
        }
        print("Hyper-Language Tokenizer v3: NLLB-200 Pivot [快速:ja] lossless multi-lang compress")

    def _get_pipe(self):
        if self.translate_pipe is None:
            model_name = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="auto")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.translate_pipe = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                batch_size=64  # Large scale
            )
            print(f"NLLB loaded on {'CUDA' if self.device >= 0 else 'CPU'}")
        return self.translate_pipe

    def _detect_lang(self, text: str) -> str:
        text = text.strip()
        if len(text) < 3:
            return self._guess_lang(text)
        try:
            ld_lang = langdetect.detect(text)
            return 'zh' if ld_lang.startswith('zh') else ld_lang[:2]  # Normalize
        except:
            return self._guess_lang(text)

    def _guess_lang(self, text: str) -> str:
        text = text.lower().strip()
        if re.match(r'^[a-z]+$', text):
            return 'en'
        if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', text):
            return 'ja'
        if re.match(r'^[\u4e00-\u9fff]+$', text):
            return 'zh'
        return 'unk'

    def _translate(self, texts: List[str], src_lang: str, tgt_lang: str = 'zho_Hans') -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        pipe = self._get_pipe()
        src_code = self.lang_map.get(src_lang, 'und_Latn')
        results = pipe(texts, src_lang=src_code, tgt_lang=tgt_lang, max_length=512)
        return [r['translation_text'].strip() for r in results]

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

    def build_vocab(self, texts: List[str]):
        print("Building v3 NLLB vocab (sentence pivot)...")
        # Concepts first
        for _, words in self.concepts.items():
            zh_words = [w for w in words if self._guess_lang(w) == 'zh']
            if zh_words:
                zh_rep = zh_words[0]
                self.zh_bases.add(zh_rep)
                self.hl_vocab[zh_rep] = f'[HL{zh_rep}]'
                for w in words:
                    if self._guess_lang(w) != 'zh':
                        lang = self._guess_lang(w)
                        vtoken = f'[HL{zh_rep}:{lang}]'
                        self.variant_rev[vtoken] = w
                        self.hl_vocab[w] = vtoken

        zh_counts = Counter()
        non_zh_texts = []
        langs = []
        for text in tqdm(texts, desc="Pivot texts"):
            lang = self._detect_lang(text)
            if lang == 'zh':
                words = [w for w in jieba.lcut(text) if len(w) > 1]
            else:
                zh_trans = self._translate([text], lang)[0]
                self.trans_cache[text] = (zh_trans, lang)
                words = [w for w in jieba.lcut(zh_trans) if len(w) > 1]
                non_zh_texts.append(text)
                langs.append(lang)
            for w in words:
                zh_counts[w] += 1

        top_zh = zh_counts.most_common(self.vocab_size)
        for zh_rep, _ in top_zh:
            if zh_rep not in self.zh_bases:
                self.zh_bases.add(zh_rep)
                self.hl_vocab[zh_rep] = f'[HL{zh_rep}]'

        print(f"Vocab: {len(self.zh_bases)} zh bases, {len(self.hl_vocab)} entries")

    def encode(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ''
        lang = self._detect_lang(text)
        if lang == 'zh':
            words = [w for w in jieba.lcut(text) if len(w) > 1]
            return ' '.join([f'[{w}:zh]' for w in words])
        else:
            zh_trans = self._translate([text], lang)[0]
            self.trans_cache[text] = (zh_trans, lang)
            words = [w for w in jieba.lcut(zh_trans) if len(w) > 1]
            return ' '.join([f'[{w}:{lang}]' for w in words])

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        decoded = []
        for token in words:
            m = re.match(r'\[([^\]:]+):([a-z]+)\]', token)
            if m:
                zh_word, src_lang = m.groups()
                if src_lang == 'zh':
                    decoded.append(zh_word)
                else:
                    # Reverse: zh -> src_lang
                    rev_trans = self._translate([zh_word], 'zh', self.lang_map.get(src_lang, 'und_Latn'))[0]
                    decoded.append(rev_trans)
                    self.rev_cache[(zh_word, src_lang)] = rev_trans
            else:
                decoded.append(token)
        return re.sub(r' +', ' ', ' '.join(decoded)).strip()

    def compress_ratio(self, orig: str, compressed: str) -> float:
        orig_words = len([w for w in jieba.lcut(self._pivot_to_zh(orig)) if len(w) > 1])
        comp_tokens = len(compressed.split())
        return 1 - (comp_tokens / orig_words) if orig_words else 1.0

    def _pivot_to_zh(self, text):
        lang = self._detect_lang(text)
        if lang == 'zh':
            return text
        return self.trans_cache.get(text, (text,))[0]

if __name__ == '__main__':
    tokenizer = HLTokenizer(vocab_size=2000)
    samples = [
        "你好世界 蘋果。",
        "The quick brown fox jumps over the lazy dog.",
        "Pomme bonjour chien.",
        "速い狐犬りんご。"
    ]
    tokenizer.build_vocab(samples)
    print("Vocab sample:", list(tokenizer.zh_bases)[:10])
    print("\n--- Tests ---")
    for text in samples:
        comp = tokenizer.encode(text)
        decoded = tokenizer.decode(comp)
        ratio = tokenizer.compress_ratio(text, comp)
        print(f"Orig: {text}")
        print(f"HL: {comp} (ratio: {ratio:.2f})")
        print(f"Decoded: {decoded}\\n")