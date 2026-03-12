import re
from collections import Counter
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import jieba
import langdetect

class HLTokenizer:
    def __init__(self):
        self.trans_cache: Dict[str, tuple[str, str]] = {}  # orig -> (zh, lang)
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.lang_map = {
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'en': 'eng_Latn',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'de': 'deu_Latn',
            'ko': 'kor_Hang',
        }
        print("Hyper-Language Tokenizer v3.2: Tagged Preprocess Flow [:lang:待trans] + Post-Translate Tokenize")

    def _load_nllb(self):
        if self.nllb_model is None:
            model_name = "facebook/nllb-200-distilled-600M"
            print(f"Loading NLLB {model_name}...")
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.nllb_model.to(self.device)
            self.nllb_model.eval()
            self.nllb_model.config.tie_word_embeddings = False
            print("NLLB loaded OK")

    def _translate(self, texts: List[str], src_lang: str, tgt_lang: str = 'zho_Hans') -> List[str]:
        self._load_nllb()
        src_code = self.lang_map.get(src_lang, 'und_Latn')
        self.nllb_tokenizer.src_lang = src_code
        inputs = self.nllb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            gen_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=256,
                max_length=None,
                num_beams=2,
                early_stopping=True
            )
        return self.nllb_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    def _detect_lang(self, text: str) -> str:
        text = text.strip()
        if any(0x4e00 <= ord(c) <= 0x9fff for c in text):
            return 'zh'
        if len(text) < 3:
            return self._guess_lang(text)
        try:
            ld_lang = langdetect.detect(text)
            return 'zh' if ld_lang.startswith('zh') else ld_lang[:2]
        except:
            return self._guess_lang(text)

    def _guess_lang(self, text: str) -> str:
        text = text.lower().strip()
        if re.match(r'^[a-z0-9\s\.\,\!\?\'\-\']+$', text):
            return 'en'
        if any(0x3040 <= ord(c) <= 0x30ff or 0x31f0 <= ord(c) <= 0x31ff for c in text):
            return 'ja'
        if any(0x4e00 <= ord(c) <= 0x9fff for c in text):
            return 'zh'
        return 'unk'

    def encode(self, text: str) -> str:
        """
        Step 1: Preprocess - tokenize ZH directly, tag non-ZH whole-sentence for translation.
        """
        text = text.strip()
        if not text:
            return ''
        lang = self._detect_lang(text)
        if lang == 'zh':
            words = [w for w in jieba.lcut(text) if len(w) > 1]
            return ' '.join([f'[{w}:zh]' for w in words])
        else:
            return f'[:{lang}:待trans] {text}'

    def finalize_after_translation(self, translated_zh: str, lang: str) -> str:
        """
        Step 2: After translating the tagged sentence to ZH, finalize lang-tagged tokens.
        """
        words = [w for w in jieba.lcut(translated_zh) if len(w) > 1]
        return ' '.join([f'[{w}:{lang}]' for w in words])

    def encode_full(self, text: str) -> str:
        """
        Convenience: auto-translate non-ZH (NLLB, GPU-heavy, for testing only).
        """
        lang = self._detect_lang(text)
        if lang == 'zh':
            return self.encode(text)
        zh_trans = self._translate([text], lang)[0]
        self.trans_cache[text] = (zh_trans, lang)
        return self.finalize_after_translation(zh_trans, lang)

    def decode(self, hl_text: str) -> str:
        words = hl_text.split()
        zh_words = []
        langs = Counter()
        decoded = []
        zh_start = -1
        zh_end = -1
        i = 0
        for token in words:
            m = re.match(r'\[([^\]:]+):([a-z]+)\]', token)
            if m:
                zh_word, src_lang = m.groups()
                if zh_start == -1:
                    zh_start = i
                zh_end = i
                zh_words.append(zh_word)
                langs[src_lang] += 1
            else:
                if zh_start != -1:
                    # process block
                    if len(langs) == 1 and list(langs.keys())[0] != 'zh':
                        main_lang = list(langs.keys())[0]
                        tgt_code = self.lang_map.get(main_lang, 'eng_Latn')
                        zh_sentence = ' '.join(zh_words)
                        rev_trans = self._translate([zh_sentence], 'zh', tgt_code)[0]
                        decoded = decoded[:zh_start] + [rev_trans] + decoded[zh_end+1:]
                    else:
                        decoded[zh_start:zh_end+1] = zh_words
                    zh_words = []
                    langs = Counter()
                    zh_start = -1
                    zh_end = -1
                decoded.append(token)
            i += 1
        # last block
        if zh_words:
            if len(langs) == 1 and list(langs.keys())[0] != 'zh':
                main_lang = list(langs.keys())[0]
                tgt_code = self.lang_map.get(main_lang, 'eng_Latn')
                zh_sentence = ' '.join(zh_words)
                rev_trans = self._translate([zh_sentence], 'zh', tgt_code)[0]
                decoded = decoded[:zh_start] + [rev_trans] + decoded[zh_end+1:]
            else:
                decoded[zh_start:zh_end+1] = zh_words
        return re.sub(r' +', ' ', ' '.join(decoded)).strip()

if __name__ == '__main__':
    tokenizer = HLTokenizer()
    samples = [
        "你好世界 蘋果。",
        "The quick brown fox jumps over the lazy dog.",
        "Pomme bonjour chien.",
        "速い狐犬りんご。"
    ]
    print("--- Step 1: encode (preprocess/tag) ---")
    for text in samples:
        pre = tokenizer.encode(text)
        print(f"Orig: {text}")
        print(f"Pre:  {pre}\\n")
    print("--- Step 2 Example: manual trans + finalize ---")
    ja_trans = "快速 狐 犬 苹果"  # assume batch trans "[:ja:待trans] 速い狐犬りんご。" -> this
    final_ja = tokenizer.finalize_after_translation(ja_trans, 'ja')
    print(f"ZH trans: {ja_trans}")
    print(f"Final: {final_ja}")
    print(f"Decode: {tokenizer.decode(final_ja)}\\n")
    print("--- Full auto encode/decode (test) ---")
    for text in samples[1:]:
        full = tokenizer.encode_full(text)
        dec = tokenizer.decode(full)
        ratio = len(full.split()) / len(jieba.lcut(text))  # rough
        print(f"Orig: {text}")
        print(f"Full HL: {full} (tokens: {len(full.split())}, ratio ~{ratio:.1f}x orig words)")
        print(f"Decoded: {dec}\\n")