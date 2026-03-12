import re
from collections import Counter
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import jieba
import langdetect

class HLTokenizer:
    def __init__(self):
        self.trans_cache: Dict[str, Tuple[str, str]] = {}  # orig -> (zh, lang)
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
        print("Hyper-Language Tokenizer v4.0: Sentence-wise ['sent':'lang':'待trans'] -> ['zh':'lang':'translated'] -> [HLword:lang]")

    def _load_nllb(self):
        if self.nllb_model is None:
            model_name = \"facebook/nllb-200-distilled-600M\"
            print(f\"Loading NLLB {model_name}...\")
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.nllb_model.to(self.device)
            self.nllb_model.eval()
            self.nllb_model.config.tie_word_embeddings = False
            print(\"NLLB loaded OK\")

    def _translate(self, texts: List[str], src_lang: str, tgt_lang: str = 'zho_Hans') -> List[str]:
        self._load_nllb()
        src_code = self.lang_map.get(src_lang, 'und_Latn')
        self.nllb_tokenizer.src_lang = src_code
        inputs = self.nllb_tokenizer(texts, return_tensors=\"pt\", padding=True, truncation=True).to(self.device)
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
        if re.match(r'^[a-z0-9\\s\\.\\,\\!\\?\\'\\-\\']+$', text):
            return 'en'
        if any(0x3040 <= ord(c) <= 0x30ff or 0x31f0 <= ord(c) <= 0x31ff for c in text):
            return 'ja'
        if any(0x4e00 <= ord(c) <= 0x9fff for c in text):
            return 'zh'
        return 'unk'

    def sentence_split(self, text: str) -> List[str]:
        # Simple sentence splitter: . ! ? followed by space or newline
        sents = re.split(r'(?<=[\.!\?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip()]

    def parse_tag(self, tagged: str) -> Tuple[str, str]:
        \"\"\"Parse [\"sent\":\"lang\":\"待trans\"] -> (sent, lang)\"\"\"
        m = re.match(r'\\[\\\"(.*?)\\\":\\\"([a-z]+)\\\":\\\"待trans\\\"\\]', tagged)
        if m:
            return m.groups()
        return tagged, 'unk'

    def encode_sentence(self, sent: str) -> str:
        lang = self._detect_lang(sent)
        if lang == 'zh':
            words = [w for w in jieba.lcut(sent) if len(w) > 1]
            return ' '.join([f'[HL{w}:zh]' for w in words])
        else:
            return f'[\"{sent}\":\"{lang}\":\"待trans\"]'

    def encode(self, text: str) -> str:
        sents = self.sentence_split(text)
        return ' | '.join(self.encode_sentence(sent) for sent in sents)  # | as sent separator

    def finalize_sentence(self, zh_trans: str, lang: str) -> str:
        words = [w for w in jieba.lcut(zh_trans) if len(w) > 1]
        return ' '.join([f'[HL{w}:{lang}]' for w in words])

    def encode_full(self, text: str) -> str:
        lang = self._detect_lang(text)
        tagged = self.encode_sentence(text)
        if lang == 'zh':
            return tagged
        orig, lng = self.parse_tag(tagged)
        zh_trans = self._translate([orig], lng)[0]
        self.trans_cache[orig] = (zh_trans, lng)
        return self.finalize_sentence(zh_trans, lng)

    def decode(self, hl_text: str) -> str:
        parts = hl_text.split(' | ')
        decoded_parts = []
        for part in parts:
            words = part.split()
            block_words = []
            block_lang = None
            for token in words:
                m = re.match(r'\\\[HL([^\\]:]+):([a-z]+)\\\\]', token)
                if m:
                    zh_word, lng = m.groups()
                    block_words.append(zh_word)
                    if block_lang is None:
                        block_lang = lng
                    elif block_lang != lng:
                        block_lang = 'mixed'
                else:
                    block_words.append(token)
            if block_words and block_lang and block_lang != 'zh' and block_lang != 'mixed':
                zh_sent = ' '.join(block_words)
                tgt_code = self.lang_map.get(block_lang, 'eng_Latn')
                rev = self._translate([zh_sent], 'zh', tgt_code)[0]
                decoded_parts.append(rev)
            else:
                decoded_parts.append(' '.join(block_words))
        return '. '.join(decoded_parts).strip()

if __name__ == '__main__':
    tokenizer = HLTokenizer()
    samples = [
        \"你好世界 蘋果。\",
        \"The quick brown fox jumps over the lazy dog.\",
        \"Pomme bonjour chien.\",
        \"速い狐犬りんご。\"
    ]
    print(\"--- encode (sentence-wise tag / HL prefix) ---\")
    for text in samples:
        enc = tokenizer.encode(text)
        print(f\"Orig: {text}\")
        print(f\"Enc:  {enc}\\n\")
    print(\"--- finalize example ---\")
    zh_ja = \"敏捷的 棕色 狐狸 跳過 懶狗\"
    final = tokenizer.finalize_sentence(zh_ja, 'ja')
    print(f\"ZH trans: {zh_ja}\")
    print(f\"Final HL: {final}\")
    print(f\"Decode: {tokenizer.decode(final)}\\n\")
    print(\"--- full pipeline test ---\")
    for text in samples[1:]:
        full = tokenizer.encode_full(text)
        dec = tokenizer.decode(full)
        print(f\"Orig: {text}\")
        print(f\"Full HL: {full}\")
        print(f\"Decoded: {dec}\\n\")