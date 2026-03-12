import re
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jieba
import langdetect

class HLTokenizer:
    def __init__(self):
        self.trans_cache: Dict[str, Tuple[str, str]] = {}
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model_name = "facebook/nllb-200-3.3B"
        self.lang_map = {
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'en': 'eng_Latn',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'de': 'deu_Latn',
            'ko': 'kor_Hang',
        }
        print("Hyper-Language Tokenizer v5.0: lang-block encode - continuous stream, [lang][HLw]...[/lang] or bare [HLw] for zh")
        print(f"Using NLLB model: {self.model_name} on {self.device}")

    def _load_nllb(self):
        if self.nllb_model is None:
            print(f"Loading NLLB {self.model_name}...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"Using dtype: {torch_dtype}, device: {self.device}")
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
            self.nllb_model.to(self.device)
            self.nllb_model.eval()
            self.nllb_tokenizer.pad_token = self.nllb_tokenizer.eos_token
            self.nllb_model.config.tie_word_embeddings = False
            print("NLLB loaded OK")

    def _translate(self, texts: List[str], src_lang: str, tgt_lang: str = 'zho_Hans') -> List[str]:
        self._load_nllb()
        src_code = self.lang_map.get(src_lang, 'und_Latn')
        tgt_code = self.lang_map.get(tgt_lang, 'und_Latn')
        self.nllb_tokenizer.src_lang = src_code
        inputs = self.nllb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            gen_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=512,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.nllb_tokenizer.pad_token_id,
                repetition_penalty=1.3,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
        return self.nllb_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    def _detect_lang(self, text: str) -> str:
        text = text.strip()
        if len(text) < 3:
            return self._guess_lang(text)
        h_count, k_count, l_count, h_p, k_p, l_p = self._count_scripts(text)
        if k_p > 0.2 and k_count > h_count:
            return 'ja'
        if h_p > 0.2:
            return 'zh'
        if l_p > 0.5:
            return 'en'
        try:
            ld_lang = langdetect.detect(text)
            return 'zh' if ld_lang.startswith('zh') else ld_lang[:2]
        except:
            return self._guess_lang(text)

    def _guess_lang(self, text: str) -> str:
        text = text.lower().strip()
        if re.match(r'^[a-z0-9 ,.!?\'-]+$', text):
            return 'en'
        if any(0x3040 <= ord(c) <= 0x30ff or 0x31f0 <= ord(c) <= 0x31ff for c in text):
            return 'ja'
        if any(0x4e00 <= ord(c) <= 0x9fff for c in text):
            return 'zh'
        return 'unk'

    def _count_scripts(self, text: str) -> Tuple[int, int, int, float, float, float]:
        hanzi_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        kana_count = sum(1 for c in text if (0x3040 <= ord(c) <= 0x30FF or 0x31F0 <= ord(c) <= 0x31FF))
        latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total = len(text)
        hanzi_p = hanzi_count / total if total > 0 else 0.0
        kana_p = kana_count / total if total > 0 else 0.0
        latin_p = latin_count / total if total > 0 else 0.0
        return hanzi_count, kana_count, latin_count, hanzi_p, kana_p, latin_p

    def phrase_split(self, text: str) -> List[str]:
        # split on comma, period, etc.
        phrases = re.split(r'[,.;。！？]', text.strip())
        return [p.strip() for p in phrases if p.strip()]

    def encode(self, text: str) -> str:
        phrases = self.phrase_split(text)
        parts = []
        for phrase in phrases:
            lang = self._detect_lang(phrase)
            if lang == 'zh':
                words = [w for w in jieba.lcut(phrase) if len(w) > 1 and w.strip()]
                hl_part = ''.join(f'[HL{w}]' for w in words)
                parts.append(hl_part)
            else:
                parts.append(f'[{lang}]{phrase}[/{lang}]')
        return ''.join(parts)

    def translate_pending(self, text: str) -> str:
        def repl(m):
            lang = m.group(1)
            orig = m.group(2).strip()
            if orig in self.trans_cache:
                zh, _ = self.trans_cache[orig]
            else:
                zh_list = self._translate([orig], lang)
                zh = zh_list[0]
                if self._detect_lang(zh) != 'zh':
                    print(f"Fallback via en for {lang}: direct {repr(zh)}")
                    en_list = self._translate([orig], lang, 'eng_Latn')
                    en_trans = en_list[0]
                    zh = en_trans  # Use accurate en proxy for jieba/HL (avoids repetition)
                    print(f"  → en: {repr(en_trans)} (proxy for HL)")
                self.trans_cache[orig] = (zh, lang)
            print(f"DEBUG translate_pending({lang}): {repr(orig)} → {repr(zh)}")
            return f'[{lang}]{zh}[/{lang}]'
        pattern = r'\[([a-z]{2})\]([^[\]]+)\[/\1\]'
        return re.sub(pattern, repl, text)

    def finalize(self, text: str) -> str:
        def repl(m):
            lang = m.group(1)
            zh_text = m.group(2).strip()
            words = [w for w in jieba.lcut(zh_text) if len(w) > 1 and w.strip()]
            hl_words = ''.join(f'[HL{w}]' for w in words)
            return f'[{lang}]{hl_words}[/{lang}]'
        pattern = r'\[([a-z]{2})\]([^[\]]+)\[/\1\]'
        return re.sub(pattern, repl, text)

    def encode_full(self, text: str) -> str:
        pending = self.encode(text)
        transed = self.translate_pending(pending)
        hl_final = self.finalize(transed)
        return hl_final

    def decode(self, hl_text: str) -> str:
        self._load_nllb()
        phrases = []
        current_lang = None
        current_words = []
        pos = 0
        while pos < len(hl_text):
            m_open = re.match(r'\[([a-z]{2})\]', hl_text[pos:])
            if m_open:
                lang = m_open.group(1)
                if current_lang is not None and current_words:
                    # close prev block? but streaming, assume sequential
                    pass  # handled at close
                current_lang = lang
                current_words = []
                pos += m_open.end()
                continue
            m_close = re.match(r'\[/([a-z]{2})\]', hl_text[pos:])
            if m_close:
                close_lang = m_close.group(1)
                if current_lang == close_lang and current_words:
                    zh_sent = ''.join(current_words)
                    tgt_code = self.lang_map.get(current_lang, 'eng_Latn')
                    orig_list = self._translate([zh_sent], 'zh', current_lang)
                    orig = orig_list[0]
                    phrases.append(orig)
                current_words = []
                current_lang = None
                pos += m_close.end()
                continue
            m_hl = re.match(r'\[HL([^]]+)\]', hl_text[pos:])
            if m_hl:
                word = m_hl.group(1)
                current_words.append(word)
                pos += m_hl.end()
                continue
            # non-token char? skip or error
            pos += 1
        # final block
        if current_words:
            if current_lang:
                zh_sent = ''.join(current_words)
                tgt_code = self.lang_map.get(current_lang, 'eng_Latn')
                orig_list = self._translate([zh_sent], 'zh', current_lang)
                orig = orig_list[0]
                phrases.append(orig)
            else:
                phrases.append(''.join(current_words))
        return ', '.join(phrases).strip()  # comma for phrases

if __name__ == '__main__':
    tokenizer = HLTokenizer()
    print("\n=== Test JA ===")
    ja = "こんにちは世界"
    print(f"JA input: {ja}")
    pending_ja = tokenizer.encode(ja)
    print(f"Pending: {pending_ja}")
    trans_ja = tokenizer.translate_pending(pending_ja)
    print(f"Trans: {trans_ja}")
    final_ja = tokenizer.finalize(trans_ja)
    print(f"Final: {final_ja}")
    dec_ja = tokenizer.decode(final_ja)
    print(f"Decode: {dec_ja}")

    print("\n=== Test EN ===")
    en = "Hello World, good morning."
    print(f"EN input: {en}")
    pending_en = tokenizer.encode(en)
    print(f"Pending: {pending_en}")
    trans_en = tokenizer.translate_pending(pending_en)
    print(f"Trans: {trans_en}")
    final_en = tokenizer.finalize(trans_en)
    print(f"Final: {final_en}")
    dec_en = tokenizer.decode(final_en)
    print(f"Decode: {dec_en}")
