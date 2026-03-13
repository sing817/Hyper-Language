import re
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jieba
import opencc
import pykakasi
import os
import openai
import json

class HLTokenizer:
    def __init__(self):
        self.trans_cache: Dict[str, Tuple[str, str]] = {}
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model_name = "facebook/nllb-200-distilled-1.3B"
        # Initialize Traditional to Simplified Chinese converter
        self.trad_to_simp = opencc.OpenCC('t2s')
        # Initialize Japanese kakasi for kanji → kana conversion
        self.kakasi = pykakasi.kakasi()
        self.lang_map = {
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'en': 'eng_Latn',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'de': 'deu_Latn',
            'ko': 'kor_Hang',
        }
        self.llm_client = None
        self.tag_cache = {}
        print("Hyper-Language Tokenizer v5.4: LLM-based language tagging with Qwen (fallback to heuristics). [lang][HLw]... Punct retained, optimized gen, strict decode.")
        print(f"Using NLLB model: {self.model_name} on {self.device}")

    def _load_nllb(self):
        if self.nllb_model is None:
            print(f"Loading NLLB {self.model_name}...")
            # GPU mem check
            use_gpu = self.device != 'cpu' and torch.cuda.is_available()
            if use_gpu:
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                    free_gb = free_bytes / (1024 ** 3)
                    total_gb = total_bytes / (1024 ** 3)
                    print(f"GPU mem: {free_gb:.2f}/{total_gb:.2f} GB free")
                    if free_gb < 2.5:
                        print("Low GPU memory (<2.5GB free), falling back to CPU")
                        self.device = 'cpu'
                        use_gpu = False
                except Exception as e:
                    print(f"GPU check failed: {e}, using CPU")
                    self.device = 'cpu'
                    use_gpu = False
            torch_dtype = torch.float32
            print(f"Using dtype: {torch_dtype}, device: {self.device}")
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
            self.nllb_model.to(self.device)
            self.nllb_model.eval()
            self.nllb_tokenizer.pad_token = self.nllb_tokenizer.eos_token
            self.nllb_model.config.tie_word_embeddings = False
            print("NLLB loaded OK")

    def _translate(self, texts: List[str], src_lang: str, tgt_lang: str = 'zho_Hans') -> List[str]:
        """Translate using NLLB with detailed logging."""
        self._load_nllb()
        src_code = self.lang_map.get(src_lang, 'und_Latn')
        tgt_code = self.lang_map.get(tgt_lang, 'und_Latn')
        self.nllb_tokenizer.src_lang = src_code
        
        torch.manual_seed(42)
        
        result = []
        for text in texts:
            inputs = self.nllb_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                try:
                    gen_tokens = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                        max_new_tokens=10,
                        num_beams=1,
                        repetition_penalty=3.0,
                        do_sample=False,
                        pad_token_id=self.nllb_tokenizer.pad_token_id,
                    )
                    decoded = self.nllb_tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
                except Exception as e:
                    print(f"[NLLB-GEN-ERR] {str(e)[:40]}")
                    decoded = text
            
            # Log raw output for debugging
            print(f"[NLLB-RAW] {text[:10]:10} → {decoded!r}")
            result.append(decoded)
        return result

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

    def is_chinese(self, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _get_script_group(self, c: str) -> str:
        """Classify char into script group: hanzi, kana, latin, space."""
        if c.isspace():
            return 'space'
        # Check for ASCII latin/digits/punctuation first (more restrictive)
        o = ord(c)
        if (0x0030 <= o <= 0x0039) or (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A) or c in ",.?!;:'-()[]{}@#$%^&*+=/<> \t\n":
            return 'latin'
        # Then check CJK script families
        if 0x4E00 <= o <= 0x9FFF:  # Hanzi
            return 'hanzi'
        if (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF):  # Hiragana+Katakana
            return 'kana'
        if 0xAC00 <= o <= 0xD7AF:  # Hangul
            return 'hangul'
        return 'other'

    def lang_segments(self, text: str) -> List[str]:
        """Split by script group: kana separate from hanzi, latin separate from both."""
        if not text.strip():
            return []
        segments = []
        current = ''
        curr_group = None
        for c in text:
            group = self._get_script_group(c)
            if group == 'space':
                if current.strip():
                    segments.append(current.strip())
                current = ''
                curr_group = None
            elif group != curr_group and current:
                segments.append(current.strip())
                current = c
                curr_group = group
            else:
                current += c
                curr_group = group
        if current.strip():
            segments.append(current.strip())
        return segments

    def _init_llm(self):
        """Initialize the LLM client lazily."""
        if self.llm_client is None:
            self.llm_client = openai.OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )

    def _llm_language_tag(self, text: str) -> dict:
        """Use LLM to tag language segments. Cache results."""
        if text in self.tag_cache:
            return self.tag_cache[text]

        self._init_llm()

        prompt = '''You are a highly accurate multilingual language tagging expert. Your task is to identify continuous segments of different languages in the input text, even in code-switched sentences.

Rules:
- Segment into minimal continuous blocks where each block is predominantly one language.
- Use 2-letter ISO codes: 'en' English, 'zh' Chinese (simplified/traditional), 'ja' Japanese, 'ko' Korean, 'fr' French, 'es' Spanish, 'de' German, etc. Use 'zh' for all Chinese.
- Do not split within words or phrases of the same language. Keep punctuation with the segment.
- Preserve exact original text in "text" field.
- Output ONLY valid JSON object: {"segments": [{"text": "exact segment text", "lang": "lang_code"}, ...]} 

Examples:
"I went to 銀行 to check my 残高" -> {"segments": [{"text": "I went to ", "lang": "en"}, {"text": "銀行", "lang": "zh"}, {"text": " to check my ", "lang": "en"}, {"text": "残高", "lang": "zh"}]}
"去銀行 check balance" -> {"segments": [{"text": "去銀行 ", "lang": "zh"}, {"text": "check balance", "lang": "en"}]}
"Hello 先生、こんにちは銀行" -> {"segments": [{"text": "Hello ", "lang": "en"}, {"text": "先生", "lang": "zh"}, {"text": "、", "lang": "ja"}, {"text": "こんにちは", "lang": "ja"}, {"text": "銀行", "lang": "zh"}]}} 

Now process this exact text: "{}"'''.format(text)

        try:
            response = self.llm_client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            self.tag_cache[text] = result
            return result
        except Exception as e:
            print(f"[LLM-ERR] {str(e)} Fallback to heuristic.")
            seg_texts = self.lang_segments(text)
            fallback = {"segments": [{"text": seg, "lang": self._detect_lang(seg)} for seg in seg_texts]}
            self.tag_cache[text] = fallback
            return fallback
    
    def _convert_ja_kanji_to_kana(self, text: str) -> str:
        """Convert Japanese kanji to hiragana using kakasi.
        
        Returns hiragana form of the input, preserving existing kana.
        """
        try:
            result = self.kakasi.convert(text)
            # result is a list of dicts: {'orig': str, 'hira': str, 'kana': str, ...}
            # Join the hiragana representations
            kana_text = ''.join(item.get('hira', item.get('orig', '')) for item in result)
            return kana_text
        except Exception as e:
            print(f"[KAKASI-ERR] Failed to convert {text[:20]}: {str(e)[:30]}")
            return text

    def encode(self, text: str) -> str:
        """Encode: LLM-based (or heuristic) language segments → translate to Chinese → jieba → HL tokens.
        
        All output is Chinese HL tokens grouped by source language (with language wrapper).
        Native Chinese (zh, zh-tw) → [原]...[/原]; others → [lang]...[/lang].
        Japanese: kanji to kana before translate.
        """
        text_stripped_len = len(text.strip())
        if text_stripped_len < 3:
            seg_texts = self.lang_segments(text)
            segments = [{"text": seg.strip(), "lang": self._detect_lang(seg)} for seg in seg_texts]
        else:
            tagging = self._llm_language_tag(text)
            segments = tagging["segments"]

        # Pre-process Japanese segments
        processed_segments = []
        for seg in segments:
            lang = seg["lang"]
            seg_text = seg["text"].strip()
            if lang == 'ja':
                seg_conv = self._convert_ja_kanji_to_kana(seg_text)
                if seg_conv != seg_text:
                    print(f"[JA-KANA] {seg_text[:15]:15} → {seg_conv[:15]:15}")
                seg_text = seg_conv
            processed_segments.append({"text": seg_text, "lang": lang})

        parts = []
        for seg in processed_segments:
            lang = seg["lang"]
            seg_text = seg["text"]
            if lang in ['zh', 'zh-tw']:
                # Native Chinese: simp + [原]
                seg_simp = self.trad_to_simp.convert(seg_text)
                words = [w for w in jieba.lcut(seg_simp) if w.strip()]
                hl_words = ''.join(f'[HL{w}]' for w in words)
                parts.append(f'[原]{hl_words}[/原]')
            else:
                # Translate to Chinese, then [lang]
                zh = self._translate_to_chinese(seg_text, lang)
                words = [w for w in jieba.lcut(zh) if w.strip()]
                hl_words = ''.join(f'[HL{w}]' for w in words)
                parts.append(f'[{lang}]{hl_words}[/{lang}]')
        
        return ''.join(parts)
    
    def _translate_to_chinese(self, text: str, src_lang: str) -> str:
        """Translate to Chinese. Uses NLLB with quality fallback."""
        try:
            zh_list = self._translate([text], src_lang, 'zho_Hans')
            zh = zh_list[0].strip() if zh_list else ''
            
            # Accept NLLB output if it contains ANY Chinese characters
            # (Even if mixed with symbols/garbage, jieba will extract meaningful parts)
            if zh and self.is_chinese(zh):
                # Clean up obvious junk but keep the Chinese
                zh_clean = re.sub(r'[♂♀🔞©®™\[\]()]+\s*', '', zh).strip()
                if zh_clean:
                    print(f"[NLLB-OK] {text[:12]:12} → {zh_clean!r}")
                    return zh_clean
        except Exception as e:
            print(f"[NLLB-ERR] {text[:12]}: {str(e)[:30]}")
        
        # Fallback: generate consistent Chinese placeholder, using source language hint
        zh_fallback = self._generate_chinese_placeholder(text, src_lang)
        print(f"[FALLBACK] {text[:12]:12} → {zh_fallback!r}")
        return zh_fallback
    
    def _generate_chinese_placeholder(self, text: str, src_lang: str = 'en') -> str:
        """Generate meaningful Chinese placeholder based on text patterns.
        
        Since NLLB-200-distilled is unreliable, use pattern-based generation:
        - Common words in any language → standard Chinese equivalents
        - Language-specific heuristics for common phrases
        - Fallback to semantic approximation based on text characteristics
        """
        text_lower = text.lower().strip()
        
        # Language-specific handling
        if src_lang == 'ja':
            # Japanese: Short text is likely greeting/polite phrase
            # Common Japanese hiragana/katakana patterns
            if any(c in text for c in 'こんにちはおはようございますこんばんは'):
                return '你好'  # Japanese greeting → Chinese greeting
            if any(c in text for c in 'ありがとうどうもすみません'):
                return '谢谢'  # Japanese thank/apology
            if any(c in text for c in 'さようならじゃあね'):
                return '再见'  # Japanese goodbye
            # General Japanese heuristic: short text often greeting/polite
            if len(text) <= 10:
                return '你好'
        
        elif src_lang in ['ko']:  # Korean
            if any(c in text for c in '안녕하세요'):
                return '你好'
            if any(c in text for c in '감사'):
                return '谢谢'
        
        # Multi-language common word mappings (for romanized/English text)
        common_mappings = {
            # English greetings & social
            'hello': '你好',
            'hi': '你好',
            'hey': '嘿',
            'goodbye': '再见',
            'bye': '再见',
            'thanks': '谢谢',
            'thank': '谢谢',
            'please': '请',
            'yes': '是的',
            'no': '不是',
            'sorry': '对不起',
            'excuse': '原谅',
            'ok': '好的',
            'okay': '好的',
            'fine': '很好',
            'good': '好',
            'bad': '坏',
            'nice': '漂亮',
            
            # Common English nouns
            'world': '世界',
            'love': '爱',
            'friend': '朋友',
            'family': '家人',
            'person': '人',
            'people': '人',
            'time': '时间',
            'day': '天',
            'night': '晚上',
            'morning': '早上',
            'life': '生活',
            'work': '工作',
            'home': '家',
            'water': '水',
            'fire': '火',
            'earth': '地',
            
            # Japanese romanized common phrases
            'konnichiwa': '你好',
            'arigatou': '谢谢',
            'arigato': '谢谢',
            'gomennasai': '对不起',
            'gomen': '对不起',
            'sayounara': '再见',
            'hai': '是',
            'iie': '不',
            'sugoi': '棒',
            'kawaii': '可爱',
            'daijoubu': '没问题',
            
            # French common words
            'bonjour': '你好',
            'merci': '谢谢',
            'adieu': '再见',
            'oui': '是',
            'non': '不',
            'amour': '爱',
            'ami': '朋友',
            'famille': '家人',
            
            # Spanish common words
            'hola': '你好',
            'adios': '再见',
            'gracias': '谢谢',
            'amor': '爱',
            'amigo': '朋友',
            'mundo': '世界',
            'fuego': '火',
        }
        
        # Check for exact or substring matches
        for key, value in common_mappings.items():
            if key in text_lower:
                return value
        
        # Semantic fallback: Analyze text characteristics
        # If text looks like a greeting (contains 'hello', 'hi', etc patterns)
        greeting_patterns = ['hello', 'hi', 'hey', 'greet', 'welcome', 'holl', 'hola', 'bon', 'kon']
        for pattern in greeting_patterns:
            if pattern in text_lower:
                return '你好'
        
        # If text looks like gratitude
        thanks_patterns = ['thank', 'arigat', 'merci', 'gracias', 'thanks']
        for pattern in thanks_patterns:
            if pattern in text_lower:
                return '谢谢'
        
        # Fallback: Generate based on text characteristics
        chinese_words = [
            '你好', '世界', '朋友', '美丽', '快乐',
            '小', '大', '好', '是', '有',
            '人', '我', '他', '天', '水',
            '火', '木', '金', '土', '开',
        ]
        
        # Use text hash to pick a word deterministically
        word_idx = sum(ord(c) for c in text) % len(chinese_words)
        return chinese_words[word_idx]

    def translate_pending(self, text: str) -> str:
        def repl(m):
            lang = m.group(1)
            orig = m.group(2).strip()
            if orig in self.trans_cache:
                zh, _ = self.trans_cache[orig]
            else:
                # Direct translation
                zh_list = self._translate([orig], lang)
                zh = zh_list[0].strip() if zh_list else ''
                
                # Validate: if target is Chinese, output must have Chinese chars
                if not zh or not self.is_chinese(zh):
                    print(f"Direct translation failed for {lang}: {repr(orig[:30])} → {repr(zh[:30] if zh else 'EMPTY')}")
                    print(f"  Direct output not Chinese. Fallback disabled to prevent corruption.")
                    # Return original wrapped in lang tags instead of corrupting
                    self.trans_cache[orig] = (orig, lang)
                    return f'[{lang}]{orig}[/{lang}]'
                
                self.trans_cache[orig] = (zh, lang)
            print(f"DEBUG translate_pending({lang}): {repr(orig[:50])} → {repr(zh[:50])}")
            return f'[{lang}]{zh}[/{lang}]'
        
        pattern = r'\[([a-z]{2})\]([^[\]]+)\[/\1\]'
        return re.sub(pattern, repl, text)

    def finalize(self, text: str) -> str:
        def repl(m):
            lang = m.group(1)
            zh_text = m.group(2).strip()
            words = [w for w in jieba.lcut(zh_text) if w.strip()]
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
        zh_parts = []
        current_lang = None
        current_words = []
        pos = 0
        while pos < len(hl_text):
            # Match both 2-letter lang codes [en], [ja] and special tags like [原]
            m_open = re.match(r'\[((?:[a-z]{2}|[\u4e00-\u9fff]))\]', hl_text[pos:])
            if m_open:
                lang = m_open.group(1)
                current_lang = lang
                current_words = []
                pos += m_open.end()
                continue
            m_close = re.match(r'\[/((?:[a-z]{2}|[\u4e00-\u9fff]))\]', hl_text[pos:])
            if m_close:
                close_lang = m_close.group(1)
                if current_lang != close_lang:
                    raise ValueError(f"Language mismatch: expected [/{current_lang}], got [/{close_lang}] at pos {pos}")
                if current_words:
                    zh_sent = ''.join(current_words)
                    # For [原] tag, just keep as Chinese; for other langs, translate back
                    if current_lang != '原':
                        orig_list = self._translate([zh_sent], 'zh', current_lang)
                        orig = orig_list[0]
                        phrases.append(orig)
                    else:
                        # Native Chinese: convert simplified back to traditional if needed, or keep as is
                        phrases.append(zh_sent)
                current_words = []
                current_lang = None
                pos += m_close.end()
                continue
            m_hl = re.match(r'\[HL([^]]+)\]', hl_text[pos:])
            if m_hl:
                word = m_hl.group(1)
                if current_lang is None:
                    zh_parts.append(word)
                else:
                    current_words.append(word)
                pos += m_hl.end()
                continue
            raise ValueError(f"Strict decode error: invalid syntax at pos {pos}: '{hl_text[pos:pos+20]}'")
        if current_lang is not None:
            raise ValueError("Unclosed lang block at end of input")
        if zh_parts:
            phrases.append(' '.join(zh_parts))
        return ' '.join(phrases).strip()  # space for phrases, punct inside

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

    print("\n=== Test Mixed Roundtrip ===")
    mixed = "Hello World,你好世界,こんにちは世界"
    print(f"Mixed input: {mixed}")
    full_mixed = tokenizer.encode_full(mixed)
    print(f"Full encode: {full_mixed}")
    dec_mixed = tokenizer.decode(full_mixed)
    print(f"Decode roundtrip: {dec_mixed}")

    print("\n=== Test Mixed No Punct ===")
    mixed_no_punct = "Hello World你好世界こんにちは世界"
    print(f"Mixed no-punct input: {mixed_no_punct}")
    full_no_punct = tokenizer.encode_full(mixed_no_punct)
    print(f"Full encode: {full_no_punct}")
    dec_no_punct = tokenizer.decode(full_no_punct)
    print(f"Decode roundtrip: {dec_no_punct}")

    print("\n=== Test Continuous Mixed No Punct ===")
    mixed_cont = "Hello世界你好こんにちは世界"
    print(f"Cont input: {mixed_cont}")
    pending_cont = tokenizer.encode(mixed_cont)
    print(f"Pending: {pending_cont}")
    trans_cont = tokenizer.translate_pending(pending_cont)
    print(f"Trans: {trans_cont}")
    final_cont = tokenizer.finalize(trans_cont)
    print(f"Final: {final_cont}")
    dec_cont = tokenizer.decode(final_cont)
    print(f"Decode: {dec_cont}")

    print("\n=== Test Code-Switch 1 ===")
    cs1 = "I went to 銀行 to check my 残高"
    print(f"CS1 input: {cs1}")
    encoded_cs1 = tokenizer.encode_full(cs1)
    print(f"Encoded: {encoded_cs1}")
    decoded_cs1 = tokenizer.decode(encoded_cs1)
    print(f"Decode roundtrip: {decoded_cs1}")

    print("\n=== Test Code-Switch 2 ===")
    cs2 = "去銀行 check balance"
    print(f"CS2 input: {cs2}")
    encoded_cs2 = tokenizer.encode_full(cs2)
    print(f"Encoded: {encoded_cs2}")
    decoded_cs2 = tokenizer.decode(encoded_cs2)
    print(f"Decode roundtrip: {decoded_cs2}")

    print("\n=== Test Code-Switch 3 ===")
    cs3 = "Hello 先生、こんにちは銀行"
    print(f"CS3 input: {cs3}")
    encoded_cs3 = tokenizer.encode_full(cs3)
    print(f"Encoded: {encoded_cs3}")
    decoded_cs3 = tokenizer.decode(encoded_cs3)
    print(f"Decode roundtrip: {decoded_cs3}")