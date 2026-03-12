# HL Tokenizer v5.1 Flow

## Encode Pipeline
1. **phrase_split** `[,.;。！？]` → phrases
2. **encode**:
   - zh (script-prop): jieba → bare `[HLw1][HLw2]`
   - non-zh/mix: `[lang]orig[/lang]` (prop: kana>20%→ja等)
3. **translate_pending** regex `[lang]orig[/lang]` → `[lang]NLLB_zh[/lang]` (cache+DEBUG)
4. **finalize** regex `[lang]zh[/lang]` → `[lang][HLw1 jieba][HLw2]...[/lang]` (filter len>1 strip)

Continuous stream, no `|`.

## Decode Parser
- State machine: [lang] open → collect HLwords → [/lang] zh_join → NLLB zh→lang
- Bare HL → zh_join
- Flush pending

## Setup
```
cd hyper-language
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # torch(transformers), jieba, langdetect
python hl_tokenizer.py  # ~2GB NLLB first-run
```
GPU CUDA for fast NLLB (8GB VRAM OK).

## Notes
- Roundtrip ~lossless (NLLB quality).
- v5.1 fixes: script-prop detect, trans debug, jieba filter.
- Git: f05918e (2026-03-12)