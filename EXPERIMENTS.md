# Hyper Language Experiments Log\n\n## Timeline\n- **2026-03-11 18:00 HK**: Restart. Fresh clone. Re-add tokenizer + logs.\n\n## Exp-001: MVP Tokenizer\n- **Hypothesis**: Vocab compress 30%+.\n- **Results**: 43% on samples, lossless decode (case fix pending).\n- **Code**: hl_tokenizer.py\n\n## Exp-002: Baseline vs HL Train (Abstract Shared v1)
- **Status**: Running `train_baseline_hl.py` (1000 samples, 4 langs).
- **Vocab**: 1000 (concepts first ~69, freq top).
- **Expected**: 40% token save, +30% PPL hit.
- WandB: hyper-language/HL Abstract Shared v1

Table pending (post-run)...

## Exp-003: Multi-Baseline
Next: `train_multi_base.py` vs Tiktoken/GPT2.

## Exp-005: Multi Baseline Debug (2026-03-12 02:39 HK)\n- **Bug**：Tik cl100k/GPT2 無 trunc → avg tokens 1500+ (C4 長文)。\n- **Fix**：`train_multi_base_fixed.py` (Tik min(encode,128), GPT2 full trunc)。\n- **Run1**：HL PPL 7.5 vs Qwen 25.7 (-71%！勝)，token savings 0% (trunc=128 滿)。\n- **Next**：fixed run + max_length=512 測真壓縮 + 100k tokenizer train。\n\n---\n*OpenClaw Agent (2026-03-12)*