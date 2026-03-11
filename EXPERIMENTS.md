# Hyper Language Experiments Log\n\n## Timeline\n- **2026-03-11 18:00 HK**: Restart. Fresh clone. Re-add tokenizer + logs.\n\n## Exp-001: MVP Tokenizer\n- **Hypothesis**: Vocab compress 30%+.\n- **Results**: 43% on samples, lossless decode (case fix pending).\n- **Code**: hl_tokenizer.py\n\n## Exp-002: Baseline vs HL Train (Abstract Shared v1)
- **Status**: Running `train_baseline_hl.py` (1000 samples, 4 langs).
- **Vocab**: 1000 (concepts first ~69, freq top).
- **Expected**: 40% token save, +30% PPL hit.
- WandB: hyper-language/HL Abstract Shared v1

Table pending (post-run)...

## Exp-003: Multi-Baseline
Next: `train_multi_base.py` vs Tiktoken/GPT2.

---\n*OpenClaw Agent (2026-03-12)*