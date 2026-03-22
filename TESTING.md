# Scientific Testing Methodology for HL Tokenizer v5.3.1

## Overview

This document outlines the scientific testing framework for validating the Hyper-Language Tokenizer.

## Quick Start

```bash
# Step 1: Clean up old data
python3 cleanup_old_data.py

# Step 2: Run test harness
python3 test_harness.py

# Step 3: Generate scientific report
python3 generate_report.py

# Step 4: View results
cat results/test_results/SCIENTIFIC_REPORT.md
cat results/test_results/test_results.json
```

## Test Components

### 1. Test Harness (`test_harness.py`)
**Purpose**: Core testing infrastructure with validators

**Features**:
- Structured test suites (single language, mixed language, edge cases)
- Automatic validator execution
- Latency measurement
- JSON output for analysis
- Markdown report generation

**Test Suites**:
- **Single Language** (3): English, Chinese, Japanese
- **Mixed Language** (3): EN↔ZH, EN↔JA, All Three
- **Edge Cases** (6): Empty, Numbers, Punctuation, Symbols, Long text, Kanji

**Validators**:
1. Format validation (regex `[lang][HL...][/lang]`)
2. Chinese token validation (U+4E00-U+9FFF)
3. No data corruption check
4. Roundtrip capability (placeholder)

**Output**:
```
results/test_results/
├── test_results.json      # Machine-readable metrics
└── test_report.md         # Basic markdown report
```

### 2. Data Cleanup (`cleanup_old_data.py`)
**Purpose**: Remove legacy test files and organize results

**Actions**:
- Archives old test files (test_*.py, debug_*.py) to `results/archived_tests/`
- Creates clean directory structure
- Generates `results/INDEX.md` for navigation

**Safe**: Files are backed up, not deleted

### 3. Report Generation (`generate_report.py`)
**Purpose**: Create comprehensive scientific report

**Features**:
- Parses test_results.json
- Generates detailed markdown report
- Includes performance analysis
- Feature validation checklist
- Production readiness assessment

**Output**:
```
results/test_results/SCIENTIFIC_REPORT.md
  ├── Executive Summary
  ├── Test Methodology (Scope, Categories, Design)
  ├── Detailed Results (by category + examples)
  ├── Performance Analysis (Latency breakdown)
  ├── Quality Metrics (Success rates)
  ├── Feature Validation Checklist
  ├── Recommendations (Production, Enhancement)
  └── Appendix (Raw data)
```

## Test Design Principles

### Dimension 1: Format Validation
```python
# Required format:
[lang][HL中文1][HL中文2][/lang]
[原][HL中文][/原]

# Validator regex:
r'\[[a-z]{2}|原\](.*?)\[/(?:[a-z]{2}|原)\]'
```

### Dimension 2: Multilingual Coverage
- **Single Language**: Baseline for each language (EN, ZH, JA)
- **Mixed Language**: Real-world scenarios (combinations of 2-3 languages)
- **Edge Cases**: Boundary conditions and special scenarios

### Dimension 3: Data Integrity
- No character loss
- Proper script family separation
- Safe handling of mixed scripts

## Metrics Captured

### Performance Metrics
```json
{
    "total_time": 12345.67,           // Total test execution (ms)
    "avg_latency_ms": 456.78,         // Average per test (ms)
    "min_latency_ms": 123.45,         // Best case (ms)
    "max_latency_ms": 789.01          // Worst case (ms)
}
```

### Quality Metrics
```json
{
    "total_tests": 18,               // Total test cases
    "passed": 17,                    // Passed suites
    "failed": 1,                     // Failed suites
    "error_rate": 5.56               // Failure percentage
}
```

### Per-Test Data
```json
{
    "name": "Single_English",
    "passed": true,
    "latency_ms": 456.78,
    "errors": [],
    "outputs": [
        {
            "input": "Hello",
            "output": "[en][HL你好][/en]"
        }
    ],
    "validations": [
        {
            "validator": "validate_format",
            "passed": true,
            "message": "Format valid"
        }
    ]
}
```

## Test Execution Results Interpretation

### Success Rate Classification
| Rate | Status | Action |
|------|--------|--------|
| ≥95% | 🟢 Production Ready | Deploy |
| 80-95% | 🟡 Acceptable | Minor fixes |
| <80% | 🔴 Needs Work | Debug and fix |

### Common Success Metrics
- **Format validation**: Should be 100% (validates basic structure)
- **Chinese token validation**: Should be 100% (all HL tokens must be Chinese)
- **Data integrity**: Should be 100% (no corruption)
- **Overall success**: Typically 95%+ for stable v5.3.1

## Typical Test Output Example

```
[2026-03-14 10:30:45] [INFO] Starting HL Tokenizer v5.3.1 Test Harness
============================================================
[2026-03-14 10:30:46] [INFO] ✅ PASS Single Language (English): 123.45ms
[2026-03-14 10:30:47] [INFO] ✅ PASS Single Language (Chinese): 456.78ms
[2026-03-14 10:30:48] [INFO] ✅ PASS Single Language (Japanese): 234.56ms
[2026-03-14 10:30:49] [INFO] ✅ PASS Mixed Language (EnglishChinese): 345.67ms
[2026-03-14 10:30:50] [INFO] ✅ PASS Mixed Language (EnglishJapanese): 456.78ms
[2026-03-14 10:30:51] [INFO] ✅ PASS Mixed Language (AllThree): 567.89ms
[2026-03-14 10:30:52] [INFO] ✅ PASS Edge Case (EmptyString): 12.34ms
...
============================================================
✅ Passed: 11
❌ Failed: 1
📊 Error Rate: 8.3%
⏱️  Avg Latency: 345.67ms
============================================================

Results saved to results/test_results/test_results.json
Report saved to results/test_results/test_report.md
```

## Adding New Test Cases

To extend the test suite, modify `test_harness.py`:

### Example: Add new language pair test
```python
def test_korean_mixing(self):
    """Test Korean + English mixing"""
    test_cases = {
        "KoreanEnglish": [
            "Hello안녕",
            "How are you 어떻게",
        ]
    }
    
    validators = [
        self.validate_format,
        self.validate_all_chinese_tokens,
        self.validate_no_corruption
    ]
    
    for scenario, inputs in test_cases.items():
        result = self.run_test(f"Mixed_{scenario}", inputs, validators)
        self.results["test_suites"][f"mixed_{scenario.lower()}"] = result

# Add to run_all_tests():
self.test_korean_mixing()
```

## Performance Optimization Tips

### Latency Reduction
1. **NLLB Model Inference** dominates (500-1000ms per segment)
   - Consider batch processing for throughput
   - Or use smaller models (though less accurate)

2. **Jieba Segmentation** is fast (~10-50ms)

3. **LangDetect** adds ~50-100ms

### For High-Throughput Applications
```python
# Use batch encoding (not yet implemented)
inputs = ["Hello", "你好", "こんにちは"]
outputs = tokenizer.encode_batch(inputs)  # Parallelized NLLB
```

## Report Analysis

### Viewing Results

**Machine-readable JSON**:
```bash
cat results/test_results/test_results.json | python3 -m json.tool
```

**Human-readable Report**:
```bash
# View main report
cat results/test_results/SCIENTIFIC_REPORT.md

# View basic test report
cat results/test_results/test_report.md
```

### Key Sections to Check
1. **Executive Summary**: Quick status overview
2. **Test Design**: Confirm test scope matches requirements
3. **Detailed Results**: Check each test suite passed/failed
4. **Performance**: Verify latency is acceptable
5. **Feature Checklist**: Confirm all v5.3.1 features working
6. **Recommendations**: Identify next steps

## Continuous Testing

For ongoing validation after code changes:

```bash
#!/bin/bash
# run_tests.sh - Automated test runner

echo "🧹 Cleaning old data..."
python3 cleanup_old_data.py

echo "🧪 Running tests..."
python3 test_harness.py

echo "📊 Generating report..."
python3 generate_report.py

echo "✅ Done! View: results/test_results/SCIENTIFIC_REPORT.md"
```

## Troubleshooting

### Tests timeout or hang
- Check if NLLB model is downloading (~2.5GB)
- Monitor GPU memory if using CUDA
- Check for network issues (transformers library downloads)

### Some validators fail
- Check `results/test_results/test_results.json` for detailed error messages
- Look at "outputs" array for actual vs expected output
- Run `python3 -c "from hl_tokenizer import HLTokenizer; t = HLTokenizer(); print(t.encode('test'))"` to debug directly

### Latency too high
- Normal for first run (model loading + compilation)
- Second run should be faster
- NLLB inference is inherently slow (~500ms); consider model optimization

---

**Last Updated**: 2026-03-14  
**Framework Version**: 1.0  
**HL Tokenizer Version**: v5.3.1
