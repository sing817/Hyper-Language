# HL Tokenizer v5.3.1 Test Report

**Generated**: 2026-03-14T07:13:25.611408

## Summary

| Metric | Value |
|--------|-------|
| Total Test Sets | 12 |
| Passed | 0 ✅ |
| Failed | 12 ❌ |
| Error Rate | 100.0% |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Time | 177537.73ms |
| Avg Latency | 14794.81ms |
| Min Latency | 0.04ms |
| Max Latency | 30068.18ms |

## Test Suite Results

### single_language_english ❌ FAIL

**Latency**: 18580.19ms

**Errors**:
- Validator validate_no_corruption failed for 'Hello': All content was stripped
- Validator validate_no_corruption failed for 'Hello World': All content was stripped
- Validator validate_no_corruption failed for 'The quick brown fox jumps over the lazy dog': All content was stripped

**Sample Outputs**:
```
Input: Hello
Output: [en][HL你好][/en]

Input: Hello World
Output: [en][HL你好][/en]

Input: The quick brown fox jumps over the lazy dog
Output: [en][HL金][/en]

```

### single_language_chinese ❌ FAIL

**Latency**: 4289.32ms

**Errors**:
- Validator validate_no_corruption failed for '你好': All content was stripped
- Validator validate_no_corruption failed for '你好世界': All content was stripped
- Validator validate_no_corruption failed for '我是一個測試用例': All content was stripped

**Sample Outputs**:
```
Input: 你好
Output: [原][HL你好][/原]

Input: 你好世界
Output: [原][HL你好][HL世界][/原]

Input: 我是一個測試用例
Output: [原][HL我][HL是][HL一个][HL测试用例][/原]

```

### single_language_japanese ❌ FAIL

**Latency**: 11716.32ms

**Errors**:
- Validator validate_no_corruption failed for 'こんにちは': All content was stripped
- Validator validate_no_corruption failed for 'ありがとうございます': All content was stripped
- Validator validate_no_corruption failed for '日本語のテスト': All content was stripped

**Sample Outputs**:
```
Input: こんにちは
Output: [ja][HL你好][/ja]

Input: ありがとうございます
Output: [ja][HL你好][/ja]

Input: 日本語のテスト
Output: [ja][HL你好][/ja]

```

### mixed_englishchinese ❌ FAIL

**Latency**: 19062.56ms

**Errors**:
- Validator validate_no_corruption failed for 'Hello你好': All content was stripped
- Validator validate_no_corruption failed for 'Hello World 你好世界': All content was stripped
- Validator validate_no_corruption failed for 'Hi 你好 Thanks 謝謝': All content was stripped

**Sample Outputs**:
```
Input: Hello你好
Output: [en][HL你好][/en][原][HL你好][/原]

Input: Hello World 你好世界
Output: [en][HL你好][/en][原][HL你好][HL世界][/原]

Input: Hi 你好 Thanks 謝謝
Output: [en][HL你好][/en][原][HL你好][/原][en][HL谢谢][/en][原][HL谢谢][/原]

```

### mixed_englishjapanese ❌ FAIL

**Latency**: 18798.01ms

**Errors**:
- Validator validate_no_corruption failed for 'Helloこんにちは': All content was stripped
- Validator validate_no_corruption failed for 'Hello World とても良い': All content was stripped
- Validator validate_no_corruption failed for 'Thank you ありがとう': All content was stripped

**Sample Outputs**:
```
Input: Helloこんにちは
Output: [en][HL你好][/en][ja][HL你好][/ja]

Input: Hello World とても良い
Output: [en][HL你好][/en][ja][HL你好][/ja]

Input: Thank you ありがとう
Output: [en][HL谢谢][/en][ja][HL你好][/ja]

```

### mixed_allthree ❌ FAIL

**Latency**: 28115.33ms

**Errors**:
- Validator validate_no_corruption failed for 'Hello你好こんにちは': All content was stripped
- Validator validate_no_corruption failed for 'Hi 你好 日本': All content was stripped
- Validator validate_no_corruption failed for 'World 世界 セカイ': All content was stripped

**Sample Outputs**:
```
Input: Hello你好こんにちは
Output: [en][HL你好][/en][原][HL你好][/原][ja][HL你好][/ja]

Input: Hi 你好 日本
Output: [en][HL你好][/en][原][HL你好][HL日本][/原]

Input: World 世界 セカイ
Output: [en][HL世界][/en][ja][HL你好][/ja][ja][HL你好][/ja]

```

### edge_case_emptystring ❌ FAIL

**Latency**: 0.04ms

**Errors**:
- Validator validate_format failed for '': Output is empty
- Validator validate_all_chinese_tokens failed for '': No HL tokens found
- Validator validate_no_corruption failed for '': All content was stripped

**Sample Outputs**:
```
Input: 
Output: [ERROR]

```

### edge_case_numbers ❌ FAIL

**Latency**: 30068.18ms

**Errors**:
- Validator validate_no_corruption failed for '123': All content was stripped
- Validator validate_no_corruption failed for '2024年': All content was stripped

**Sample Outputs**:
```
Input: 123
Output: [en][HL人][/en]

Input: 2024年
Output: [ja][HL你好][/ja]

```

### edge_case_punctuation ❌ FAIL

**Latency**: 10486.63ms

**Errors**:
- Validator validate_no_corruption failed for 'Hello, world!': All content was stripped
- Validator validate_all_chinese_tokens failed for '你好！': Non-Chinese token found: ！
- Validator validate_no_corruption failed for '你好！': All content was stripped
- Validator validate_no_corruption failed for 'こんにちは。': All content was stripped

**Sample Outputs**:
```
Input: Hello, world!
Output: [en][HL你好][/en]

Input: 你好！
Output: [原][HL你好][HL！][/原]

Input: こんにちは。
Output: [ja][HL你好][/ja]

```

### edge_case_symbols ❌ FAIL

**Latency**: 26948.25ms

**Errors**:
- Validator validate_no_corruption failed for '@#$%': All content was stripped
- Validator validate_no_corruption failed for '© ® ™': All content was stripped

**Sample Outputs**:
```
Input: @#$%
Output: [en][HL他][/en]

Input: 🎉😊
Output: [unk][HL美丽][/unk]

Input: © ® ™
Output: [en][HL有][/en]

```

### edge_case_longtext ❌ FAIL

**Latency**: 4961.07ms

**Errors**:
- Validator validate_no_corruption failed for 'The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox ': All content was stripped

**Sample Outputs**:
```
Input: The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox The quick brown fox 
Output: [en][HL土][/en]

```

### edge_case_japanesekanji ❌ FAIL

**Latency**: 4511.83ms

**Errors**:
- Validator validate_no_corruption failed for '日本語': All content was stripped
- Validator validate_no_corruption failed for '机上': All content was stripped
- Validator validate_no_corruption failed for '人生': All content was stripped

**Sample Outputs**:
```
Input: 日本語
Output: [ja][HL你好][/ja]

Input: 机上
Output: [原][HL机上][/原]

Input: 人生
Output: [原][HL人生][/原]

```

