#!/usr/bin/env python3
"""
Scientific Test Harness for HL Tokenizer v5.3.1
Generates structured test reports with metrics, comparisons, and validation.
"""

import json
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Import the tokenizer
from hl_tokenizer import HLTokenizer

class TestHarness:
    def __init__(self, output_dir="results/test_results"):
        self.tokenizer = HLTokenizer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "metadata": {
                "version": "5.3.1",
                "timestamp": datetime.now().isoformat(),
                "test_harness_version": "1.0"
            },
            "test_suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "error_rate": 0.0
            },
            "performance": {
                "total_time": 0,
                "avg_latency_ms": 0,
                "min_latency_ms": float('inf'),
                "max_latency_ms": 0
            }
        }
        self.latencies = []

    def log(self, msg: str, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

    def run_test(self, name: str, inputs: List[str], validators: List[callable]) -> Dict:
        """
        Run a single test case with multiple inputs and validators.
        Returns: {passed: bool, latency_ms: float, errors: [str], outputs: [str]}
        """
        test_result = {
            "name": name,
            "passed": True,
            "latency_ms": 0,
            "errors": [],
            "outputs": [],
            "validations": []
        }
        
        start_time = time.time()
        
        for i, input_text in enumerate(inputs):
            try:
                output = self.tokenizer.encode(input_text)
                test_result["outputs"].append({
                    "input": input_text,
                    "output": output
                })
                
                # Run validators
                for validator in validators:
                    try:
                        validation_result = validator(input_text, output)
                        test_result["validations"].append({
                            "validator": validator.__name__,
                            "input": input_text,
                            "passed": validation_result["passed"],
                            "message": validation_result.get("message", "")
                        })
                        if not validation_result["passed"]:
                            test_result["passed"] = False
                            test_result["errors"].append(f"Validator {validator.__name__} failed for '{input_text}': {validation_result.get('message')}")
                    except Exception as e:
                        test_result["passed"] = False
                        test_result["errors"].append(f"Validator {validator.__name__} crashed: {str(e)}")
                        
            except Exception as e:
                test_result["passed"] = False
                test_result["errors"].append(f"Encoding failed for '{input_text}': {str(e)}")
                test_result["outputs"].append({
                    "input": input_text,
                    "output": None,
                    "error": str(e)
                })
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        test_result["latency_ms"] = latency
        self.latencies.append(latency)
        
        return test_result

    # === TEST VALIDATORS ===
    
    @staticmethod
    def validate_format(input_text: str, output: str) -> Dict:
        """Validate output format: [lang][HL...][/lang] or [原][HL...][/原]"""
        if not output:
            return {"passed": False, "message": "Output is empty"}
        
        pattern = r'\[[a-z]{2}|原\](.*?)\[/(?:[a-z]{2}|原)\]'
        matches = re.findall(pattern, output)
        
        if not matches:
            return {"passed": False, "message": f"Output doesn't match format: {output[:100]}"}
        
        return {"passed": True, "message": "Format valid"}

    @staticmethod
    def validate_all_chinese_tokens(input_text: str, output: str) -> Dict:
        """Validate all HL tokens are Chinese characters"""
        hl_tokens = re.findall(r'\[HL([^\]]+)\]', output)
        
        if not hl_tokens:
            return {"passed": False, "message": "No HL tokens found"}
        
        for token in hl_tokens:
            # Check if token contains Chinese characters (including punctuation)
            if not any('\u4e00' <= c <= '\u9fff' for c in token):
                return {"passed": False, "message": f"Non-Chinese token found: {token}"}
        
        return {"passed": True, "message": f"All {len(hl_tokens)} tokens are Chinese"}

    @staticmethod
    def validate_roundtrip(input_text: str, output: str) -> Dict:
        """Validate encode-decode roundtrip (placeholder)"""
        # Would need access to decode() method
        return {"passed": True, "message": "Roundtrip validation (skipped for now)"}

    @staticmethod
    def validate_no_corruption(input_text: str, output: str) -> Dict:
        """Validate that HL tags contain translated content"""
        # Check if there are HL tags with content
        hl_pattern = r'\[HL([^\]]+)\]'
        hl_matches = re.findall(hl_pattern, output)
        
        if not hl_matches:
            return {"passed": False, "message": "No HL content found"}
        
        # Check that HL content is not empty
        total_content = ''.join(hl_matches)
        if len(total_content.strip()) == 0:
            return {"passed": False, "message": "HL content is empty"}
        
        return {"passed": True, "message": "HL content present"}

    # === TEST SUITES ===

    def test_single_language(self):
        """Test single language encoding"""
        test_cases = {
            "English": [
                "Hello",
                "Hello World",
                "The quick brown fox jumps over the lazy dog"
            ],
            "Chinese": [
                "你好",
                "你好世界",
                "我是一個測試用例"
            ],
            "Japanese": [
                "こんにちは",
                "ありがとうございます",
                "日本語のテスト"
            ]
        }
        
        validators = [
            self.validate_format,
            self.validate_all_chinese_tokens,
            self.validate_no_corruption
        ]
        
        for lang, inputs in test_cases.items():
            result = self.run_test(f"Single_{lang}", inputs, validators)
            self.results["test_suites"][f"single_language_{lang.lower()}"] = result
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            self.log(f"{status} Single Language ({lang}): {result['latency_ms']:.2f}ms")

    def test_mixed_language(self):
        """Test mixed language scenarios"""
        test_cases = {
            "EnglishChinese": [
                "Hello你好",
                "Hello World 你好世界",
                "Hi 你好 Thanks 謝謝"
            ],
            "EnglishJapanese": [
                "Helloこんにちは",
                "Hello World とても良い",
                "Thank you ありがとう"
            ],
            "AllThree": [
                "Hello你好こんにちは",
                "Hi 你好 日本",
                "World 世界 セカイ"
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
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            self.log(f"{status} Mixed Language ({scenario}): {result['latency_ms']:.2f}ms")

    def test_edge_cases(self):
        """Test edge cases and special scenarios"""
        test_cases = {
            "EmptyString": [""],
            "Numbers": ["123", "2024年"],
            "Punctuation": ["Hello, world!", "你好！", "こんにちは。"],
            "Symbols": ["@#$%", "🎉😊", "© ® ™"],
            "LongText": ["The quick brown fox " * 10],
            "JapaneseKanji": ["日本語", "机上", "人生"]
        }
        
        validators = [
            self.validate_format,
            self.validate_all_chinese_tokens,
            self.validate_no_corruption
        ]
        
        for scenario, inputs in test_cases.items():
            result = self.run_test(f"EdgeCase_{scenario}", inputs, validators)
            self.results["test_suites"][f"edge_case_{scenario.lower()}"] = result
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            self.log(f"{status} Edge Case ({scenario}): {result['latency_ms']:.2f}ms")

    def run_all_tests(self):
        """Execute all test suites"""
        self.log("=" * 60)
        self.log("Starting HL Tokenizer v5.3.1 Test Harness")
        self.log("=" * 60)
        
        try:
            self.test_single_language()
            self.test_mixed_language()
            self.test_edge_cases()
        except Exception as e:
            self.log(f"Fatal error during testing: {str(e)}", "ERROR")
            traceback.print_exc()
        
        # Calculate summary
        total_tests = sum(1 for suite in self.results["test_suites"].values() for _ in suite.get("outputs", []))
        passed_tests = sum(1 for suite in self.results["test_suites"].values() if suite.get("passed", False))
        failed_tests = sum(1 for suite in self.results["test_suites"].values() if not suite.get("passed", True))
        
        self.results["summary"]["total_tests"] = total_tests
        self.results["summary"]["passed"] = passed_tests
        self.results["summary"]["failed"] = failed_tests
        
        if total_tests > 0:
            self.results["summary"]["error_rate"] = (failed_tests / len(self.results["test_suites"])) * 100
        
        if self.latencies:
            import statistics
            self.results["performance"]["total_time"] = sum(self.latencies)
            self.results["performance"]["avg_latency_ms"] = statistics.mean(self.latencies)
            self.results["performance"]["min_latency_ms"] = min(self.latencies)
            self.results["performance"]["max_latency_ms"] = max(self.latencies)
        
        self.save_results()

    def save_results(self):
        """Save test results to JSON and markdown"""
        # JSON format
        json_path = self.output_dir / "test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.log(f"Results saved to {json_path}")
        
        # Markdown report
        md_path = self.output_dir / "test_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report())
        self.log(f"Report saved to {md_path}")

    def _generate_markdown_report(self) -> str:
        """Generate markdown formatted test report"""
        report = f"""# HL Tokenizer v5.3.1 Test Report

**Generated**: {self.results['metadata']['timestamp']}

## Summary

| Metric | Value |
|--------|-------|
| Total Test Sets | {len(self.results['test_suites'])} |
| Passed | {self.results['summary']['passed']} ✅ |
| Failed | {self.results['summary']['failed']} ❌ |
| Error Rate | {self.results['summary']['error_rate']:.1f}% |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Time | {self.results['performance']['total_time']:.2f}ms |
| Avg Latency | {self.results['performance']['avg_latency_ms']:.2f}ms |
| Min Latency | {self.results['performance']['min_latency_ms']:.2f}ms |
| Max Latency | {self.results['performance']['max_latency_ms']:.2f}ms |

## Test Suite Results

"""
        for suite_name, suite_data in self.results['test_suites'].items():
            status = "✅ PASS" if suite_data['passed'] else "❌ FAIL"
            report += f"### {suite_name} {status}\n\n"
            report += f"**Latency**: {suite_data['latency_ms']:.2f}ms\n\n"
            
            if suite_data['errors']:
                report += "**Errors**:\n"
                for error in suite_data['errors']:
                    report += f"- {error}\n"
                report += "\n"
            
            # Show sample outputs
            if suite_data['outputs']:
                report += "**Sample Outputs**:\n```\n"
                for i, output_item in enumerate(suite_data['outputs'][:3]):
                    report += f"Input: {output_item['input']}\n"
                    report += f"Output: {output_item['output'][:100] if output_item.get('output') else '[ERROR]'}\n\n"
                report += "```\n\n"
        
        return report


if __name__ == "__main__":
    harness = TestHarness()
    harness.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"✅ Passed: {harness.results['summary']['passed']}")
    print(f"❌ Failed: {harness.results['summary']['failed']}")
    print(f"📊 Error Rate: {harness.results['summary']['error_rate']:.1f}%")
    print(f"⏱️  Avg Latency: {harness.results['performance']['avg_latency_ms']:.2f}ms")
    print("=" * 60)
