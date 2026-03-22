#!/bin/bash
# run_full_test.sh - Complete scientific testing workflow

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   HL Tokenizer v5.3.1 - Scientific Testing Workflow       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Cleanup
echo "📋 Step 1: Cleaning up old test data..."
echo "─────────────────────────────────────────────────────────────"
python3 cleanup_old_data.py
echo ""

# Step 2: Run tests
echo "🧪 Step 2: Running comprehensive test suite..."
echo "─────────────────────────────────────────────────────────────"
echo "(This may take 2-5 minutes due to NLLB model inference)"
echo ""
python3 test_harness.py
echo ""

# Step 3: Generate report
echo "📊 Step 3: Generating scientific report..."
echo "─────────────────────────────────────────────────────────────"
python3 generate_report.py
echo ""

# Step 4: Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   ✅ Testing Complete!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results Location:"
echo "   └─ results/test_results/"
echo "      ├─ test_results.json           (Machine-readable metrics)"
echo "      ├─ test_report.md              (Basic markdown report)"
echo "      └─ SCIENTIFIC_REPORT.md        (Full analysis report)"
echo ""
echo "📖 View Results:"
echo "   cat results/test_results/SCIENTIFIC_REPORT.md"
echo ""
echo "🔍 Inspect Raw Data:"
echo "   cat results/test_results/test_results.json | python3 -m json.tool"
echo ""
echo "📋 Test Documentation:"
echo "   cat TESTING.md"
echo ""
