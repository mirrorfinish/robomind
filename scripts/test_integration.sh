#!/bin/bash
# RoboMind Integration Test Script
# Tests full analysis workflow on a project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="/tmp/robomind_integration_test"

echo "============================================"
echo "RoboMind Integration Test"
echo "============================================"
echo ""

# Clean up previous test
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

echo "1. Running pytest..."
echo "--------------------------------------------"
python -m pytest tests/ -v --tb=short
echo ""

echo "2. Testing CLI commands..."
echo "--------------------------------------------"

echo "  -> robomind info"
python -m robomind info
echo ""

echo "  -> robomind scan (on test fixtures)"
python -m robomind scan tests/fixtures -v
echo ""

echo "  -> robomind launch (on test fixtures)"
python -m robomind launch tests/fixtures -v
echo ""

echo "  -> robomind graph (on test fixtures)"
python -m robomind graph tests/fixtures --coupling -v
echo ""

echo "  -> robomind analyze (full analysis)"
python -m robomind analyze tests/fixtures -o "$OUTPUT_DIR/analysis" -f json -f yaml -f html -v
echo ""

echo "  -> robomind visualize (standalone)"
python -m robomind visualize tests/fixtures -o "$OUTPUT_DIR/standalone_viz.html"
echo ""

echo "3. Verifying output files..."
echo "--------------------------------------------"

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null)
        echo "  [OK] $1 ($SIZE bytes)"
    else
        echo "  [FAIL] $1 not found!"
        exit 1
    fi
}

check_file "$OUTPUT_DIR/analysis/system_graph.json"
check_file "$OUTPUT_DIR/analysis/system_context.yaml"
check_file "$OUTPUT_DIR/analysis/CONTEXT_SUMMARY.yaml"
check_file "$OUTPUT_DIR/analysis/visualization.html"
check_file "$OUTPUT_DIR/standalone_viz.html"

echo ""
echo "4. Validating JSON output..."
echo "--------------------------------------------"
python -c "import json; json.load(open('$OUTPUT_DIR/analysis/system_graph.json')); print('  [OK] JSON is valid')"

echo ""
echo "5. Validating YAML output..."
echo "--------------------------------------------"
python -c "import yaml; yaml.safe_load(open('$OUTPUT_DIR/analysis/system_context.yaml')); print('  [OK] YAML is valid')"

echo ""
echo "6. Checking HTML contains expected content..."
echo "--------------------------------------------"
if grep -qi "d3" "$OUTPUT_DIR/analysis/visualization.html" && \
   grep -q "nodes" "$OUTPUT_DIR/analysis/visualization.html"; then
    echo "  [OK] HTML contains D3.js and graph data"
else
    echo "  [FAIL] HTML missing expected content"
    exit 1
fi

echo ""
echo "============================================"
echo "All integration tests passed!"
echo "============================================"
echo ""
echo "Output files available at: $OUTPUT_DIR"
echo ""
echo "To view visualization:"
echo "  firefox $OUTPUT_DIR/analysis/visualization.html"
echo ""
