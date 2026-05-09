#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Evaluation ==="

cd "$SCRIPT_DIR/evaluation"
python eval_test.py

echo "=== Evaluation Done ==="
