#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Module 1: Tool Hierarchy Construction ==="

cd "$SCRIPT_DIR/module1_tool_hierarchy_construction"

# Rank tools and define tier-1 / tier-2 split
python define_hierarchy.py

# Identify test samples where tier-1 tools disagree
python find_test_disagreement.py

echo "=== Module 1 Done ==="
