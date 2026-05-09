#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3="$SCRIPT_DIR/module3_tool_conflict_resolution"

echo "=== Module 3: Tool Conflict Resolution ==="

# Generate guided demonstrations from dev disagreement samples
cd "$M3/tool_conflict_memory"
python build_memory.py

# Retrieve similar dev demos via FAISS and use LLM to select the best tool
cd "$M3/memory_augmented_conflict_resolution"
python conflict_resolution.py

echo "=== Module 3 Done ==="
