#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M2="$SCRIPT_DIR/module2_utility_aware_tool_prioritization"

echo "=== Module 2: Utility-Aware Tool Prioritization ==="

# --- Pattern Extraction ---
cd "$M2/reaction_specific_tool_utility_assessment/pattern_extraction"

# Sample random subsets of dev examples per tool
python subset_construction.py

# Use LLM to detect reaction rules from each subset
python rule_detection.py

# --- Pattern Refinement ---
cd "$M2/reaction_specific_tool_utility_assessment/pattern_refinement"

# Compute align_score for each rule (no LLM)
python rule_verify.py

# Validate each rule with LLM; only rules with align_score >= tau_1 are validated
python validate_rules.py

# Keep rules where align_score >= tau_1 and coverage_score >= tau_2
python rule_filtering.py

# --- Pattern Consolidation ---
cd "$M2/reaction_specific_tool_utility_assessment/pattern_consolidation"

# Merge per-(tool, N) filtered outputs into one file per tool
python aggregation.py

# Deduplicate same-named rules across subsets via LLM
python prune_rules.py

# --- Dev Rule Matching ---
cd "$M2/pattern_based_tool_selection/rule_matching"

# Match pruned rules against dev disagreement samples
python match_rules.py --phase dev
python aggregate_matching.py --phase dev

# --- Ruleset Generation ---
cd "$M2/reaction_specific_tool_utility_assessment/pattern_consolidation"

# Compute per-rule dev accuracy and select top rules per tool (tau_3 / top_p)
python compute_rule_stats.py
python generate_ruleset.py

# --- Test Rule Matching ---
cd "$M2/pattern_based_tool_selection/rule_matching"

# Match selected rules against test disagreement samples
python match_rules.py --phase test
python aggregate_matching.py --phase test

# --- Candidate Preparation ---
cd "$M2/pattern_based_tool_selection/candidate_preparation"

# Build ranked candidate lists for each disagreement sample
python preprocess.py

echo "=== Module 2 Done ==="
