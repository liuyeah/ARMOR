#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_matching.py

After running match_rules.py for every tool, aggregate the per-rule per-sample
result files into one matching_res.json per phase (dev / test).

match_rules.py writes:
  outputs/{tool_name}/{phase}/CORRECT_RULES/RULE_{n}: {rule_name}.json
  Each file: one JSON line per sample → {idx, belongs_to_rule, confidence, reason}
  idx is the combined index from disagreement.jsonl (pos: 0-49, neg: 50-99)

This script writes:
  outputs/{phase}/matching_res.json
  N_COMBINED lines (one per combined idx 0..N_COMBINED-1).
  Line i corresponds to combined_idx i; empty list [] if no match.

  Candidate dict schema:
    {tool, rule_name, rule_explanation, rule_confidence, tool_prediction}

Usage:
  python aggregate_matching.py
  python aggregate_matching.py --phase dev
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent.parent

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST

PRUNE_DIR        = (CODES_DIR / "module2_utility_aware_tool_prioritization"
                    / "reaction_specific_tool_utility_assessment"
                    / "pattern_consolidation" / "prune_outputs")
DEMO_DATA_DIR = CODES_DIR / "data" / "demo"

LLM = CFG["llm"]

PHASES = ["dev", "test"]
PHASE_TO_SPLIT = {"dev": "dev", "test": "test"}


def get_n_combined(split_type):
    """Infer total sample count from data/demo idx values."""
    max_idx = -1
    for tool_name in TOOL_LIST:
        path = DEMO_DATA_DIR / split_type / f"{tool_name}.jsonl"
        if not path.exists():
            continue
        for r in load_jsonl(path):
            max_idx = max(max_idx, int(r["idx"]))
    return max_idx + 1


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tool_preds(split_type, tool_name):
    """Returns {combined_idx: pred_str} for all samples in the split."""
    path = DEMO_DATA_DIR / split_type / f"{tool_name}.jsonl"
    if not path.exists():
        return {}
    return {int(r["idx"]): str(r.get("tool_prediction", "NA"))
            for r in load_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None, choices=PHASES,
                    help="Process a single phase (default: both dev and test)")
    args = ap.parse_args()

    phases_to_run = [args.phase] if args.phase else PHASES

    for phase in phases_to_run:
        split_type = PHASE_TO_SPLIT[phase]
        out_dir    = BASE_DIR / "outputs" / phase
        out_dir.mkdir(parents=True, exist_ok=True)

        n_combined   = get_n_combined(split_type)
        matching_res = [[] for _ in range(n_combined)]
        print(f"{phase}: {n_combined} total samples")

        for tool_name in TOOL_LIST:
            rules_path = PRUNE_DIR / f"valid_by_{LLM}" / f"{tool_name}.json"
            if not rules_path.exists():
                continue
            doc   = json.loads(rules_path.read_text(encoding="utf-8"))
            rules = doc.get("often_correct", [])

            tool_preds = load_tool_preds(split_type, tool_name)

            rule_dir = BASE_DIR / "outputs" / tool_name / phase / "CORRECT_RULES"
            if not rule_dir.exists():
                continue

            for r_idx, rule in enumerate(rules):
                rule_name_safe = rule.get("name", "").replace("/", "_")
                rule_file = rule_dir / f"RULE_{r_idx + 1}: {rule_name_safe}.json"
                if not rule_file.exists():
                    continue

                results = load_jsonl(rule_file)
                llm_val = rule.get("llm_validation", {})
                rule_confidence = (
                    llm_val.get("agreement_rate", 0.0)
                    if isinstance(llm_val, dict)
                    else float(rule.get("confidence", 0.0))
                )

                for res in results:
                    if not res.get("belongs_to_rule", False):
                        continue
                    combined_idx = int(res["idx"])
                    if not (0 <= combined_idx < n_combined):
                        continue
                    matching_res[combined_idx].append({
                        "tool":             tool_name,
                        "rule_name":        rule.get("name", ""),
                        "rule_explanation": rule.get("explanation", ""),
                        "rule_confidence":  round(float(rule_confidence), 4),
                        "tool_prediction":  tool_preds.get(combined_idx, "NA"),
                    })

        out_path = out_dir / "matching_res.json"
        with open(out_path, "w", encoding="utf-8") as f:
            for cands in matching_res:
                f.write(json.dumps(cands, ensure_ascii=False) + "\n")

        n_matched = sum(1 for c in matching_res if c)
        print(f"{phase}: {n_combined} lines, "
              f"{n_matched} with ≥1 match → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
