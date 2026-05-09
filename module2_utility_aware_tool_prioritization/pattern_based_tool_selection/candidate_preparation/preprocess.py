#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py

For the selected rule set, process matching_res into clean per-sample candidate lists.

For each disagreement sample in dev/test splits:
  1. Load all matched candidates from matching_res
  2. Filter: only keep (tool, rule_name) pairs in rule_set.json
  3. Save to processed/{phase}.jsonl

Output format per line:
  {
    "idx":        <original combined idx from disagreement.jsonl>,
    "label":      "0" or "1",
    "reactants":  "...",
    "product":    "...",
    "candidates": [
      {"tool": ..., "rule_name": ..., "rule_explanation": ...,
       "rule_confidence": ..., "tool_prediction": ...},
      ...
    ]
  }

Usage:
  python preprocess.py
"""

import sys
import json
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CODES_DIR  = BASE_DIR.parent.parent.parent

RULESET_GEN_DIR = (BASE_DIR.parent.parent
                   / "reaction_specific_tool_utility_assessment"
                   / "pattern_consolidation")
HIER_DEF_DIR    = CODES_DIR / "module1_tool_hierarchy_construction"
MATCHING_RES    = CODES_DIR / "module2_utility_aware_tool_prioritization" / \
                  "pattern_based_tool_selection" / "rule_matching" / "outputs"

SPLITS = [
    ("dev",  HIER_DEF_DIR / "splits"      / "disagreement.jsonl"),
    ("test", HIER_DEF_DIR / "test_splits" / "disagreement.jsonl"),
]


def load_ruleset():
    path = RULESET_GEN_DIR / "rule_set.json"
    if not path.exists():
        raise FileNotFoundError(
            f"rule_set.json not found: {path}\nRun generate_ruleset.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(r["tool"], r["rule_name"]) for r in data}


def load_disagreement(dis_path):
    rxn_map = {}
    for line in dis_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rxn_map[int(row["idx"])] = row
    return rxn_map


def process_split(valid_pairs, phase, dis_path):
    mr_path = MATCHING_RES / phase / "matching_res.json"
    if not mr_path.exists():
        print(f"  [SKIP] matching_res not found: {mr_path}")
        return

    rxn_map  = load_disagreement(dis_path)
    out_path = BASE_DIR / "processed" / f"{phase}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows  = []
    lines = mr_path.read_text(encoding="utf-8").splitlines()
    for combined_idx, line in enumerate(lines):
        if not line.strip():
            continue
        rxn = rxn_map.get(combined_idx)
        if rxn is None:
            continue

        raw_cands = json.loads(line)
        if not raw_cands:
            rows.append({
                "idx":        combined_idx,
                "label":      str(int(rxn["label"])),
                "reactants":  rxn["reactants"],
                "product":    rxn["product"],
                "candidates": [],
            })
            continue

        filtered = [
            c for c in raw_cands
            if (c.get("tool", ""), c.get("rule_name", "")) in valid_pairs
        ]

        clean_cands = []
        for c in filtered:
            clean_cands.append({
                "tool":             c["tool"],
                "rule_name":        c["rule_name"],
                "rule_explanation": c.get("rule_explanation", ""),
                "rule_confidence":  round(float(c.get("rule_confidence", 0.0)), 4),
                "tool_prediction":  str(c.get("tool_prediction", "")),
            })

        rows.append({
            "idx":        combined_idx,
            "label":      str(int(rxn["label"])),
            "reactants":  rxn["reactants"],
            "product":    rxn["product"],
            "candidates": clean_cands,
        })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    non_empty = sum(1 for r in rows if r["candidates"])
    print(f"  {phase}: {len(rows)} samples, "
          f"{non_empty} with candidates → {out_path}")


def main():
    valid_pairs = load_ruleset()
    print(f"=== preprocess: valid (tool, rule) pairs: {len(valid_pairs)} ===")

    for phase, dis_path in SPLITS:
        process_split(valid_pairs, phase, dis_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
