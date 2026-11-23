#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_test_disagreement.py: Split test samples into unanimous / disagreement
based on tier1 tool predictions defined in hierarchy.json.

Usage:
    python find_test_disagreement.py
"""

import sys
import json
from pathlib import Path

BASE_DIR         = Path(__file__).parent
CODES_DIR        = BASE_DIR.parent
HIERARCHY_PATH   = BASE_DIR / "hierarchy.json"
OUTPUT_DIR       = BASE_DIR / "test_splits"
DEMO_DATA_DIR = CODES_DIR / "data" / "demo"

sys.path.insert(0, str(CODES_DIR / "utils"))


def load_preds(tool, label=None):
    """Load test predictions for a tool from data/demo/test.
    If label is given (1 or 0), filter by that label only.
    """
    path = DEMO_DATA_DIR / "test" / f"{tool}.jsonl"
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if label is None or str(r["label"]) == str(label):
                preds.append(r)
    return preds


def main():
    with open(HIERARCHY_PATH) as f:
        hierarchy = json.load(f)
    tier1 = hierarchy["tier1"]
    print(f"Tier1 tools: {tier1}")

    tier1_preds_pos = {t: {r["idx"]: str(r["tool_prediction"]) for r in load_preds(t, label=1)} for t in tier1}
    tier1_preds_neg = {t: {r["idx"]: str(r["tool_prediction"]) for r in load_preds(t, label=0)} for t in tier1}

    ref_pos = load_preds(tier1[0], label=1)
    ref_neg = load_preds(tier1[0], label=0)

    unanimous, disagreement = [], []
    for sample in ref_pos + ref_neg:
        idx = sample["idx"]
        is_pos = str(sample["label"]) == "1"
        preds_map = tier1_preds_pos if is_pos else tier1_preds_neg
        idx_lookup = idx      # use original data/demo idx for both pos and neg lookups
        votes = {preds_map[t].get(idx_lookup) for t in tier1}
        entry = {k: sample[k] for k in sample}
        entry["tier1_predictions"] = {t: preds_map[t].get(idx_lookup) for t in tier1}
        entry["tier1_consensus"] = list(votes)[0] if len(votes) == 1 else None
        if len(votes) == 1:
            unanimous.append(entry)
        else:
            disagreement.append(entry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(OUTPUT_DIR / "unanimous.jsonl", unanimous)
    write_jsonl(OUTPUT_DIR / "disagreement.jsonl", disagreement)

    print(f"unanimous:    {len(unanimous)} samples")
    print(f"disagreement: {len(disagreement)} samples")
    print(f"Saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
