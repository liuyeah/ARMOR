#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_rule_stats.py

For each (tool, rule_name) pair found in dev matching_res,
compute:
  - confidence   : rule_confidence value (from candidate entry)
  - match_count  : number of dev disagreement samples that match this rule
  - match_rate   : match_count / total_dev_disagreement_samples

Output: rule_stats.json (one JSON object per line)

Usage:
  python compute_rule_stats.py
"""

import json
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent.parent

MATCHING_RES_DIR = CODES_DIR / "module2_utility_aware_tool_prioritization" / \
                   "pattern_based_tool_selection" / "rule_matching" / "outputs"
DIS_PATH         = CODES_DIR / "module1_tool_hierarchy_construction" / "splits" / "disagreement.jsonl"
OUT_PATH         = BASE_DIR / "rule_stats.json"


def load_dis_idxs():
    dis_idxs = set()
    for line in DIS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dis_idxs.add(int(r["idx"]))
    return dis_idxs


def main():
    dis_idxs  = load_dis_idxs()
    dis_total = len(dis_idxs)
    print(f"Dev disagreement: total={dis_total}")

    stats: dict = {}

    path = MATCHING_RES_DIR / "dev" / "matching_res.json"
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return
    lines = path.read_text().splitlines()
    for combined_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cands = json.loads(line)
        in_dis = combined_idx in dis_idxs
        for c in cands:
            tool      = c.get("tool", "")
            rule_name = c.get("rule_name", "")
            conf      = float(c.get("rule_confidence", 0.0))
            key = f"{tool}:{rule_name}"
            if key not in stats:
                stats[key] = {"tool": tool, "rule_name": rule_name,
                              "confidence": conf, "match_count": 0}
            if in_dis:
                stats[key]["match_count"] += 1

    rows = []
    for key, s in sorted(stats.items()):
        rows.append({
            "key":         key,
            "tool":        s["tool"],
            "rule_name":   s["rule_name"],
            "confidence":  s["confidence"],
            "match_count": s["match_count"],
            "match_rate":  s["match_count"] / dis_total,
        })

    rows.sort(key=lambda r: (-r["match_rate"], r["key"]))

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} (tool, rule) entries → {OUT_PATH}")
    rates = [r["match_rate"] for r in rows]
    print(f"match_rate range: [{min(rates):.4f}, {max(rates):.4f}]")
    confs = [r["confidence"] for r in rows]
    print(f"confidence range: [{min(confs):.4f}, {max(confs):.4f}]")


if __name__ == "__main__":
    main()
