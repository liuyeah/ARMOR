#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_ruleset.py

Generate rule_set.json from rule_stats.json.
Reads tau_3 (min confidence) and top_p (max rules per tool) from config.yaml.

Output: rule_set.json in the same directory as this script.

Usage:
  python generate_ruleset.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR        = Path(__file__).parent
CODES_DIR       = BASE_DIR.parent.parent.parent
RULE_STATS_PATH = BASE_DIR / "rule_stats.json"

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG


def load_rule_stats():
    stats = []
    for line in RULE_STATS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            stats.append(json.loads(line))
    return stats


def build_ruleset(stats, conf_min, top_k):
    by_tool = defaultdict(list)
    for s in stats:
        by_tool[s["tool"]].append(s)

    result = []
    for tool, rules in sorted(by_tool.items()):
        rules_sorted = sorted(rules, key=lambda x: -x["confidence"])
        if conf_min is not None:
            rules_sorted = [r for r in rules_sorted if r["confidence"] >= conf_min]
        if top_k is not None:
            rules_sorted = rules_sorted[:top_k]
        for r in rules_sorted:
            result.append({
                "tool":       r["tool"],
                "rule_name":  r["rule_name"],
                "confidence": r["confidence"],
                "match_rate": r["match_rate"],
            })
    return result


def main():
    tau_3 = CFG["ruleset"]["tau_3"]
    top_p = CFG["ruleset"]["top_p"]

    stats   = load_rule_stats()
    ruleset = build_ruleset(stats, tau_3, top_p)

    out_path = BASE_DIR / "rule_set.json"
    out_path.write_text(
        json.dumps(ruleset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_tool = defaultdict(list)
    for r in ruleset:
        by_tool[r["tool"]].append(r)

    print(f"tau_3={tau_3}, top_p={top_p}")
    print(f"Total rules: {len(ruleset)}, Tools with rules: {len(by_tool)}")
    print(f"{'tool':<25} {'n_rules':>7}  {'top_conf':>9}  {'top_rule'}")
    for tool, rules in sorted(by_tool.items()):
        top = rules[0]
        print(f"{tool:<25} {len(rules):>7}  {top['confidence']:>9.3f}  {top['rule_name'][:40]}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
