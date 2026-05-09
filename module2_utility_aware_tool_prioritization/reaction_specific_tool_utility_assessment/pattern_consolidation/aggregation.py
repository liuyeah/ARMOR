#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregation.py

Collect filtered rule patterns from all (N, backbone) combinations
for each tool, and merge into one JSON file per (tool, valid_llm).

Reads the output of rule_filtering.py (filtered + per-(tool,N,backbone) aggregated).

Input:
  pattern_refinement/outputs_json_format/N={N}/{tool_name}/
    {llm}_recognition_pattern_validated_by_{valid_llm}.json

Output:
  aggregation_outputs/valid_by_{valid_llm}/{tool_name}.json

Usage:
  python aggregation.py
"""

import sys
import json
from pathlib import Path

BASE_DIR       = Path(__file__).parent
CODES_DIR      = BASE_DIR.parent.parent.parent
VALIDATION_DIR = BASE_DIR.parent / "pattern_refinement" / "outputs_json_format"
OUTPUT_BASE    = BASE_DIR / "aggregation_outputs"

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST


def main():
    llm    = CFG["llm"]
    N_list = CFG["subset"]["N_list"]

    out_dir = OUTPUT_BASE / f"valid_by_{llm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tool_name in TOOL_LIST:
        output_data = {"tool_name": tool_name, "often_correct": []}

        for N in N_list:
            input_path = (VALIDATION_DIR / f"N={N}" / tool_name
                          / f"{llm}_recognition_pattern_validated_by_{llm}.json")
            if not input_path.exists():
                continue
            with open(input_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            for c_r in doc.get("often_correct", []):
                c_r["resource"] = str(input_path)
                output_data["often_correct"].append(c_r)

        out_path = out_dir / f"{tool_name}.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(output_data, f_out, indent=2, ensure_ascii=False)
        print(f"{tool_name} | valid_by_{llm} | "
              f"rules: {len(output_data['often_correct'])} → {out_path}")


if __name__ == "__main__":
    main()
