#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subset_construction.py

For each tool and each N in N_LIST, generate M random subsets of size ~4N
(N pos_correct + N neg_correct + N pos_wrong + N neg_wrong), sampling from
the tool's dev predictions in example_data/dev/.
N_LIST and M are read from config.yaml.

Output: outputs/N={N}/{tool}/sample_{i}.jsonl

Usage:
  python subset_construction.py
  python subset_construction.py --tool BERT
"""

import sys
import json
import random
import argparse
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent.parent

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST

EXAMPLE_DATA_DIR = CODES_DIR / "example_data"
OUTPUT_BASE      = BASE_DIR / "outputs"


def sample_up_to(lst, n, rng):
    return rng.sample(lst, min(n, len(lst)))


def load_dev_samples(tool_name):
    """Returns (pos_correct, pos_wrong, neg_correct, neg_wrong) lists."""
    path = EXAMPLE_DATA_DIR / "dev" / f"{tool_name}.jsonl"
    pos_correct, pos_wrong, neg_correct, neg_wrong = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r     = json.loads(line)
            label = str(r.get("label", ""))
            pred  = str(r.get("tool_prediction", "NA"))
            if label == "1":
                (pos_correct if pred == "1" else pos_wrong).append(r)
            else:
                (neg_correct if pred == "0" else neg_wrong).append(r)
    return pos_correct, pos_wrong, neg_correct, neg_wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default=None, help="Single tool to process (default: all from config)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    N_list = CFG["subset"]["N_list"]
    M      = CFG["subset"]["M"]
    tools  = [args.tool] if args.tool else TOOL_LIST
    rng    = random.Random(args.seed)

    for tool_name in tools:
        pos_correct, pos_wrong, neg_correct, neg_wrong = load_dev_samples(tool_name)
        print(f"{tool_name}: pos_correct={len(pos_correct)}, pos_wrong={len(pos_wrong)}, "
              f"neg_correct={len(neg_correct)}, neg_wrong={len(neg_wrong)}")

        for N in N_list:
            out_dir = OUTPUT_BASE / f"N={N}" / tool_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i in range(M):
                subset = (
                    sample_up_to(pos_correct, N, rng) +
                    sample_up_to(neg_correct, N, rng) +
                    sample_up_to(pos_wrong,   N, rng) +
                    sample_up_to(neg_wrong,   N, rng)
                )
                out_path = out_dir / f"sample_{i}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for ex in subset:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"  N={N}: {M} subsets → {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
