#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_memory.py

Build guided_strict demonstrations from dev disagreement samples.

Candidate preparation (applied at runtime, not stored in processed data):
  1. top-1-per-tool: among each tool's matched rules, keep highest rule_confidence
  2. top-L: keep top_L tools by their best-rule confidence
  Either way, candidates are sorted by confidence descending.

Output:
  memory/guided_strict/tiers_true/top_t_{L}/dev[_chunk{id}].jsonl

Demo schema (compatible with utils._format_demo_*):
  {
    "idx":        <original combined idx>,
    "demo_id":    "<idx>#<sub_id>",
    "reaction":   {"reactants": ..., "product": ...},
    "candidates": [...],
    "demo": {
      "tool":         ...,
      "evidence":     [...],
      "elimination":  [...],
      "final_reason": ...
    },
    "meta": {
      "memory_type": "guided_strict",
      "show_tiers":  true,
      "top_t":       int|null,
      "model_key":   ...,
      "label":       "0"|"1",
      "gold_tools":  [...],
      "correct":     true
    },
    "_raw": ...
  }

Usage:
  python build_memory.py
  python build_memory.py --chunk_id 0 --n_chunks 4
"""

import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent

import yaml as _yaml
with open(CODES_DIR / "config.yaml") as _f:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_yaml.safe_load(_f).get("cuda_visible_devices", "0"))

import argparse
from tqdm import tqdm
sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import (
    load_jsonl, save_jsonl, load_pipe, CFG,
    prepare_candidates,
    llm_guided_demo,
    top_t_str,
)

CAND_PREP_DIR = (CODES_DIR / "module2_utility_aware_tool_prioritization"
                 / "pattern_based_tool_selection" / "candidate_preparation")


def get_gold_tools(cands, label):
    """Tools in cands whose tool_prediction matches label."""
    return sorted({c["tool"] for c in cands if c.get("tool_prediction") == label})


def get_neg_tools(cands, gold_tool, label):
    """guided_strict: tools with wrong prediction, excluding gold_tool."""
    return sorted({
        c["tool"] for c in cands
        if c.get("tool") != gold_tool and c.get("tool_prediction") != label
    })


def get_out_path(top_t, chunk_id, n_chunks):
    out_dir = BASE_DIR / "memory" / f"top_t_{top_t_str(top_t)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_chunks > 1:
        return out_dir / f"dev_chunk{chunk_id}.jsonl"
    return out_dir / "dev.jsonl"


def apply_chunking(rows, chunk_id, n_chunks):
    if n_chunks <= 1:
        return rows, None, None
    chunk_size = (len(rows) + n_chunks - 1) // n_chunks
    start = chunk_id * chunk_size
    end   = min(start + chunk_size, len(rows))
    return rows[start:end], start, end


def run_guided(pipe, top_t, llm, args):
    proc_path = CAND_PREP_DIR / "processed" / "dev.jsonl"
    out_path  = get_out_path(top_t, args.chunk_id, args.n_chunks)

    if out_path.exists():
        print(f"[SKIP] already exists: {out_path}")
        return

    rows = load_jsonl(proc_path)
    rows, start, end = apply_chunking(rows, args.chunk_id, args.n_chunks)
    if start is not None:
        print(f"  Chunk {args.chunk_id}/{args.n_chunks}: rows [{start}, {end})")

    out_rows = []

    for row in tqdm(rows, desc="guided_strict"):
        cands = prepare_candidates(row["candidates"], top_t)
        if not cands:
            continue

        votes = [c["tool_prediction"] for c in cands]
        if votes.count("1") == len(cands) or votes.count("0") == len(cands):
            continue  # top-L already unanimous → no conflict to learn from

        label      = row["label"]
        gold_tools = get_gold_tools(cands, label)
        if not gold_tools:
            continue

        sub_id = 0
        for gold_tool in gold_tools[:args.max_gold_tools]:
            neg_tools = get_neg_tools(cands, gold_tool, label)

            if not neg_tools:
                continue

            res = llm_guided_demo(
                pipe=pipe,
                reactants=row["reactants"],
                product=row["product"],
                cands=cands,
                gold_tool=gold_tool,
                neg_tools=neg_tools,
                show_tiers=True,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            if res is None:
                continue

            out_rows.append({
                "idx":      row["idx"],
                "demo_id":  f"{row['idx']}#{sub_id}",
                "reaction": {"reactants": row["reactants"], "product": row["product"]},
                "candidates": cands,
                "demo":     res["demo"],
                "meta": {
                    "memory_type": "guided_strict",
                    "show_tiers":  True,
                    "top_t":       top_t,
                    "model_key":   llm,
                    "label":       label,
                    "gold_tool":   gold_tool,
                    "gold_tools":  gold_tools,
                    "correct":     True,
                },
                "_raw": res.get("_raw", ""),
            })
            sub_id += 1

    save_jsonl(out_rows, out_path)
    print(f"  Saved {len(out_rows)} guided demos → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens",     type=int, default=512)
    ap.add_argument("--retries",        type=int, default=3)
    ap.add_argument("--max_gold_tools", type=int, default=100,
                    help="Max gold tools per sample (default: all)")
    ap.add_argument("--chunk_id",       type=int, default=0)
    ap.add_argument("--n_chunks",       type=int, default=1)
    args = ap.parse_args()

    top_t = CFG["selection"]["L"]
    llm   = CFG["llm"]

    proc_check = CAND_PREP_DIR / "processed" / "dev.jsonl"
    if not proc_check.exists():
        raise FileNotFoundError(
            f"Processed data not found: {proc_check}\n"
            "Run preprocess.py first."
        )

    print(f"=== build_memory: top_t={top_t} ===")

    pipe = load_pipe(llm)
    run_guided(pipe, top_t, llm, args)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
