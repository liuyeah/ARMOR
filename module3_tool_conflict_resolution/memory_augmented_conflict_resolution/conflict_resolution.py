#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conflict_resolution.py

DRFP+FAISS similarity retrieval to find reference demos from memory,
then LLM resolves tool conflicts on test disagreement samples.

Input:
  {CAND_PREP_DIR}/processed/test.jsonl
  {MEM_DIR}/memory/top_t_{L}/dev.jsonl

Output:
  resolved/top_t_{L}/k{top_k}/test/resolved.jsonl

Usage:
  python conflict_resolution.py
"""

import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent

import yaml as _yaml
with open(CODES_DIR / "config.yaml") as _f:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_yaml.safe_load(_f).get("cuda_visible_devices", "0"))

import random
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import faiss
from drfp import DrfpEncoder

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import (
    load_jsonl, save_jsonl, load_pipe, CFG,
    prepare_candidates, llm_choose_tool,
    top_t_str,
)

CAND_PREP_DIR = (CODES_DIR / "module2_utility_aware_tool_prioritization"
                 / "pattern_based_tool_selection" / "candidate_preparation")
MEM_DIR       = CODES_DIR / "module3_tool_conflict_resolution" / "tool_conflict_memory"


# ── DRFP / FAISS ──────────────────────────────────────────────────────────────

def encode_reactions(smiles_list: list) -> np.ndarray:
    fps = DrfpEncoder.encode(smiles_list)
    return np.packbits(fps, axis=1)


def build_faiss_index(packed: np.ndarray):
    dim   = packed.shape[1] * 8
    index = faiss.IndexBinaryFlat(dim)
    index.add(packed)
    return index


# ── Memory loading ─────────────────────────────────────────────────────────────

def load_memory_grouped(memory_path: Path, tag: str):
    """
    Load memory demos and group by idx.
    Returns (smiles_list, namespaced_idx_list, namespaced_demo_groups).
    """
    if not memory_path.exists():
        print(f"  [WARN] memory not found: {memory_path}")
        return [], [], {}

    demos = load_jsonl(memory_path)
    demo_groups = defaultdict(list)
    for d in demos:
        demo_groups[d["idx"]].append(d)

    idx_list    = sorted(demo_groups.keys())
    smiles_list = []
    for idx in idx_list:
        rxn = demo_groups[idx][0]["reaction"]
        smiles_list.append(f"{rxn['reactants']}>>{rxn['product']}")

    namespaced_groups = {f"{tag}_{k}": v for k, v in demo_groups.items()}
    namespaced_idx    = [f"{tag}_{i}" for i in idx_list]

    print(f"  Loaded {len(demos)} demos ({len(idx_list)} unique reactions) from {memory_path.name}")
    return smiles_list, namespaced_idx, namespaced_groups


# ── Retrieval ──────────────────────────────────────────────────────────────────

def pick_best_demo(demo_group: list):
    """Randomly pick a demo from the group."""
    if not demo_group:
        return None
    return random.choice(demo_group)


def retrieve_drfp(rxn_smiles: str, index, idx_list: list, demo_groups: dict,
                  k: int) -> list:
    if index is None or not idx_list:
        return []
    packed_q = encode_reactions([rxn_smiles])
    _, I     = index.search(packed_q, k=min(k, len(idx_list)))
    demos = []
    for pos in I[0]:
        if pos < 0 or pos >= len(idx_list):
            continue
        key  = idx_list[pos]
        best = pick_best_demo(demo_groups[key])
        if best is not None:
            demos.append(best)
    return demos


# ── Core run ───────────────────────────────────────────────────────────────────

def run(pipe, top_t, top_k, llm, args,
        index_all, idx_all, groups_all):
    tt_str = top_t_str(top_t)

    proc_path = CAND_PREP_DIR / "processed" / "test.jsonl"
    if not proc_path.exists():
        print(f"  [SKIP] processed file not found: {proc_path}")
        return

    out_path = BASE_DIR / "resolved" / f"top_t_{tt_str}" / f"k{top_k}" / "test" / "resolved.jsonl"
    if out_path.exists():
        print(f"  [SKIP] already exists: {out_path}")
        return

    rows     = load_jsonl(proc_path)
    out_rows = []

    for row in tqdm(rows, desc="conflict resolution"):
        cands = prepare_candidates(row["candidates"], top_t)
        if not cands:
            continue

        votes = [c["tool_prediction"] for c in cands]
        if votes.count("1") == len(cands) or votes.count("0") == len(cands):
            continue  # top-L already unanimous — skip, no conflict to resolve

        rxn_smiles = f"{row['reactants']}>>{row['product']}"
        demos = retrieve_drfp(rxn_smiles, index_all, idx_all, groups_all, top_k)

        decision = llm_choose_tool(
            pipe=pipe,
            reactants=row["reactants"],
            product=row["product"],
            raw_cands=cands,
            demos=demos,
            with_confidence=True,
            tier_bias="none",
            demo_format="full",
            show_tiers=True,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )

        out_rows.append({
            "idx":    row["idx"],
            "tool":   decision["tool"],
            "reason": decision["reason"],
            "reaction": {
                "reactants": row["reactants"],
                "product":   row["product"],
                "label":     row.get("label"),
            },
            "meta": {
                "top_t":     top_t,
                "top_k":     top_k,
                "n_demos":   len(demos),
                "model_key": llm,
            },
            "_raw": decision.get("_raw", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_rows, out_path)
    print(f"  Saved {len(out_rows)} rows → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens",  type=int, default=512)
    ap.add_argument("--retries",     type=int, default=3)
    args = ap.parse_args()

    top_t = CFG["selection"]["L"]
    top_k = CFG["selection"]["top_k"]
    llm   = CFG["llm"]
    tt_str = top_t_str(top_t)

    print(f"=== conflict_resolution: top_t={top_t}, top_k={top_k} ===")

    mem_base = MEM_DIR / "memory" / f"top_t_{tt_str}"
    smiles_all, idx_all, groups_all = load_memory_grouped(mem_base / "dev.jsonl", "dev")
    index_all = build_faiss_index(encode_reactions(smiles_all)) if smiles_all else None
    print(f"  FAISS index: {len(smiles_all)} unique reactions")

    pipe = load_pipe(llm)
    run(pipe, top_t, top_k, llm, args, index_all, idx_all, groups_all)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
