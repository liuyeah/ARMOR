#!/usr/bin/env python3
"""
eval_test.py

Evaluates performance on the example_data test set.
  1. Each individual tool
  2. ARMOR (our method)

Metrics: overall_acc (%), pos_acc, neg_acc, F1 for feasible & infeasible, MCC

Usage:
  python eval_test.py
"""

import sys
import json
import math
from pathlib import Path

BASE      = Path(__file__).parent
CODES_DIR = BASE.parent
sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import TOOL_LIST, load_jsonl, save_jsonl, CFG

EXAMPLE_DATA_DIR = CODES_DIR / "example_data"
TEST_SPLITS      = CODES_DIR / "module1_tool_hierarchy_construction" / "test_splits"
DEV_SPLITS       = CODES_DIR / "module1_tool_hierarchy_construction" / "splits"
SEL_DIR          = (CODES_DIR / "module3_tool_conflict_resolution"
                    / "memory_augmented_conflict_resolution"
                    / "resolved")
CAND_PREP_DIR    = (CODES_DIR / "module2_utility_aware_tool_prioritization"
                    / "pattern_based_tool_selection" / "candidate_preparation")


def compute_metrics(tp, fn, tn, fp):
    pos_n    = tp + fn
    neg_n    = tn + fp
    total    = pos_n + neg_n
    overall  = (tp + tn) / total * 100 if total else 0.0
    pos_acc  = tp / pos_n * 100 if pos_n else 0.0
    neg_acc  = tn / neg_n * 100 if neg_n else 0.0
    prec_pos = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    prec_neg = tn / (tn + fn) * 100 if (tn + fn) else 0.0
    d_f1p    = 2 * tp + fp + fn
    d_f1n    = 2 * tn + fn + fp
    f1_pos   = 2 * tp / d_f1p * 100 if d_f1p else 0.0
    f1_neg   = 2 * tn / d_f1n * 100 if d_f1n else 0.0
    d_mcc    = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc      = (tp * tn - fp * fn) / math.sqrt(d_mcc) if d_mcc else 0.0
    return {
        "overall":  round(overall, 2),
        "prec_pos": round(prec_pos, 2),
        "pos_acc":  round(pos_acc, 2),
        "f1_pos":   round(f1_pos, 2),
        "prec_neg": round(prec_neg, 2),
        "neg_acc":  round(neg_acc, 2),
        "f1_neg":   round(f1_neg, 2),
        "mcc":      round(mcc, 4),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def compute_metrics_from_preds(preds, labels):
    """preds: {idx: pred_str}, labels: {idx: label_str}"""
    tp = fn = tn = fp = 0
    for idx, label in labels.items():
        pred = preds.get(idx, "NA")
        if label == "1":
            if pred == "1": tp += 1
            else:           fn += 1
        elif label == "0":
            if pred == "0": tn += 1
            else:           fp += 1
    return compute_metrics(tp, fn, tn, fp)


def load_raw_preds(tool):
    """Returns {combined_idx: pred_str}."""
    path = EXAMPLE_DATA_DIR / "test" / f"{tool}.jsonl"
    preds = {}
    if not path.exists():
        return preds
    for r in load_jsonl(path):
        preds[int(r["idx"])] = str(r.get("tool_prediction", "NA"))
    return preds


def load_test_labels():
    """Returns {combined_idx: label_str} from example_data/test/."""
    path = EXAMPLE_DATA_DIR / "test" / "Prompt_T3Q.jsonl"
    return {int(r["idx"]): str(r.get("label", "")) for r in load_jsonl(path)}


# ── Individual tools ──────────────────────────────────────────────────────────

ALL_TOOLS = TOOL_LIST


def eval_single_tool(tool, labels):
    preds = load_raw_preds(tool)
    return compute_metrics_from_preds(preds, labels)


# ── Hierarchy method ──────────────────────────────────────────────────────────

def _load_selection(path):
    if not Path(path).exists():
        return {}
    return {int(r["idx"]): r["tool"] for r in load_jsonl(path)}


def make_predictions(sel_path, proc_path, L=CFG["selection"]["L"], fallback_tool="Prompt_T3Q"):
    uni = load_jsonl(TEST_SPLITS / "unanimous.jsonl")
    dis = load_jsonl(TEST_SPLITS / "disagreement.jsonl")

    sel         = _load_selection(sel_path)
    cand_lookup = {}
    if Path(proc_path).exists():
        for r in load_jsonl(proc_path):
            cand_lookup[int(r["idx"])] = r.get("candidates", [])

    fallback   = load_raw_preds(fallback_tool)
    tool_cache = {}

    def get_tool_pred(tool, idx):
        if tool not in tool_cache:
            tool_cache[tool] = load_raw_preds(tool)
        return tool_cache[tool].get(idx, "NA")

    preds = {}

    for r in uni:
        idx = int(r["idx"])
        c   = r.get("tier1_consensus")
        preds[idx] = str(c) if c is not None else fallback.get(idx, "NA")

    for r in dis:
        idx   = int(r["idx"])
        cands = cand_lookup.get(idx, [])
        best  = {}
        for c in cands:
            t = c["tool"]
            if t not in best or c["rule_confidence"] > best[t]["rule_confidence"]:
                best[t] = c
        top = sorted(best.values(), key=lambda c: c["rule_confidence"], reverse=True)[:L]
        if top:
            votes = [c["tool_prediction"] for c in top]
            ones = votes.count("1"); zeros = votes.count("0")
            if ones == len(top):
                preds[idx] = "1"
                continue
            elif zeros == len(top):
                preds[idx] = "0"
                continue

        selected = sel.get(idx)
        if selected and selected != "abstain":
            pred = get_tool_pred(selected, idx)
            if pred == "NA":
                pred = fallback.get(idx, "NA")
        else:
            pred = fallback.get(idx, "NA")
        preds[idx] = pred

    return preds


# ── Hierarchy config enumeration ──────────────────────────────────────────────

def hierarchy_configs():
    """Yields (name, sel_path, proc_path)."""
    proc  = CAND_PREP_DIR / "processed" / "test.jsonl"
    L     = CFG["selection"]["L"]
    top_k = CFG["selection"]["top_k"]
    base  = SEL_DIR / f"top_t_{L}" / f"k{top_k}"
    yield (
        "ARMOR",
        base / "test" / "resolved.jsonl",
        proc,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    labels    = load_test_labels()
    all_preds = {t: load_raw_preds(t) for t in ALL_TOOLS}
    tools     = ALL_TOOLS

    col = (f"{'name':<32} {'overall':>8}"
           f" {'pos_acc':>8} {'neg_acc':>8}"
           f" {'f1_pos':>7} {'f1_neg':>7}"
           f" {'mcc':>7}")
    sep = "-" * len(col)

    def row(name, m):
        return (f"{name:<32} {m['overall']:>8.2f}"
                f" {m['pos_acc']:>8.2f} {m['neg_acc']:>8.2f}"
                f" {m['f1_pos']:>7.2f} {m['f1_neg']:>7.2f}"
                f" {m['mcc']:>7.4f}")

    print(col)
    print(sep)

    for tool in tools:
        print(row(tool, eval_single_tool(tool, labels)))

    print(sep)
    for name, sel_path, proc_path in hierarchy_configs():
        out_path = BASE / f"predictions_{name}.jsonl"
        if out_path.exists():
            preds = {int(r["idx"]): r["pred"] for r in load_jsonl(out_path)}
        else:
            preds = make_predictions(sel_path, proc_path)
            save_jsonl(
                [{"idx": idx, "pred": pred, "label": labels.get(idx, "")}
                 for idx, pred in sorted(preds.items())],
                out_path,
            )
            print(f"  Saved predictions → {out_path}")
        print(row(name, compute_metrics_from_preds(preds, labels)))


if __name__ == "__main__":
    main()
