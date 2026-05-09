"""
define_hierarchy.py: Compute per-tool accuracy on dev set,
define a two-tier hierarchy by selecting top-K tools as tier 1, and split
samples into unanimous / disagreement sets based on tier 1 predictions.

K is computed from config.yaml: k = round(n_tools * hierarchy.p)

Usage:
    python define_hierarchy.py
    python define_hierarchy.py --output hierarchy.json --splits_dir splits/
"""

import sys
import argparse
import json
from pathlib import Path

BASE_DIR         = Path(__file__).parent
CODES_DIR        = BASE_DIR.parent
EXAMPLE_DATA_DIR = CODES_DIR / "example_data"

sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST

TOOLS = TOOL_LIST


def load_preds(split, tool, label=None):
    """Load predictions for a tool from example_data/dev or example_data/test.
    If label is given (1 or 0), filter by that label only.
    """
    path = EXAMPLE_DATA_DIR / split / f"{tool}.jsonl"
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


def compute_acc(tool):
    pos = load_preds("dev", tool, label=1)
    neg = load_preds("dev", tool, label=0)
    pc = sum(1 for r in pos if str(r["tool_prediction"]) == "1")
    nc = sum(1 for r in neg if str(r["tool_prediction"]) == "0")
    return {
        "overall": (pc + nc) / (len(pos) + len(neg)),
        "pos_acc": pc / len(pos),
        "neg_acc": nc / len(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "hierarchy.json"))
    parser.add_argument("--splits_dir", type=str, default=str(BASE_DIR / "splits"), help="Directory to save unanimous/disagreement splits")
    args = parser.parse_args()

    p = CFG["hierarchy"]["p"]
    k = max(1, round(len(TOOLS) * p))
    print(f"hierarchy.p={p} → tier-1 size k={k} / {len(TOOLS)} tools\n")
    print("Computing accuracy on example_data/dev...\n")

    results = []
    for tool in TOOLS:
        acc = compute_acc(tool)
        results.append({"tool": tool, **acc})

    results.sort(key=lambda x: x["overall"], reverse=True)

    print(f"  {'rank':>4}  {'tool':<25}  {'overall':>8}  {'pos_acc':>8}  {'neg_acc':>8}  tier")
    print("  " + "-" * 70)
    tier1, tier2 = [], []
    for rank, r in enumerate(results, 1):
        tier = 1 if rank <= k else 2
        marker = "  <-- tier 1" if tier == 1 else ""
        print(f"  {rank:>4}  {r['tool']:<25}  {r['overall']*100:>8.2f}  {r['pos_acc']*100:>8.2f}  {r['neg_acc']*100:>8.2f}  {tier}{marker}")
        if tier == 1:
            tier1.append(r["tool"])
        else:
            tier2.append(r["tool"])

    hierarchy = {
        "split": f"example_data/dev ({results[0]['n_pos']} pos + {results[0]['n_neg']} neg per tool)",
        "k": k,
        "tier1": tier1,
        "tier2": tier2,
        "tool_stats": {r["tool"]: {
            "overall": round(r["overall"], 4),
            "pos_acc": round(r["pos_acc"], 4),
            "neg_acc": round(r["neg_acc"], 4),
        } for r in results},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    print(f"\nTier 1 ({k} tools): {tier1}")
    print(f"Tier 2 ({len(tier2)} tools): {tier2}")
    print(f"\nSaved to: {output_path}")

    # ── Split samples into unanimous / disagreement based on tier 1 ──
    print(f"\nSplitting samples by tier 1 consensus...")

    tier1_preds_pos = {t: {r["idx"]: str(r["tool_prediction"]) for r in load_preds("dev", t, label=1)} for t in tier1}
    tier1_preds_neg = {t: {r["idx"]: str(r["tool_prediction"]) for r in load_preds("dev", t, label=0)} for t in tier1}

    ref_pos = load_preds("dev", tier1[0], label=1)
    ref_neg = load_preds("dev", tier1[0], label=0)

    unanimous, disagreement = [], []
    for sample in ref_pos + ref_neg:
        idx = sample["idx"]
        is_pos = sample["label"] == 1 or sample["label"] == "1"
        preds_map = tier1_preds_pos if is_pos else tier1_preds_neg
        idx_lookup = idx      # use original example_data idx for both pos and neg lookups
        votes = {preds_map[t].get(idx_lookup) for t in tier1}
        entry = {field: sample[field] for field in sample}
        entry["tier1_predictions"] = {t: preds_map[t].get(idx_lookup) for t in tier1}
        entry["tier1_consensus"] = list(votes)[0] if len(votes) == 1 else None
        if len(votes) == 1:
            unanimous.append(entry)
        else:
            disagreement.append(entry)

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(splits_dir / "unanimous.jsonl", unanimous)
    write_jsonl(splits_dir / "disagreement.jsonl", disagreement)

    print(f"  unanimous:    {len(unanimous)} samples")
    print(f"  disagreement: {len(disagreement)} samples")
    print(f"  Saved to: {splits_dir}/")


if __name__ == "__main__":
    main()
