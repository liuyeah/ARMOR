#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

_BASE_DIR  = Path(__file__).parent
_CODES_DIR = _BASE_DIR.parent.parent.parent

import yaml as _yaml
with open(_CODES_DIR / "config.yaml") as _f:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_yaml.safe_load(_f).get("cuda_visible_devices", "0"))

import argparse, json, time
from tqdm import tqdm
from typing import Any, Dict, Optional
sys.path.insert(0, str(_CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST, load_pipe, call_llm_single as _call_llm_single, validate_output_single

PATTERN_EXTRACTION_DIR = _BASE_DIR / "outputs_json_format"   # written by rule_verify.py
EXAMPLE_DATA_DIR       = _CODES_DIR / "example_data"


def _norm_pred(v: Any) -> Optional[int]:
    if v is None: return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("na","n/a","none",""): return None
        if s.isdigit():
            iv=int(s); return iv if iv in (0,1) else None
        return None
    if isinstance(v,(int,float)):
        iv=int(v); return iv if iv in (0,1) else None
    return None

def load_idx_map_slice(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    idx_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, ln in enumerate(f):
            ln = ln.strip()
            if not ln: continue
            obj = json.loads(ln)
            idx_map[int(obj["idx"])] = obj
    return idx_map

def validate_one_pattern_with_llm_per_example(
    pat: Dict[str, Any],
    idx_map: Dict[int, Dict[str, Any]],
    call_llm_single_fn,
    validate_output_single_fn,
    retries: int = 3,
    sleep: float = 0.2,
) -> Dict[str, Any]:
    rule_name = pat.get("name", "")
    rule_expl = pat.get("explanation", "")

    examples = []
    for idx in pat.get("examples_idx", []):
        row = idx_map.get(int(idx))
        if row is None:
            continue
        examples.append({
            "idx": int(idx),
            "reactants": row.get("reactants", ""),
            "product": row.get("product", ""),
        })

    results = []
    failed_idx = []

    for ex in examples:
        ex_idx = ex["idx"]
        last_err = None
        out = None
        for attempt in range(1, retries + 1):
            try:
                out = call_llm_single_fn(rule_name, rule_expl, ex)
                validate_output_single_fn(out, rule_name, ex_idx)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, 0.5 * attempt))
        if out is None or last_err:
            failed_idx.append(ex_idx)
            results.append({
                "idx": ex_idx,
                "belongs_to_rule": False,
                "confidence": "low",
                "reason": f"Model/parse failure: {type(last_err).__name__}" if last_err else "Model/parse failure",
            })
        else:
            results.append({
                "idx": int(out["idx"]),
                "belongs_to_rule": bool(out["belongs_to_rule"]),
                "confidence": out["confidence"],
                "reason": out["reason"],
            })

        time.sleep(sleep)

    mismatch = [r["idx"] for r in results if not r["belongs_to_rule"]]
    total = len(results)
    ok = total - len(mismatch)
    rate = ok / total if total else 0.0

    return {
        "status": "ok" if not failed_idx else "partial",
        "agreement_rate": rate,
        "agreement_rate_str": f"{rate*100:.2f}%",
        "mismatch_idx": mismatch,
        "failed_idx": failed_idx,
        "results": results,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool_name",  type=str, default=None, help="Tool name (omit to run all tools)")
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--retries",    type=int, default=3)
    ap.add_argument("--sleep",      type=float, default=0.2)
    args = ap.parse_args()

    llm    = CFG["llm"]
    tau_1  = CFG["rule_filtering"]["tau_1"]
    N_list = CFG["subset"]["N_list"]
    M      = CFG["subset"]["M"]
    tools  = [args.tool_name] if args.tool_name else TOOL_LIST

    pipe = load_pipe(llm)

    def call_llm_single_fn(rule_name, rule_expl, ex):
        return _call_llm_single(rule_name, rule_expl, ex,
                                pipe=pipe, max_tokens=args.max_tokens)

    for tool_name in tools:
        for N in N_list:
            for sample_num in tqdm(range(M), desc=f"TOOL: {tool_name}, N={N}, LLM: {llm}"):
                in_json = str(PATTERN_EXTRACTION_DIR / f"N={N}" / tool_name
                              / f"sample_{sample_num}_{llm}_json_format_pattern.json")
                dataset_jsonl = str(EXAMPLE_DATA_DIR / "dev"
                                    / f"{tool_name}.jsonl")
                output_dir = _BASE_DIR / f"outputs_json_format/N={N}/{tool_name}/valid_by_{llm}"
                output_dir.mkdir(parents=True, exist_ok=True)
                out_json = str(output_dir / f"sample_{sample_num}_{llm}_json_format_pattern_validated_by_{llm}.json")

                doc      = json.load(open(in_json, "r", encoding="utf-8"))
                tool_doc = doc["tools"][0]
                idx_map  = load_idx_map_slice(dataset_jsonl)

                for section_key in ("often_correct",):
                    for pat in tool_doc.get(section_key, []):
                        if pat.get("example_matching_rate", 0.0) < tau_1:
                            continue

                        pat["llm_validation"] = validate_one_pattern_with_llm_per_example(
                            pat, idx_map,
                            call_llm_single_fn=call_llm_single_fn,
                            validate_output_single_fn=validate_output_single,
                            retries=args.retries,
                            sleep=args.sleep,
                        )
                        time.sleep(args.sleep)

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)

                print(f"[OK] wrote validated JSON to {out_json}")


if __name__ == "__main__":
    main()