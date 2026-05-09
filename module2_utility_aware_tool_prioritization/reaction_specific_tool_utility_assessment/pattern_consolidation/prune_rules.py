#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

_CODES_DIR = Path(__file__).parent.parent.parent.parent

import yaml as _yaml
with open(_CODES_DIR / "config.yaml") as _f:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_yaml.safe_load(_f).get("cuda_visible_devices", "0"))

import argparse
import json
from copy import deepcopy
from typing import Any, Dict, List
sys.path.insert(0, str(_CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST, load_pipe

SYSTEM_PROMPT = """You are a careful chemistry scientist.
Follow the user's instructions exactly.
You must output ONLY valid JSON text.
Do not use markdown, code blocks, or any text outside the JSON object."""

# -----------------------
# Basic IO
# -----------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# LLM output parsing
# -----------------------
def extract_first_json_obj(text: str) -> str:
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]
    return s

def parse_dedup_decision(text: str, n_candidates: int) -> Dict[str, Any]:
    """
    Expected JSON:
    {
      "keep_index": <int>,
      "reason": "<string>"
    }
    Returns a dict with "keep_index" and "reason".
    """
    json_text = extract_first_json_obj(text)
    obj = json.loads(json_text)

    if "keep_index" not in obj or not isinstance(obj["keep_index"], int):
        raise ValueError("Missing/invalid keep_index")

    if obj["keep_index"] < 0 or obj["keep_index"] >= n_candidates:
        raise ValueError(f"keep_index out of range: {obj['keep_index']} / {n_candidates}")

    if "reason" not in obj or not isinstance(obj["reason"], str):
        obj["reason"] = ""

    return obj

# -----------------------
# Prompting
# -----------------------
def compact_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": rule.get("name", ""),
        "explanation": rule.get("explanation", ""),
    }

def build_user_prompt(rule_name: str, rules: List[Dict[str, Any]]) -> str:
    candidates = [{"index": i, **compact_rule(r)} for i, r in enumerate(rules)]
    return f"""
You are given {len(rules)} reaction-pattern rules that all share the same name: "{rule_name}".

These rules describe the same concept but were written separately with slightly different wording.
Your task is to keep exactly ONE best rule and remove all others.

Criteria for the rule to keep:
- Most precise and complete chemistry description
- Clearest language

You MUST remove all rules except the best one. Do NOT keep more than one.

Return ONLY valid JSON (no extra text) in this exact schema:
{{
  "keep_index": <integer>,
  "reason": "<1-2 sentences>"
}}

Here are the candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
""".strip()

# -----------------------
# Core dedup
# -----------------------
def llm_pick_best(pipe, rule_name: str, rules: List[Dict[str, Any]], max_tokens: int, retries: int = 3) -> Dict[str, Any]:
    user_prompt = build_user_prompt(rule_name, rules)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    last_err = None
    for _ in range(retries + 1):
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"].strip()
        try:
            return parse_dedup_decision(text, n_candidates=len(rules))
        except Exception as e:
            last_err = e

    # fallback: keep the first one
    print(f"  [WARN] Parse failed after retries ({last_err}), keeping first rule in group.")
    return {"keep_index": 0, "reason": f"Fallback: parse error {last_err}"}


def single_pass_dedup(pipe, category: str, rules: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    print(f"\n[START] Deduplicating {category}: {len(rules)} rules")

    tagged = []
    for i, r in enumerate(rules):
        rr = deepcopy(r)
        if "orig_idx" not in rr:
            rr["orig_idx"] = i
        tagged.append(rr)

    groups: Dict[str, List] = {}
    for r in tagged:
        key = r.get("name", "").strip().lower()
        groups.setdefault(key, []).append(r)

    kept = []
    removed = []

    for name_key, group in groups.items():
        display_name = group[0].get("name", name_key)
        if len(group) == 1:
            kept.append(group[0])
            continue

        print(f"  [GROUP] '{display_name}': {len(group)} rules → picking best via LLM")
        decision = llm_pick_best(pipe, display_name, group, max_tokens=max_tokens)
        keep_idx = decision["keep_index"]

        for i, r in enumerate(group):
            if i == keep_idx:
                kept.append(r)
            else:
                item = deepcopy(r)
                item["_remove_decision"] = decision
                removed.append(item)

    print(f"[DONE] {category}: {len(rules)} → {len(kept)} (removed {len(removed)})")

    return {
        "kept": kept,
        "removed": removed,
        "original_count": len(rules),
        "final_count": len(kept),
    }

# -----------------------
# Main
# -----------------------
def main():
    BASE_DIR   = Path(__file__).parent
    INPUT_DIR  = BASE_DIR / "aggregation_outputs"
    OUTPUT_DIR = BASE_DIR / "prune_outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens for each dedup decision")
    args = parser.parse_args()

    llm  = CFG["llm"]
    pipe = load_pipe(llm)

    for tool_name in TOOL_LIST:
        input_path = str(INPUT_DIR / f"valid_by_{llm}" / f"{tool_name}.json")
        output_dir = OUTPUT_DIR / f"valid_by_{llm}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{tool_name}.json")

        data = load_json(input_path)
        out = deepcopy(data)

        print(f"\n========== {tool_name} ==========")

        for category in ["often_correct"]:
            if category not in data or not isinstance(data[category], list):
                continue

            rules = data[category]

            if len(rules) <= 1:
                tagged = []
                for i, r in enumerate(rules):
                    rr = deepcopy(r)
                    if "orig_idx" not in rr:
                        rr["orig_idx"] = i
                    tagged.append(rr)
                out[category] = tagged
                out[f"{category}_original_count"] = len(rules)
                out[f"{category}_final_count"] = len(rules)
                continue

            result = single_pass_dedup(
                pipe=pipe,
                category=category,
                rules=rules,
                max_tokens=args.max_tokens,
            )

            out[category] = result["kept"]
            out[f"removed_{category}"] = result["removed"]
            out[f"{category}_original_count"] = result["original_count"]
            out[f"{category}_final_count"] = result["final_count"]

        save_json(out, output_path)
        print(f"[OK] Wrote deduped file to: {output_path}")


if __name__ == "__main__":
    main()
