#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_rules.py: Match pruned rules against dev and test disagreement samples.

Pipeline order:
  dev  — reads all often_correct rules from prune_outputs (no filter).
  test — only runs rules selected into rule_set.json under
         pattern_consolidation/. Run generate_ruleset.py first.

Usage:
    python match_rules.py                          # all tools, both phases
    python match_rules.py --phase dev              # all tools, dev only
    python match_rules.py --tool_name BERT         # one tool, both phases
"""

import os
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).parent
CODES_DIR = BASE_DIR.parent.parent.parent

import yaml as _yaml
with open(CODES_DIR / "config.yaml") as _f:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_yaml.safe_load(_f).get("cuda_visible_devices", "0"))

import json
import argparse
from tqdm import tqdm
sys.path.insert(0, str(CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST, load_pipe, call_llm_single as _call_llm_single, validate_output_single

PRUNE_DIR = (CODES_DIR / "module2_utility_aware_tool_prioritization"
             / "reaction_specific_tool_utility_assessment"
             / "pattern_consolidation" / "prune_outputs")
RULESET_GEN_DIR = (BASE_DIR.parent.parent
                   / "reaction_specific_tool_utility_assessment"
                   / "pattern_consolidation")
DEV_DISAGREEMENT_PATH  = CODES_DIR / "module1_tool_hierarchy_construction" / "splits"      / "disagreement.jsonl"
TEST_DISAGREEMENT_PATH = CODES_DIR / "module1_tool_hierarchy_construction" / "test_splits" / "disagreement.jsonl"


def load_disagreement_examples(path):
    """Load all disagreement samples (pos + neg) with their combined idx."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append({
                "idx":       int(item["idx"]),
                "reactants": item["reactants"],
                "product":   item["product"],
            })
    return examples


def rule_matching(examples, rule, call_llm_single_fn, validate_output_single_fn,
                  output_filepath, tqdm_desc, retries=3):
    rule_name = rule.get("name", "")
    rule_expl = rule.get("explanation", "")

    results = []
    for ex in tqdm(examples, total=len(examples), desc=tqdm_desc):
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
        if out is None or last_err:
            results.append({
                "idx":             ex_idx,
                "belongs_to_rule": False,
                "confidence":      "low",
                "reason":          f"Model/parse failure: {type(last_err).__name__}" if last_err else "Model/parse failure",
            })
        else:
            results.append({
                "idx":             int(out["idx"]),
                "belongs_to_rule": bool(out["belongs_to_rule"]),
                "confidence":      out["confidence"],
                "reason":          out["reason"],
            })

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_rules_for_phase(tool_name, valid_llm, phase):
    """Return the list of rules to match for the given phase.

    dev  — all often_correct rules from prune_outputs.
    test — only rules selected into rule_set.json by generate_ruleset.py.
    """
    rules_path = PRUNE_DIR / f"valid_by_{valid_llm}" / f"{tool_name}.json"
    doc = json.loads(rules_path.read_text(encoding="utf-8"))
    all_rules = doc.get("often_correct", [])

    if phase == "dev":
        return all_rules

    # test phase: filter to rules already selected by generate_ruleset.py
    rs_path = RULESET_GEN_DIR / "rule_set.json"
    if not rs_path.exists():
        raise FileNotFoundError(
            f"rule_set.json not found: {rs_path}\n"
            "Run generate_ruleset.py before matching on test."
        )

    data = json.loads(rs_path.read_text(encoding="utf-8"))
    selected_names = {r["rule_name"] for r in data if r["tool"] == tool_name}

    filtered = [r for r in all_rules if r.get("name") in selected_names]
    skipped = len(all_rules) - len(filtered)
    if skipped:
        print(f"  [test] {tool_name}: {len(all_rules)} → {len(filtered)} rules "
              f"(skipped {skipped} not in rule set)")
    return filtered


def run_matching(pipe, tool_name, valid_llm, phase=None):
    def call_llm_single(rule_name, rule_expl, ex):
        return _call_llm_single(rule_name, rule_expl, ex, pipe=pipe)

    all_examples = {
        "dev":  load_disagreement_examples(DEV_DISAGREEMENT_PATH),
        "test": load_disagreement_examples(TEST_DISAGREEMENT_PATH),
    }

    phases_to_run = [phase] if phase else ["dev", "test"]

    # Load all rules once to use global indices for consistent file naming.
    # aggregate_matching.py iterates with global indices, so file names must match.
    all_rules_path = PRUNE_DIR / f"valid_by_{valid_llm}" / f"{tool_name}.json"
    all_rules = json.loads(all_rules_path.read_text(encoding="utf-8")).get("often_correct", [])

    for phase_key in phases_to_run:
        correct_rules = load_rules_for_phase(tool_name, valid_llm, phase_key)
        selected_names = {r.get("name") for r in correct_rules}
        examples = all_examples[phase_key]
        print(f"  [match] tool={tool_name}, phase={phase_key}, rules={len(correct_rules)}")

        for global_r_idx, rule in enumerate(all_rules):
            if rule.get("name") not in selected_names:
                continue  # dev: all pass; test: only rules in rule_set.json
            rule_name = rule.get("name", "").replace("/", "_")
            out_dir  = BASE_DIR / "outputs" / tool_name / phase_key / "CORRECT_RULES"
            out_path = str(out_dir / f"RULE_{global_r_idx + 1}: {rule_name}.json")

            if Path(out_path).exists():
                continue

            rule_matching(
                examples=examples,
                rule=rule,
                call_llm_single_fn=call_llm_single,
                validate_output_single_fn=validate_output_single,
                output_filepath=out_path,
                tqdm_desc=f"{tool_name} {phase_key} RULE_{global_r_idx + 1}/{len(all_rules)}",
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool_name",  default=None, help="Tool name (omit to run all tools)")
    ap.add_argument("--phase", default=None, choices=["dev", "test"],
                    help="Run a single phase only (default: run both dev and test)")
    args = ap.parse_args()

    llm   = CFG["llm"]
    pipe  = load_pipe(llm)
    tools = [args.tool_name] if args.tool_name else TOOL_LIST
    for tool in tools:
        run_matching(pipe, tool, llm, args.phase)
