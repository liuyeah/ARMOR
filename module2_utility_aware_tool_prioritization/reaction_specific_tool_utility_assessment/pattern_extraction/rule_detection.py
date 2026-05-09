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

import argparse
import json
from tqdm import tqdm
from typing import Any, Dict
sys.path.insert(0, str(_CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST, load_pipe

SYSTEM_PROMPT = """You are a careful chemistry scientist.
Follow the user's instructions exactly.
You must output ONLY valid JSON text.
Do not use markdown, code blocks, or any text outside the JSON object."""


def empty_result_json() -> Dict[str, Any]:
    return {
        "tool_acc": "NA",
        "often_correct_on": [],
    }


def parse_and_validate_json_output(
    text: str,
    output_filepath: str,
) -> Dict[str, Any]:
    """
    Extract, parse, and minimally validate JSON output from the model.
    On any failure, save raw output and return an empty-result JSON.
    """

    json_text = text.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.find("{"): json_text.rfind("}") + 1]

    try:
        obj = json.loads(json_text)
    except Exception:
        return empty_result_json()

    try:
        for k in ["tool_acc", "often_correct_on"]:
            if k not in obj:
                return empty_result_json()

        def _check_patterns(lst):
            if not isinstance(lst, list):
                return False
            for p in lst:
                if not isinstance(p, dict):
                    return False
                for kk in ["name", "explanation", "examples_idx"]:
                    if kk not in p:
                        return False
                if (
                    not isinstance(p["examples_idx"], list)
                    or len(p["examples_idx"]) != 5
                    or not all(isinstance(x, int) for x in p["examples_idx"])
                ):
                    return False
            return True

        if not _check_patterns(obj["often_correct_on"]):
            return empty_result_json()

    except Exception:
        return empty_result_json()

    return obj


def load_jsonl_as_text(path: str) -> str:
    """Load a JSONL file and return its contents as a single string."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return "\n".join(lines)

def build_user_prompt(dataset_text: str) -> str:
    # Keep prompt small but explicit; forbid borrowing info from other rules.
    return f"""
You will be given a single dataset file related to reaction feasibility prediction.

The file is in JSON Lines format, where each line corresponds to one chemical reaction.
Each reaction entry includes:
- idx: an integer identifier for referencing specific reactions;
- reactants: a SMILES string representing the reactant molecules;
- product: a SMILES string representing the reaction product;
- label: a ground-truth binary label indicating whether the reaction is feasible;
- prediction from a single tool, where the prediction may be 0, 1, or NA (missing).

Your task is to analyze predictions from the tool only and provide a concise, pattern-level summary of this tool’s behavior.
Focus on recurring trends rather than exhaustive coverage.

You must complete the task in a single response.
Do NOT ask to split the task across messages.
Do NOT request scope confirmation.

Treat NA, None, or missing predictions as WRONG when computing ACC. Please compute exact ACC and extract representative indices programmatically.

Strict rules (must follow):
• Output valid JSON only. Do NOT use code blocks or markdown. Do NOT include any text outside the JSON object.

IMPORTANT PRACTICAL CONSTRAINTS:
• Report reaction patterns the tool is often CORRECT on
• Patterns should capture recurring reaction behaviors and be formulated as decision-relevant categories, such that a given reaction can be judged as either belonging to or not belonging to the pattern based on observable characteristics.

Output the following structure exactly as a single JSON object:

{{
  "tool_acc": "xx.xx%",
  "often_correct_on": [
    {{
      "name": "Short, human-readable, chemistry-level name.",
      "explanation": "Describe explicit, observable reaction characteristics (e.g., bond changes, functional groups appearing or disappearing, or structural motifs) that enable an LLM to make a YES/NO decision on whether a given reaction belongs to this pattern. Where possible, the explanation should specify both inclusion cues (what must be present) and exclusion cues (what would disqualify a reaction from the pattern).",
      "examples_idx": [0, 0, 0, 0, 0]
    }}
  ]
}}

Hard constraints:
• Reaction pattern names must be short and human-readable.
• Explanations may include chemical concepts, structural cues, or transformation characteristics, as long as it helps the model make a reliable yes/no decision about whether a given reaction belongs to the pattern. Explanations should prioritize decisiveness and discriminability over abstract or high-level chemical generalization.
• Each examples_idx must contain exactly 5 integers that appear as idx values in the dataset.

Analyze ONLY the dataset provided next. After completing the analysis, internally verify whether the generated reaction patterns and explanations can correctly classify the listed example reactions. If mismatches are found, refine the pattern definitions and explanations to resolve the inconsistencies. Perform this verification and refinement silently, and output ONLY the final, consolidated results that conform to the required output structure. Do NOT output intermediate versions, alternative drafts, or self-correction steps.


Dataset input:
{dataset_text}
""".strip()



def detection(input_filepath, pipe, output_filepath, max_tokens=2048):

    dataset_text = load_jsonl_as_text(input_filepath)
    user_prompt = build_user_prompt(dataset_text)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    outputs = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_full_text=False,
    )
    
    text = outputs[0]["generated_text"].strip()
    
    if not text:
        obj = empty_result_json()
    else:
        obj = parse_and_validate_json_output(text, output_filepath)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def main():

    OUTPUT_DIR = Path(__file__).parent / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--tool",       type=str, default=None, help="Tool name (omit to run all tools)")
    parser.add_argument("--N",          type=int, default=None, help="Subset size N (omit to run all N in config)")
    parser.add_argument("--sample_num", type=int, default=None, help="Subset index (omit to run all M samples)")
    args = parser.parse_args()

    llm  = CFG["llm"]
    pipe = load_pipe(llm)

    tools      = [args.tool]       if args.tool       is not None else TOOL_LIST
    n_list     = [args.N]          if args.N           is not None else CFG["subset"]["N_list"]
    sample_ids = [args.sample_num] if args.sample_num  is not None else list(range(CFG["subset"]["M"]))

    for tool in tools:
        for N in n_list:
            for sample_num in tqdm(sample_ids, desc=f"{tool} N={N}"):
                subset_dir      = OUTPUT_DIR / f"N={N}" / tool
                input_filepath  = str(subset_dir / f"sample_{sample_num}.jsonl")
                output_filepath = str(subset_dir / f"sample_{sample_num}_{llm}.jsonl")
                if Path(output_filepath).exists():
                    continue
                detection(input_filepath=input_filepath, pipe=pipe,
                          output_filepath=output_filepath, max_tokens=2048)
    


if __name__ == "__main__":
    main()