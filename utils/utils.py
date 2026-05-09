#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for the ARMOR pipeline.

Includes:
  - CFG, TOOL_LIST: global config and tool registry
  - load_pipe(): load a HuggingFace text-generation pipeline
  - call_llm_single(), validate_output_single(): per-example rule evaluation
  - prepare_candidates(): top-1-per-tool + top-t filtering
  - build_guided_prompt(), llm_guided_demo(): guided memory construction
  - build_prompt(), llm_choose_tool(): LLM-based tool selection
"""

import argparse
import json
import yaml
import torch
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────

CODES_DIR        = Path(__file__).parent.parent
EXAMPLE_DATA_DIR = CODES_DIR / "example_data"
_HIERARCHY_PATH  = CODES_DIR / "module1_tool_hierarchy_construction" / "hierarchy.json"
_CONFIG_PATH     = CODES_DIR / "config.yaml"

# ── Load global hyperparameters ───────────────────────────────────────────────

try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
        CFG = yaml.safe_load(_f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"config.yaml not found at {_CONFIG_PATH}. "
        "Please create it before running the pipeline."
    )

# ── Tool list from config ──────────────────────────────────────────────────────

TOOL_LIST = CFG["tool_list"]

# ── Model paths — set model_paths in config.yaml ──────────────────────────────

HF_MODELS = CFG.get("model_paths", {})

SYSTEM_PROMPT = """You are a careful chemistry scientist.
Follow the user's instructions exactly.
You must output ONLY valid JSON text.
Do not use markdown, code blocks, or any text outside the JSON object."""

# Tier definitions — loaded from hierarchy.json at import time
_TIER1_TOOLS: Set[str] = None

def get_tier1_tools() -> Set[str]:
    global _TIER1_TOOLS
    if _TIER1_TOOLS is None:
        if not _HIERARCHY_PATH.exists():
            raise FileNotFoundError(
                f"hierarchy.json not found: {_HIERARCHY_PATH}\n"
                "Run first: python module1_tool_hierarchy_construction/define_hierarchy.py"
            )
        with _HIERARCHY_PATH.open("r", encoding="utf-8") as f:
            _TIER1_TOOLS = set(json.load(f)["tier1"])
    return _TIER1_TOOLS


def parse_top_t(v: str):
    if v.lower() in ("none", "null", "all"):
        return None
    val = int(v)
    if val <= 0:
        raise argparse.ArgumentTypeError("top_t must be a positive integer or 'none'")
    return val


def top_t_str(top_t) -> str:
    return str(top_t) if top_t is not None else "none"


# ── IO ────────────────────────────────────────────────────────────────────────

def load_jsonl(path) -> List[Any]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON parse failed at line {i}: {e}")
    return rows


def save_jsonl(rows: List[Any], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── Rule evaluation helpers ───────────────────────────────────────────────────

RULE_EVAL_SYSTEM_PROMPT = (
    "You are a careful chemistry reaction-rule evaluator.\n"
    "Follow the user's instructions exactly.\n"
    "You must output ONLY valid JSON (no markdown, no extra text)."
)


def _build_rule_eval_prompt_single(rule_name: str, rule_explanation: str, ex: Dict) -> str:
    return f"""You will be given ONE reaction rule with name, explanation, and an example.
Your task: for THIS rule ONLY, judge whether THIS example belongs to the rule.

Instructions:
- Judge this example independently.
- Do NOT assume the example is correct.
- Base judgment only on:
  (1) rule name,
  (2) rule explanation,
  (3) example reactants/product SMILES.
- When making the judgment, treat the rule explanation as the primary and authoritative criterion for deciding whether an example belongs to the rule; the rule name serves only as a high-level label.

Output exactly ONE JSON object with format:
{{
  "name": <rule name>,
  "idx": <example_idx>,
  "belongs_to_rule": true/false,
  "confidence": "high"|"medium"|"low",
  "reason": "brief explanation (1–2 sentences)"
}}

Output requirements:
- Output MUST be valid JSON.
- Output ONE JSON object only.
- Do NOT include any text outside the JSON object.
- Do NOT repeat the input examples verbatim.

Rule input:
{{
  "name": {json.dumps(rule_name, ensure_ascii=False)},
  "explanation": {json.dumps(rule_explanation, ensure_ascii=False)}
}}

Example:
{json.dumps(ex, ensure_ascii=False)}""".strip()


def _extract_json_str(text: str) -> str:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start:end + 1]


def call_llm_single(
    rule_name: str,
    rule_explanation: str,
    ex: Dict,
    pipe=None,
    max_tokens: int = 600,
) -> Dict:
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "system", "content": RULE_EVAL_SYSTEM_PROMPT},
         {"role": "user",   "content": _build_rule_eval_prompt_single(rule_name, rule_explanation, ex)}],
        tokenize=False, add_generation_prompt=True,
    )
    outputs = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)
    text = outputs[0]["generated_text"]
    if not text.strip():
        raise ValueError("Empty model output.")
    return json.loads(_extract_json_str(text))


def validate_output_single(obj: Dict, rule_name: str, idx: int) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")
    if obj.get("name") != rule_name:
        raise ValueError(f'name mismatch expected="{rule_name}" got="{obj.get("name")}"')
    if int(obj.get("idx")) != int(idx):
        raise ValueError(f'idx mismatch expected={idx} got={obj.get("idx")}')
    if obj.get("belongs_to_rule") not in (True, False):
        raise ValueError("belongs_to_rule must be boolean.")
    if obj.get("confidence") not in ("high", "medium", "low"):
        raise ValueError("confidence must be high|medium|low.")
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        raise ValueError("reason must be non-empty string.")


# ── Model ─────────────────────────────────────────────────────────────────────

def load_pipe(model_key: str):
    from transformers import pipeline
    if model_key not in HF_MODELS:
        raise ValueError(f"Unknown model_key: {model_key}. Available: {list(HF_MODELS)}")
    path = HF_MODELS[model_key]
    if not path:
        raise ValueError(
            f"Model path for '{model_key}' is not set. "
            f"Please set model_paths.{model_key} in config.yaml."
        )
    return pipeline(
        "text-generation",
        model=HF_MODELS[model_key],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


# ── Prompt building ───────────────────────────────────────────────────────────

def _format_cand_entry(c: Dict, with_confidence: bool) -> Dict:
    entry = {
        "tool":             c.get("tool", ""),
        "rule_name":        c.get("rule_name", ""),
        "rule_explanation": c.get("rule_explanation", ""),
        "tool_prediction":  c.get("tool_prediction", ""),
    }
    if with_confidence and "rule_confidence" in c:
        entry["rule_confidence"] = round(c["rule_confidence"], 4)
    return entry


def _format_tier_block(label: str, description: str, cands: List[Dict]) -> str:
    if not cands:
        return f"[{label}] {description}\n  (no candidates from this tier)"
    lines = [f"[{label}] {description}"]
    lines.append(json.dumps(cands, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def _format_demo_compact(demo: Dict, i: int) -> str:
    """Compact reference: reaction + evidence + chosen tool."""
    rxn    = demo.get("reaction", {})
    d      = demo.get("demo", {})
    ev     = d.get("evidence", [])
    ev_str = "; ".join(ev) if ev else ""
    reason = d.get("final_reason", "")
    tool   = d.get("tool", demo.get("meta", {}).get("gold_tool", ""))
    lines  = [f"[Reference {i + 1}]",
              f"Reaction: {rxn.get('reactants', '')} >> {rxn.get('product', '')}"]
    if ev_str:
        lines.append(f"Evidence: {ev_str}")
    lines.append(f"Chosen: {tool} — {reason}")
    return "\n".join(lines)


def _format_demo_full(demo: Dict, i: int, show_tiers: bool = True) -> str:
    """Full reference: reaction + candidates + evidence + elimination + chosen tool."""
    rxn    = demo.get("reaction", {})
    cands  = demo.get("candidates", [])
    d      = demo.get("demo", {})
    ev     = d.get("evidence", [])
    elim   = d.get("elimination", [])
    reason = d.get("final_reason", "")
    tool   = d.get("tool", demo.get("meta", {}).get("gold_tool", ""))
    lines  = [f"[Reference {i + 1}]",
              f"Reaction: {rxn.get('reactants', '')} >> {rxn.get('product', '')}"]
    if cands:
        def _slim(c):
            return {"tool": c.get("tool"), "rule_name": c.get("rule_name"),
                    "tool_prediction": c.get("tool_prediction")}
        if show_tiers:
            t1 = [_slim(c) for c in cands if c.get("tool") in get_tier1_tools()]
            t2 = [_slim(c) for c in cands if c.get("tool") not in get_tier1_tools()]
            lines.append("[Tier-1] Candidates: " + (json.dumps(t1, ensure_ascii=False) if t1 else "(no candidates from this tier)"))
            lines.append("[Tier-2] Candidates: " + (json.dumps(t2, ensure_ascii=False) if t2 else "(no candidates from this tier)"))
        else:
            lines.append("Candidates: " + json.dumps(
                [_slim(c) for c in cands], ensure_ascii=False))
    if ev:
        lines.append("Evidence:")
        for e in ev:
            lines.append(f"  - {e}")
    if elim:
        lines.append("Elimination:")
        for e in elim:
            lines.append(f"  - {e.get('tool', '')}: {e.get('why_not', '')}")
    lines.append(f"Chosen: {tool} — {reason}")
    return "\n".join(lines)


def _format_demos_block(demos: List[Dict], demo_format: str, show_tiers: bool = True) -> str:
    """Format demos into a reference block. Returns empty string if no demos."""
    if not demos:
        return ""
    if demo_format == "full":
        parts = [_format_demo_full(d, i, show_tiers) for i, d in enumerate(demos)]
    else:
        parts = [_format_demo_compact(d, i) for i, d in enumerate(demos)]
    body   = "\n\n".join(parts)
    note   = (
        "Note: The reference cases are from different reactions. "
        "Use them only to guide your reasoning process — do not copy their tool choices. "
        "Your selection must be based solely on the reaction and candidates "
        "provided at the beginning of this prompt."
    )
    return (
        "The following are reference cases from different reactions, "
        "showing how to reason about candidate selection:\n\n"
        + body + "\n\n" + note
    )


def build_prompt(
    reactants: str,
    product: str,
    cands: List[Dict],
    demos: List[Dict],
    with_confidence: bool,
    tier_bias: str = "none",      # "none" | "tier1" | "tier2"
    demo_format: str = "compact", # "compact" | "full"
    show_tiers: bool = True,      # True = Tier-1/Tier-2 blocks; False = flat list
) -> str:
    allowed_tools = sorted({c["tool"] for c in cands if c.get("tool")})
    allowed_str   = ", ".join([f'"{t}"' for t in allowed_tools] + ['"abstain"'])

    conf_hint = (
        "- Higher rule_confidence means the tool is more reliable on samples matching this rule.\n"
        if with_confidence else ""
    )

    if tier_bias == "tier1":
        tiebreak = "- When the rule match quality is similar across tiers, prefer Tier-1 tools.\n"
    elif tier_bias == "tier2":
        tiebreak = "- When the rule match quality is similar across tiers, prefer Tier-2 tools.\n"
    else:
        tiebreak = ""

    demos_block   = _format_demos_block(demos, demo_format, show_tiers)
    demos_section = f"\n{demos_block}\n" if demos_block else ""

    if show_tiers:
        tier1_cands = [_format_cand_entry(c, with_confidence) for c in cands if c.get("tool") in get_tier1_tools()]
        tier2_cands = [_format_cand_entry(c, with_confidence) for c in cands if c.get("tool") not in get_tier1_tools()]
        tier1_block = _format_tier_block(
            "Tier-1", "Generally strong tools with broad coverage across reaction types.", tier1_cands)
        tier2_block = _format_tier_block(
            "Tier-2", "Specialized tools that may excel on specific reaction types, though narrower in overall coverage.", tier2_cands)
        cands_section = (
            f"Candidate tools are organized into two hierarchy levels based on their characteristics:\n\n"
            f"{tier1_block}\n\n{tier2_block}"
        )
    else:
        flat_cands    = [_format_cand_entry(c, with_confidence) for c in cands]
        cands_section = (
            f"Candidate tools (matched rule per tool):\n"
            f"{json.dumps(flat_cands, ensure_ascii=False, indent=2)}"
        )

    return f"""You are given a chemical reaction:
- reactants (SMILES): {reactants}
- product (SMILES): {product}

{cands_section}

Selection principles:
- Choose the tool whose matched rule best explains the actual chemical transformation.
- Prefer more specific and chemically plausible rule explanations over vague/generic ones.
{conf_hint}{tiebreak}- If multiple candidates match well, consider internal consistency across rules.
- Output "abstain" only if no candidate provides a convincing match.
{demos_section}
Return ONLY valid JSON in this exact schema:
{{
  "tool": <one of [{allowed_str}]>,
  "reason": "<1-3 sentences>"
}}""".strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def _extract_first_json(text: str) -> str:
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.find("{") : s.rfind("}") + 1]
    return s


def _parse_tool_choice(text: str, allowed_tools: List[str]) -> Dict[str, str]:
    obj  = json.loads(_extract_first_json(text))
    tool = str(obj.get("tool", "")).strip()
    if tool != "abstain" and tool not in allowed_tools:
        raise ValueError(f"tool not in allowed_tools: {tool!r}")
    return {"tool": tool, "reason": obj.get("reason", "")}


def llm_choose_tool(
    pipe,
    reactants: str,
    product: str,
    raw_cands: List[Dict],
    demos: List[Dict],
    with_confidence: bool,
    tier_bias: str = "none",
    demo_format: str = "compact",
    show_tiers: bool = True,
    max_tokens: int = 256,
    retries: int = 3,
) -> Dict[str, Any]:
    allowed_tools = sorted({c["tool"] for c in raw_cands if c.get("tool")})

    if not allowed_tools:
        return {"tool": "abstain", "reason": "No valid candidates.", "_raw": ""}

    user_prompt = build_prompt(
        reactants, product, raw_cands, demos, with_confidence, tier_bias, demo_format, show_tiers
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    last_err = None
    for _ in range(retries + 1):
        outputs = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)
        text = outputs[0]["generated_text"].strip()
        try:
            parsed = _parse_tool_choice(text, allowed_tools)
            return {"tool": parsed["tool"], "reason": parsed["reason"], "_raw": text}
        except Exception as e:
            last_err = e

    # Fallback: most frequent tool predicting "1"
    ones = [c["tool"] for c in raw_cands if str(c.get("tool_prediction")) == "1" and c.get("tool")]
    if ones:
        freq: Dict[str, int] = {}
        for t in ones:
            freq[t] = freq.get(t, 0) + 1
        best = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return {"tool": best, "reason": f"Fallback (parse error): {last_err}", "_raw": ""}

    return {"tool": "abstain", "reason": f"Fallback (parse error): {last_err}", "_raw": ""}


# ── Candidate preparation ─────────────────────────────────────────────────────

def prepare_candidates(candidates: List[Dict], top_t: int = None) -> List[Dict]:
    """
    From a sample's matched candidates (multiple rules per tool possible):
      1. top-1-per-tool: keep the highest rule_confidence rule per tool
      2. top-t: sort by that confidence, keep top_t tools (None = keep all)
    Returns a flat list of one entry per tool, sorted by confidence desc.
    """
    tool_best: Dict[str, Tuple[float, Dict]] = {}
    for c in candidates:
        tool = c.get("tool", "")
        if not tool:
            continue
        conf = float(c.get("rule_confidence", 0.0))
        if tool not in tool_best or conf > tool_best[tool][0]:
            tool_best[tool] = (conf, c)

    result = [c for _, c in sorted(tool_best.values(), key=lambda x: -x[0])]
    if top_t is not None:
        result = result[:top_t]
    return result


# ── Guided memory prompt ──────────────────────────────────────────────────────

def build_guided_prompt(
    reactants: str,
    product: str,
    cands: List[Dict],
    gold_tool: str,
    neg_tools: List[str],
    show_tiers: bool = True,
) -> str:
    """
    Prompt for guided memory construction: reveals gold_tool and asks LLM to
    generate evidence/elimination/final_reason explaining the choice.
    show_tiers=True: same Tier-1/Tier-2 structure as build_prompt.
    show_tiers=False: flat candidate list.
    """
    neg_str = ", ".join(f'"{t}"' for t in neg_tools)

    if show_tiers:
        tier1_cands = [_format_cand_entry(c, True) for c in cands if c.get("tool") in get_tier1_tools()]
        tier2_cands = [_format_cand_entry(c, True) for c in cands if c.get("tool") not in get_tier1_tools()]
        tier1_block = _format_tier_block(
            "Tier-1", "Generally strong tools with broad coverage across reaction types.", tier1_cands)
        tier2_block = _format_tier_block(
            "Tier-2", "Specialized tools that may excel on specific reaction types.", tier2_cands)
        cands_section = (
            f"Candidate tools are organized into two hierarchy levels based on their characteristics:\n\n"
            f"{tier1_block}\n\n{tier2_block}"
        )
    else:
        flat_cands    = [_format_cand_entry(c, True) for c in cands]
        cands_section = (
            f"Candidate tools (matched rule per tool):\n"
            f"{json.dumps(flat_cands, ensure_ascii=False, indent=2)}"
        )

    return f"""We are constructing a demonstration for tool selection.

Reaction (SMILES):
- reactants: {reactants}
- product: {product}

{cands_section}

In this demonstration, there is EXACTLY ONE trusted tool:
- trusted tool MUST be: "{gold_tool}"
- the following tools are NOT trusted in this demonstration: {json.dumps(neg_tools, ensure_ascii=False)}

Your job:
Write a concise and stable reasoning trace explaining WHY "{gold_tool}" is chosen over the non-trusted tools.
Base the reasoning primarily on the transformation implied by reactants→product and the rule explanations.
The tool_prediction field can be noisy; do NOT decide purely by majority vote.

Return ONLY valid JSON with this exact schema:
{{
  "tool": "{gold_tool}",
  "evidence": ["<2-4 bullet points tied to the transformation and specific explanations>"],
  "elimination": [{{"tool": "<non_trusted_tool>", "why_not": "<1 sentence>"}} ... up to 3 items],
  "final_reason": "<1-2 sentences summary>"
}}

Constraints:
- "tool" must be exactly "{gold_tool}".
- "elimination" tools must be selected from the non-trusted tools provided above and must be among: [{neg_str}].
- Do NOT mention the words: feasible, label, ground truth, training.
""".strip()


def _parse_guided_json(text: str, gold_tool: str, neg_tools: List[str]) -> Dict:
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.find("{"):s.rfind("}") + 1]
    obj = json.loads(s)

    tool = str(obj.get("tool", "")).strip()
    if tool != gold_tool:
        raise ValueError(f"tool mismatch: got={tool!r}, expected={gold_tool!r}")

    evidence = obj.get("evidence", [])
    evidence = [str(x).strip() for x in (evidence if isinstance(evidence, list) else [])
                if str(x).strip()][:6]

    cleaned_elim = []
    for x in (obj.get("elimination", []) or [])[:10]:
        if not isinstance(x, dict):
            continue
        t = str(x.get("tool", "")).strip()
        w = str(x.get("why_not", "")).strip()
        if not t or t == gold_tool or not w or t not in neg_tools:
            continue
        cleaned_elim.append({"tool": t, "why_not": w})
    cleaned_elim = cleaned_elim[:3]

    final_reason = str(obj.get("final_reason", "")).strip()
    return {
        "tool":         gold_tool,
        "evidence":     evidence,
        "elimination":  cleaned_elim,
        "final_reason": final_reason,
    }


def llm_guided_demo(
    pipe,
    reactants: str,
    product: str,
    cands: List[Dict],
    gold_tool: str,
    neg_tools: List[str],
    show_tiers: bool = True,
    max_tokens: int = 512,
    retries: int = 3,
) -> Dict:
    """
    Generate a guided demo with revealed gold_tool.
    Returns {"demo": {...}, "_raw": ...} or None if all retries fail.
    """
    user_prompt = build_guided_prompt(reactants, product, cands, gold_tool, neg_tools, show_tiers)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    last_err = None
    last_raw = ""
    for _ in range(retries + 1):
        outputs = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)
        raw = outputs[0]["generated_text"].strip()
        last_raw = raw
        try:
            parsed = _parse_guided_json(raw, gold_tool, neg_tools)
            return {"demo": parsed, "_raw": raw}
        except Exception as e:
            last_err = e

    print(f"    [WARN] guided demo failed for gold_tool={gold_tool}: {last_err}\n    raw: {last_raw[:200]}")
    return None
