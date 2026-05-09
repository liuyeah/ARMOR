import sys
import json, re
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple

_BASE_DIR  = Path(__file__).parent
_CODES_DIR = _BASE_DIR.parent.parent.parent
sys.path.insert(0, str(_CODES_DIR / "utils"))
from utils import CFG, TOOL_LIST

PATTERN_EXTRACTION_OUTPUTS = _BASE_DIR.parent / "pattern_extraction" / "outputs"
EXAMPLE_DATA_DIR = _CODES_DIR / "example_data"

PCT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$")

def _parse_acc(v: Any) -> Optional[float]:
    """
    Accepts: "80.00%", "80", 80, 0.8
    Returns: float in [0,1] or None
    """
    if v is None:
        return None
    if isinstance(v, str):
        m = PCT_RE.match(v)
        if not m:
            return None
        x = float(m.group(1))
        # if looks like percent
        return x / 100.0 if x > 1.0 else x
    if isinstance(v, (int, float)):
        x = float(v)
        return x / 100.0 if x > 1.0 else x
    return None

def _norm_pred(v: Any) -> Optional[int]:
    """Return 0/1, else None for NA/None/missing/unparseable."""
    if v is None: return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("na", "n/a", "none", ""): return None
        if s.isdigit():
            iv = int(s)
            return iv if iv in (0, 1) else None
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv in (0, 1) else None
    return None



def parse_llm_report_json(text_or_obj: Any) -> Dict[str, Any]:
    """
    Input can be:
      - a JSON string (your new file content)
      - a Python dict (already loaded)
    Output keeps your old schema keys:
      acc/acc_str/often_correct
    """
    obj = json.loads(text_or_obj) if isinstance(text_or_obj, str) else text_or_obj

    acc = _parse_acc(obj.get("tool_acc"))
    tool = {
        "acc": acc,
        "acc_str": (f"{acc*100:.2f}%" if acc is not None else None),
        "often_correct": [],
    }

    for it in obj.get("often_correct_on", []) or []:
        tool["often_correct"].append({
            "name": (it.get("name") or "").strip(),
            "explanation": (it.get("explanation") or "").strip(),
            "examples_idx": [int(x) for x in (it.get("examples_idx") or [])],
        })

    return tool


def compute_true_acc(
    jsonl_path: str,
    pred_field: str = "tool_prediction",
    label_field: str = "label",
    idx_field: str = "idx",
) -> Dict[str, Any]:
    total = correct = 0
    idx_map: Dict[int, Tuple[int, Optional[int]]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            idx = int(obj[idx_field])
            y = int(obj[label_field])
            p = _norm_pred(obj.get(pred_field, None))
            idx_map[idx] = (y, p)

            total += 1
            if p is not None and p == y:
                correct += 1

    acc = correct / total if total else 0.0
    return {
        "acc": acc,
        "acc_str": f"{acc*100:.2f}%",
        "total": total,
        "correct": correct,
        "idx_map": idx_map,
        "pred_field": pred_field,
    }


def postprocess(
    llm_text: str,
    jsonl_path: str,
    tool_name: str,
    pred_field: str = "tool_prediction",
) -> Dict[str, Any]:
    tool = parse_llm_report_json(llm_text)
    stats = compute_true_acc(jsonl_path, pred_field=pred_field)
    idx_map = stats.pop("idx_map")

    tool["pred_field_used"] = stats["pred_field"]

    tool["true_acc"] = stats["acc"]
    tool["true_acc_str"] = stats["acc_str"]
    tool["total"] = stats["total"]
    tool["correct"] = stats["correct"]

    tool["llm_reported_acc"] = tool.pop("acc", None)
    tool["llm_reported_acc_str"] = tool.pop("acc_str", None)
    tool["acc_match"] = (tool.get("llm_reported_acc_str") == tool.get("true_acc_str"))

    def is_correct(idx: int) -> Optional[bool]:
        if idx not in idx_map:
            return None
        y, p = idx_map[idx]
        if p is None:
            return False
        return (p == y)

    def annotate_pattern(pat: Dict[str, Any], section: str):
        ex = pat.get("examples_idx", []) or []
        denom = len(ex)
        match = 0
        mismatch_idx = []

        expected_ok = (section == "often_correct")

        for idx in ex:
            idx = int(idx)
            ok = is_correct(idx)
            if ok is None:
                mismatch_idx.append(idx)
                continue
            if ok == expected_ok:
                match += 1
            else:
                mismatch_idx.append(idx)

        rate = (match / denom) if denom else 0.0
        pat["example_matching_rate"] = rate
        pat["example_matching_rate_str"] = f"{rate*100:.2f}%"
        pat["mismatch_idx"] = sorted(set(mismatch_idx))

    for pat in tool.get("often_correct", []):
        annotate_pattern(pat, "often_correct")

    ordered_tool = {
        "tool_name": tool_name,

        # ---- tool-level summary FIRST ----
        "pred_field_used": tool["pred_field_used"],

        "true_acc": tool["true_acc"],
        "true_acc_str": tool["true_acc_str"],
        "total": tool["total"],
        "correct": tool["correct"],

        "llm_reported_acc": tool["llm_reported_acc"],
        "llm_reported_acc_str": tool["llm_reported_acc_str"],
        "acc_match": tool["acc_match"],

        # ---- qualitative analysis AFTER ----
        "often_correct": tool["often_correct"],
    }

    return {"tools": [ordered_tool]}



if __name__ == '__main__':
    llm    = CFG["llm"]
    N_list = CFG["subset"]["N_list"]
    M      = CFG["subset"]["M"]

    for tool in TOOL_LIST:
        for N in N_list:
            for sample_num in tqdm(range(M), desc=f"TOOL: {tool}, N={N}, LLM: {llm}"):
                llm_text = open(str(PATTERN_EXTRACTION_OUTPUTS / f"N={N}" / tool / f"sample_{sample_num}_{llm}.jsonl"), "r", encoding="utf-8").read()
                subset_jsonl = str(PATTERN_EXTRACTION_OUTPUTS / f"N={N}" / tool / f"sample_{sample_num}.jsonl")
                out = postprocess(
                    llm_text=llm_text,
                    jsonl_path=subset_jsonl,
                    tool_name=tool,
                    pred_field="tool_prediction",
                )

                output_dir = _BASE_DIR / f"outputs_json_format/N={N}/{tool}"
                output_dir.mkdir(parents=True, exist_ok=True)
                json.dump(out, open(str(output_dir / f"sample_{sample_num}_{llm}_json_format_pattern.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)