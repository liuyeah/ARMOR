import sys
import json
from pathlib import Path
from tqdm import tqdm

_BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(_BASE_DIR.parent.parent.parent / "utils"))
from utils import CFG, TOOL_LIST

def filter_rules(data, tau_1=1.0, tau_2=1.0):
    data["tools"] = [
        tool for tool in data.get("tools", [])
        if tool.get("true_acc") == tool.get("llm_reported_acc")
    ]
    for tool in data.get("tools", []):
        tool["often_correct"] = [
            r for r in tool.get("often_correct", [])
            if r.get("example_matching_rate") >= tau_1
            and (r.get("llm_validation") or {}).get("agreement_rate", 0.0) >= tau_2
        ]
    return data




def main():
    llm   = CFG["llm"]
    tau_1 = CFG["rule_filtering"]["tau_1"]
    tau_2 = CFG["rule_filtering"]["tau_2"]
    N_list = CFG["subset"]["N_list"]
    M      = CFG["subset"]["M"]

    for tool_name in TOOL_LIST:
        for N in N_list:
            output_data = {
                "tool_name": tool_name,
                "N": N,
                "often_correct": [],
            }
            for sample_num in tqdm(range(M), desc=f"TOOL: {tool_name}, N={N}, LLM: {llm}"):
                in_dir = str(_BASE_DIR / f"outputs_json_format/N={N}/{tool_name}/valid_by_{llm}"
                             / f"sample_{sample_num}_{llm}_json_format_pattern_validated_by_{llm}.json")

                data = json.load(open(in_dir, "r", encoding="utf-8"))
                data = filter_rules(data, tau_1=tau_1, tau_2=tau_2)

                for tool in data.get("tools", []):
                    output_data["often_correct"].extend(tool.get("often_correct", []))

            output_filename = str(_BASE_DIR / f"outputs_json_format/N={N}/{tool_name}"
                                  / f"{llm}_recognition_pattern_validated_by_{llm}.json")

            json.dump(
                output_data,
                open(output_filename, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
            



if __name__ == "__main__":
    main()