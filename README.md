# ARMOR Pipeline

This is the official code for the paper: **ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning**.

## Overview

A three-module pipeline for chemistry reaction feasibility prediction:

1. **Module 1: Tool Hierarchy Construction** — Build a two-tier tool hierarchy; split samples into unanimous / disagreement sets.
2. **Module 2: Utility-Aware Tool Prioritization** — Extract and refine per-tool reaction rules; match rules against disagreement samples.
3. **Module 3: Tool Conflict Resolution** — Build guided demonstrations; use FAISS + LLM to resolve tool conflicts on the test set.

> **Note:** "rule" in this repository corresponds to "pattern" in the paper.

---

## Data

**Input:** `example_data/dev/` and `example_data/test/`, each containing one JSONL file per tool with predictions and labels.

**Output:** `evaluation/predictions_ARMOR.jsonl` — per-sample final predictions, alongside the metrics table printed by `eval_test.py`.

### Input Format

ARMOR is applicable to the reaction feasibility task where multiple tools produce per-sample predictions. Format each tool's predictions as a JSONL file where every line is, like the example data:

```json
{"idx": 0, "reactants": "<SMILES>", "product": "<SMILES>", "label": "1", "tool_prediction": "1"}
```

- `idx`: integer sample index, consistent across all tool files and starting from 0
- `reactants` / `product`: SMILES strings
- `label`: ground-truth binary label (`"0"` or `"1"`)
- `tool_prediction`: tool's prediction (`"0"`, `"1"`, or `"NA"` for abstention)

Place one such file per tool under `example_data/dev/` and `example_data/test/`, named `{tool_name}.jsonl`, and update `tool_list` in `config.yaml` accordingly.

---

## Environment Setup

**Python:** 3.10+

**Step 1 — Install PyTorch** (choose the command matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
# Example: CUDA 12.4
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**Step 2 — Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

> `requirements.txt` lists `faiss-gpu` by default. Replace with `faiss-cpu` if running on CPU only.

**Step 3 — Download the LLM and set the model path** in `config.yaml`.

The pipeline uses [T3Q-qwen2.5-14b-v1.0-e3](https://huggingface.co/JungZoona/T3Q-qwen2.5-14b-v1.0-e3) as the LLM. Download it from HuggingFace and set the local path:

```yaml
model_paths:
  t3q: "/path/to/T3Q-qwen2.5-14b-v1.0-e3"
```

---

## Configuration

Edit `config.yaml` before running. Key parameters:

| Key | Default | Description |
|---|---|---|
| `llm` | `t3q` | LLM used throughout the pipeline |
| `model_paths.t3q` | `""` | Path to T3Q model directory |
| `hierarchy.p` | `0.25` | Fraction of tools in tier 1 |
| `subset.N_list` | `[5,10,25,45]` | Subset sizes for rule extraction |
| `subset.M` | `100` | Number of subsets per (tool, N) |
| `rule_filtering.tau_1` | `1.0` | Min align_score |
| `rule_filtering.tau_2` | `1.0` | Min coverage_score |
| `ruleset.tau_3` | `0.5` | Min rule confidence |
| `ruleset.top_p` | `5` | Max rules per tool |
| `selection.L` | `5` | Top-L candidates shown to LLM |
| `selection.top_k` | `8` | FAISS-retrieved reference reactions |
| `cuda_visible_devices` | `"0"` | GPU device(s) for LLM inference |

---

## Running the Pipeline

Run each module in order from the `Codes/` root directory:

```bash
# Module 1: build tool hierarchy and split dev/test samples
bash run_module1.sh

# Module 2: extract and match per-tool rules, prepare candidates
bash run_module2.sh

# Module 3: build memory and resolve test conflicts via LLM
bash run_module3.sh

# Evaluation: compute metrics and save final predictions
bash run_eval.sh
```

The four scripts must be run in order. See each script for per-step details.

---

## Repository Structure

```
Codes/
├── config.yaml
├── requirements.txt
├── run_module1.sh
├── run_module2.sh
├── run_module3.sh
├── run_eval.sh
├── utils/
│   └── utils.py
├── example_data/
│   ├── dev/        {tool}.jsonl for each tool
│   └── test/       {tool}.jsonl for each tool
├── module1_tool_hierarchy_construction/
│   ├── define_hierarchy.py
│   └── find_test_disagreement.py
│   (splits/ and test_splits/ are created at runtime)
├── module2_utility_aware_tool_prioritization/
│   ├── reaction_specific_tool_utility_assessment/
│   │   ├── pattern_extraction/
│   │   │   ├── subset_construction.py
│   │   │   └── rule_detection.py
│   │   ├── pattern_refinement/
│   │   │   ├── rule_verify.py
│   │   │   ├── validate_rules.py
│   │   │   └── rule_filtering.py
│   │   └── pattern_consolidation/
│   │       ├── aggregation.py
│   │       ├── prune_rules.py
│   │       ├── compute_rule_stats.py
│   │       └── generate_ruleset.py
│   └── pattern_based_tool_selection/
│       ├── rule_matching/
│       │   ├── match_rules.py
│       │   └── aggregate_matching.py
│       └── candidate_preparation/
│           └── preprocess.py
├── module3_tool_conflict_resolution/
│   ├── tool_conflict_memory/
│   │   └── build_memory.py
│   └── memory_augmented_conflict_resolution/
│       └── conflict_resolution.py
└── evaluation/
    └── eval_test.py
```
