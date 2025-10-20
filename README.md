# Code Completion Evaluation Suite

Clean evaluation framework for code completion models.

## Overview

Evaluation scripts for:
- **CrossCode Python**: FIM-based Python code completion
- **CrossCode Java**: FIM-based Java code completion
- **HumanEval**: Python code generation benchmark (via evalplus)

## Structure

```
.
├── eval_fim.py              # FIM evaluation for CrossCode
├── eval_humaneval.py        # HumanEval completion generation
├── eval_humaneval.sh        # HumanEval full pipeline
├── eval_utils.py            # Shared utilities
├── api_client.py            # Model API client
├── config.json              # Model configuration
├── datasets/
│   ├── crosscode_python.jsonl
│   └── crosscode_java.jsonl
└── results/
    ├── crosscode/
    │   ├── python/{ModelName}/
    │   └── java/{ModelName}/
    └── humaneval/{ModelName}/
```

## Installation

```bash
# Basic requirements
pip install tqdm python-Levenshtein

# For HumanEval
pip install datasets evalplus
```

## Usage

### CrossCode Evaluation

**Python:**
```bash
# Using 3B model (default)
python3 eval_fim.py --dataset crosscode_python --temperature 0.1 --max-tokens 56

# Using 7B model
python3 eval_fim.py --dataset crosscode_python --model 7b --temperature 0.1 --max-tokens 56

# With post-processing
python3 eval_fim.py --dataset crosscode_python --model 3b --post-process
```

**Java:**
```bash
python3 eval_fim.py --dataset crosscode_java --model 3b --temperature 0.2 --max-tokens 48
```

**Output:** `results/crosscode/{language}/{ModelName}/...`

### HumanEval Evaluation

**Full pipeline (recommended):**
```bash
# Default (3B model)
./eval_humaneval.sh

# 7B model
MODEL=7b ./eval_humaneval.sh

# Custom parameters
MODEL=7b TEMPERATURE=0.2 MAX_TOKENS=512 ./eval_humaneval.sh
```

**Step by step:**
```bash
# 1. Generate completions
python3 eval_humaneval.py --model 7b --temperature 0.2

# 2. Evaluate with evalplus
evalplus.evaluate --dataset humaneval --samples results/humaneval/Qwen/Qwen2.5-Coder-7B/samples_t20_n1.jsonl
```

**Output:** `results/humaneval/{ModelName}/...`

## Output Structure

Results organized by:
1. Dataset type (crosscode/humaneval)
2. Language (python/java) - CrossCode only
3. Full model name (e.g., Qwen/Qwen2.5-Coder-3B)

Example paths:
- `results/crosscode/python/Qwen/Qwen2.5-Coder-3B/python_t10_tok56.json`
- `results/humaneval/Qwen/Qwen2.5-Coder-7B/samples_t20_n1.jsonl`

## Metrics

### CrossCode
- **Accuracy**: % passing ES ≥ 0.7 AND EM = True
- **Edit Similarity (ES)**: Levenshtein similarity
- **Exact Match (EM)**: Normalized exact match

### HumanEval
- **pass@k**: Execution-based via evalplus
- Includes HumanEval+ (extra test cases)

## Post-Processing

`--post-process` applies legitimate corrections only:
✅ Remove markdown, balance brackets, fix indentation
❌ No ground truth tricks

## Examples

```bash
# CrossCode Python with 7B model
python3 eval_fim.py --dataset crosscode_python --model 7b --temperature 0.1 --max-tokens 56

# HumanEval with 7B model
MODEL=7b ./eval_humaneval.sh
```

## Key Features

✅ Organized by model name
✅ evalplus for HumanEval (base + plus)
✅ No ground truth tricks
✅ Clean 3-script architecture

