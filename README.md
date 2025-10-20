# Code Completion Evaluation Suite

Evaluation framework for HumanEval and CrossCode benchmarks.

## Quick Start

### HumanEval Evaluation

```bash
# End-to-end (generation + sanitization + evaluation)
MODEL=3b ./eval_humaneval.sh

# With custom parameters
MODEL=3b TEMPERATURE=0.8 MAX_TOKENS=512 ./eval_humaneval.sh

# Evaluate existing results (with sanitization)
EVAL_ONLY="results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1.jsonl" ./eval_humaneval.sh

# Generation only
python3 eval_humaneval.py --model 3b --temperature 0.2 --top-p 0.95
```

### CrossCode Evaluation (FIM)

```bash
# Python
python3 eval_fim.py --dataset crosscode_python --model 3b --temperature 0.2 --max-tokens 128

# Java
python3 eval_fim.py --dataset crosscode_java --model 3b --temperature 0.2 --max-tokens 128
```

## Available Models

Configure in `config.json`:
- `3b`: Qwen/Qwen2.5-Coder-3B (base)
- `3b-instruct`: Qwen/Qwen2.5-Coder-3B-Instruct
- `7b`: Qwen/Qwen2.5-Coder-7B (base)
- `14b`: Qwen/Qwen2.5-Coder-14B (base)

## Results Location

```
results/
├── humaneval/
│   └── {ModelName}/
│       ├── samples_t{temp}_n{samples}.jsonl               # Raw completions
│       ├── samples_t{temp}_n{samples}-sanitized.jsonl     # Sanitized code
│       └── samples_t{temp}_n{samples}-sanitized_eval_results.json  # Evaluation results
└── crosscode/
    ├── python/{ModelName}/
    │   └── python_t{temp}_tok{tokens}.json
    └── java/{ModelName}/
        └── java_t{temp}_tok{tokens}.json
```

Example:
- `results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1.jsonl` (raw)
- `results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1-sanitized.jsonl` (cleaned)
- `results/crosscode/python/Qwen/Qwen2.5-Coder-3B/python_t20_tok128.json`

## Requirements

```bash
pip install tqdm python-Levenshtein datasets evalplus

# Docker required for HumanEval evaluation
docker pull ganler/evalplus:latest
```
