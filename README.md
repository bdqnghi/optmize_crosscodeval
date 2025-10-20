# Code Completion Evaluation Suite

Evaluation framework for HumanEval and CrossCode benchmarks with automatic post-processing and metrics extraction.

## Quick Start

### HumanEval Evaluation

The `eval_humaneval.sh` script provides end-to-end evaluation including:
- Code completion generation with automatic post-processing (indentation fixes, typo correction, markdown cleanup)
- Code sanitization via evalplus
- Evaluation on HumanEval base and plus test suites
- Automatic JSON prettification and metrics extraction

**Basic Usage:**
```bash
# Default settings (temperature=0.2, max_tokens=512, num_samples=1)
MODEL=3b ./eval_humaneval.sh

# With temperature and top_p
MODEL=3b TEMPERATURE=0.2 TOP_P=0.95 ./eval_humaneval.sh

# With different temperature
MODEL=3b TEMPERATURE=0.8 TOP_P=0.95 MAX_TOKENS=512 ./eval_humaneval.sh

# All available parameters
MODEL=3b TEMPERATURE=0.2 TOP_P=0.95 MAX_TOKENS=512 NUM_SAMPLES=1 ./eval_humaneval.sh
```

**Evaluate Existing Results:**
```bash
EVAL_ONLY="path/to/samples.jsonl" ./eval_humaneval.sh
```

**Generation Only (without evaluation):**
```bash
python3 eval_humaneval.py --model 3b --temperature 0.2 --top-p 0.95 --max-tokens 512
```

**Environment Variables:**
- `MODEL`: Model size (3b, 7b, 14b, etc.) - default: 3b
- `TEMPERATURE`: Sampling temperature (0.0-1.0) - default: 0.2
- `TOP_P`: Nucleus sampling parameter (0.0-1.0) - optional
- `MAX_TOKENS`: Maximum tokens to generate - default: 512
- `NUM_SAMPLES`: Number of samples per problem - default: 1
- `EVAL_ONLY`: Path to existing samples file (skips generation)

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

## Output Structure

Results are organized in subfolders by evaluation parameters:

```
results/humaneval/{ModelName}/samples_t{temp}_p{topp}_n{samples}/
├── samples_t{temp}_p{topp}_n{samples}.jsonl                      # Raw completions with post-processing
├── samples_t{temp}_p{topp}_n{samples}-sanitized.jsonl            # Sanitized by evalplus
├── samples_t{temp}_p{topp}_n{samples}-sanitized_eval_results.json          # Full evaluation results (minified)
├── samples_t{temp}_p{topp}_n{samples}-sanitized_eval_results_pretty.json   # Human-readable results
└── samples_t{temp}_p{topp}_n{samples}-sanitized_summary.json               # Metrics summary (pass@k)
```

**Example:** `results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_p95_n1/`
- `samples_t20_p95_n1.jsonl` - 164 completions with automatic fixes
- `samples_t20_p95_n1-sanitized.jsonl` - Sanitized by evalplus
- `samples_t20_p95_n1-sanitized_eval_results_pretty.json` - Full results (readable)
- `samples_t20_p95_n1-sanitized_summary.json` - Quick metrics summary

**Summary JSON Format:**
```json
{
  "file": "samples_t20_p95_n1-sanitized_eval_results.json",
  "date": "2025-10-20 20:35",
  "hash": "fe585eb4df8c88d844eeb463ea4d0302",
  "metrics": {
    "base": {
      "pass": 65,
      "total": 164,
      "pass@1": 39.63
    },
    "plus": {
      "pass": 53,
      "total": 164,
      "pass@1": 32.32
    }
  }
}
```

**Understanding Metrics:**
- **Base** - Original HumanEval test cases (standard functionality)
- **Plus** - Extended test suite with edge cases and corner cases (more rigorous)
- **pass@1** - Percentage of problems solved correctly on first attempt

## Requirements

```bash
pip install tqdm python-Levenshtein datasets evalplus

# Docker required for HumanEval evaluation
docker pull ganler/evalplus:latest
```
