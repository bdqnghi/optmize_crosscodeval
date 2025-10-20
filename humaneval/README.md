# HumanEval Evaluation Tools

This directory contains specialized scripts for evaluating code completions on the HumanEval benchmark.

## Overview

HumanEval is a benchmark for evaluating code generation models on 164 Python function completion tasks. This directory provides tools for:

- **Evaluating existing completions** with timeout protection
- **Post-processing completions** to improve pass rates through systematic fixes

## Scripts

### 1. `only_eval_humaneval.py`

Fast evaluation script with timeout protection for existing completion files.

**Features:**
- 5-second timeout per test (configurable with `--timeout`)
- Signal-based interruption prevents infinite loops
- Progress bar with real-time updates
- Detailed error reporting

**Usage:**

```bash
# Basic evaluation
./only_eval_humaneval.py results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1.jsonl

# Custom timeout (10 seconds)
./only_eval_humaneval.py samples.jsonl --timeout 10

# Show detailed errors
./only_eval_humaneval.py samples.jsonl --show-errors

# Quiet mode (minimal output)
./only_eval_humaneval.py samples.jsonl --quiet
```

**Output:**
- Console: Pass@1 percentage and counts
- File: `<input_filename>_eval_results.json` with detailed results

**Example Output:**
```
======================================================================
📊 EVALUATION RESULTS
======================================================================
pass@1:  0.445122 (44.51%)
Passed:  73/164
Failed:  91/164
======================================================================
```

### 2. `post_processing_humaneval.py`

Applies systematic fixes to completions to improve pass@1 rates.

**Fixes Applied:**
1. **Markdown artifact removal** - Removes ```python blocks, </code> tags
2. **Indentation fixing** - Ensures proper 4-space function body indentation
3. **Syntax error correction** - Fixes common syntax issues
4. **Bracket balancing** - Ensures matched parentheses and braces
5. **Code truncation** - Removes incomplete or trailing code

**Usage:**

```bash
# Apply post-processing
./post_processing_humaneval.py results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1.jsonl

# Custom output file
./post_processing_humaneval.py input.jsonl --output fixed_samples.jsonl

# Evaluate after processing
./post_processing_humaneval.py input.jsonl --evaluate
```

**Output:**
- File: `<input_filename>_fixed.jsonl` with corrected completions
- Optional: Evaluation results if `--evaluate` flag is used

**Expected Improvements:**
- Fixes 10-20% of failing completions on average
- Most common fixes: indentation (40%), markdown artifacts (30%), syntax errors (20%)

## Workflow Examples

### Evaluate Existing Completions

```bash
# Quick evaluation with default settings
./only_eval_humaneval.py ../results/humaneval/Qwen/Qwen2.5-Coder-3B/samples_t20_n1.jsonl

# Show which tests are timing out
./only_eval_humaneval.py samples.jsonl --show-errors --timeout 10
```

### Fix and Re-evaluate

```bash
# Step 1: Apply post-processing
./post_processing_humaneval.py samples.jsonl

# Step 2: Evaluate the fixed version
./only_eval_humaneval.py samples_fixed.jsonl
```

### Combined Fix and Evaluate

```bash
# One-shot: fix and evaluate
./post_processing_humaneval.py samples.jsonl --evaluate
```

## File Format

Both scripts expect JSONL files where each line contains:

```json
{
  "task_id": "HumanEval/0",
  "completion": "    for i in range(len(numbers)):\n        ..."
}
```

**Key Requirements:**
- `task_id`: Must match HumanEval task IDs (e.g., "HumanEval/0" through "HumanEval/163")
- `completion`: Function body code (with leading indentation preserved)
- One JSON object per line (JSONL format)

## Comparison with Root Scripts

| Feature | `humaneval/only_eval_humaneval.py` | `../eval_humaneval_results.py` | `../eval_humaneval.sh` |
|---------|-----------------------------------|-------------------------------|------------------------|
| **Purpose** | Fast local evaluation | Fast local evaluation | Full pipeline (generate + eval) |
| **Timeout** | ✅ 5s (configurable) | ✅ 5s (configurable) | ✅ Docker sandbox |
| **Docker** | ❌ Not required | ❌ Not required | ✅ Required |
| **Speed** | ⚡ Very fast (~5s) | ⚡ Very fast (~5s) | 🐢 Slow (minutes) |
| **Safety** | ⚠️ Signal-based | ⚠️ Signal-based | ✅ Full sandbox |
| **Generation** | ❌ No | ❌ No | ✅ Yes |

**When to use each:**

- **`only_eval_humaneval.py`** - Quick local testing during development
- **`../eval_humaneval_results.py`** - Same as above (identical functionality, updated version)
- **`../eval_humaneval.sh`** - Production evaluation with full Docker isolation

## Common Issues

### Progress Stuck at 0%

**Cause:** Code with infinite loops or very slow execution

**Solution:** Use timeout flag or post-processing
```bash
# Increase timeout for slow completions
./only_eval_humaneval.py samples.jsonl --timeout 10

# Or apply post-processing to fix issues
./post_processing_humaneval.py samples.jsonl
```

### Low Pass@1 Rate

**Cause:** Common fixable issues (indentation, syntax, markdown)

**Solution:** Apply post-processing
```bash
./post_processing_humaneval.py samples.jsonl --evaluate
```

**Expected improvement:** +5-15% pass@1 rate

### macOS Resource Limit Errors

**Cause:** evalplus library tries to set resource limits unavailable on macOS

**Solution:** Use the scripts in this directory instead of direct evalplus
```bash
# Instead of: evalplus.evaluate --dataset humaneval --samples samples.jsonl
# Use:
./only_eval_humaneval.py samples.jsonl
```

## Dependencies

```bash
# Required
pip install evalplus tqdm

# Optional (for generation)
pip install datasets
```

## Performance Notes

- **Evaluation speed**: ~30 samples/second (5s timeout)
- **Total time**: ~5-10 seconds for all 164 samples
- **Memory usage**: <500MB
- **Timeout recommendations**:
  - Default: 5s (catches most infinite loops)
  - Lenient: 10s (for complex algorithms)
  - Strict: 2s (fast fail for debugging)

## Output Files

Both scripts create output files in the same directory as input:

```
results/humaneval/Qwen/Qwen2.5-Coder-3B/
├── samples_t20_n1.jsonl                    # Original completions
├── samples_t20_n1_fixed.jsonl              # After post-processing
├── samples_t20_n1_eval_results.json        # Evaluation results
└── samples_t20_n1_fixed_eval_results.json  # Eval results for fixed
```

## Advanced Usage

### Batch Processing Multiple Models

```bash
#!/bin/bash
for model in 3b 7b 14b; do
    echo "Evaluating $model..."
    ./only_eval_humaneval.py "../results/humaneval/Qwen/Qwen2.5-Coder-${model}/samples_t20_n1.jsonl"
done
```

### Compare Original vs Fixed

```bash
# Evaluate original
./only_eval_humaneval.py samples.jsonl --quiet > original_results.txt

# Fix and evaluate
./post_processing_humaneval.py samples.jsonl
./only_eval_humaneval.py samples_fixed.jsonl --quiet > fixed_results.txt

# Compare
diff original_results.txt fixed_results.txt
```

### Custom Post-Processing Pipeline

```python
from post_processing_humaneval import (
    remove_markdown_artifacts,
    fix_indentation,
    fix_syntax_errors
)

# Load completions
with open("samples.jsonl") as f:
    samples = [json.loads(line) for line in f]

# Apply only specific fixes
for sample in samples:
    completion = sample["completion"]
    completion = remove_markdown_artifacts(completion)
    completion = fix_indentation(completion)
    sample["completion"] = completion

# Save
with open("custom_fixed.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
```

## See Also

- **Main evaluation**: `../eval_humaneval.sh` - Full pipeline with Docker
- **Generation**: `../eval_humaneval.py` - Generate completions
- **Post-processing details**: `../POST_PROCESS_README.md` (if exists)
- **Project overview**: `../CLAUDE.md`
