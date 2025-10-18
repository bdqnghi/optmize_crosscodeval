# Code Generation Optimization for Python and Java

This repository contains evaluation scripts for optimizing code generation models on Python and Java datasets using Fill-In-the-Middle (FIM) completion tasks.

## Overview

The goal is to optimize model parameters (temperature and max_tokens) to achieve three target metrics simultaneously:
- **Edit Similarity (ES)** ≥ 0.7
- **Exact Match (EM)** ≥ 0.25
- **pass@1** ≥ 0.55

## Datasets

Datasets are stored in `datasets/` directory but excluded from git (contains API tokens):
- `python.jsonl` - 2,665 Python FIM completion tasks
- `java.jsonl` - 2,139 Java FIM completion tasks
- `python_50_samples.jsonl` - 50-sample subset for optimization
- `java_50_samples.jsonl` - 50-sample subset for optimization

Each dataset entry contains:
```json
{
  "prompt": "<|fim_prefix|>...prefix code...<|fim_suffix|>...suffix code...<|fim_middle|>",
  "groundtruth": "expected middle code"
}
```

## Evaluation Metrics

### 1. Edit Similarity (ES)
Measures how close the generated code is to the ground truth using normalized Levenshtein distance:
```python
ES = 1 - (levenshtein_distance / max_length)
```
- Range: 0.0 to 1.0 (higher is better)
- Target: ≥ 0.7

### 2. Exact Match (EM)
Binary metric indicating if prediction exactly matches ground truth after whitespace normalization:
```python
EM = (normalized_prediction == normalized_groundtruth)
```
- Range: 0 (no match) or 1 (exact match)
- Target: ≥ 0.25 (25% exact matches across dataset)

### 3. pass@1
Combined metric requiring both high similarity AND exact match:
```python
pass = (ES >= 0.7) AND (EM == True)
pass@1 = count(passed_samples) / total_samples
```
- Range: 0.0 to 1.0
- Target: ≥ 0.55 (55% of samples must pass both criteria)

### 4. Balanced Score
Normalized weighted score across all three metrics:
```python
balanced_score = (ES/0.7 + EM/0.25 + pass@1/0.55) / 3
```
Used to rank configurations during optimization.

## Evaluation Process

### 1. Code Extraction Strategies

**Python** (`contextual_extract`):
- Context-aware extraction based on surrounding code
- Detects opening/closing brackets to determine scope
- Adaptive length limits based on ground truth size
- Stops at statement boundaries (blank lines, function definitions)

**Java** (`java_aggressive_extract`):
- Language-specific boundary detection (class, method declarations)
- Balanced parentheses and braces tracking
- Statement-end detection (semicolons, closing braces)
- Single-line handling for simple completions

### 2. FIM Template
Uses Fill-In-the-Middle format:
```
<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>
```
The model generates the middle part between prefix and suffix.

### 3. Stop Tokens
Language-specific stop tokens prevent over-generation:
- **Python**: `\n\ndef `, `\nclass `, `\n\n\n`, `\nif __name__`
- **Java**: `\n\npublic `, `\n\nprivate `, `\n\nclass `, `\n\ninterface `

## Running Evaluations

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages: `tqdm`, `Levenshtein`, `requests`

### Configuration
Edit `config.json`:
```json
{
  "api_base": "http://your-api-endpoint/v1/completions",
  "model": "your-model-name",
  "template": "<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
  "stop_tokens": []
}
```

### Run Full Dataset Evaluation

**Python:**
```bash
python3 eval_python_all.py
```
- Uses best config: T=0.10, max_tokens=56
- Evaluates all 2,665 samples
- Outputs to `results/python_full/`

**Java:**
```bash
python3 eval_java_all.py
```
- Uses best config: T=0.20, max_tokens=48
- Evaluates all 2,139 samples
- Outputs to `results/java_full/`

## Results Directory Structure

```
results/
├── python_full/
│   ├── python_t10_tok56.json          # Full evaluation results
│   └── optimization_summary.json      # Summary with all metrics
└── java_full/
    ├── java_t20_tok48.json            # Full evaluation results
    └── optimization_summary.json      # Summary with all metrics
```

### optimization_summary.json
Contains:
```json
{
  "best_config": {
    "temp": 0.10,
    "tokens": 56
  },
  "best_score": 0.8143,
  "best_metrics": {
    "pass@1": 0.26,
    "edit_similarity": 0.6513,
    "exact_match": 0.26,
    "balanced_score": 0.8143,
    "total_samples": 2665,
    "passes": 692,
    "temperature": 0.10,
    "max_tokens": 56
  },
  "all_evaluations": [...]
}
```

### Individual Result Files (e.g., python_t10_tok56.json)
```json
{
  "metrics": {
    "pass@1": 0.26,
    "edit_similarity": 0.6513,
    "exact_match": 0.26,
    "balanced_score": 0.8143,
    "total_samples": 2665,
    "passes": 692,
    "temperature": 0.10,
    "max_tokens": 56
  },
  "results": [
    {
      "index": 0,
      "groundtruth": "actual code",
      "prediction": "generated code",
      "es": 0.95,
      "em": true,
      "passed": true
    },
    ...
  ]
}
```

## Optimization Results

### Python (2,665 samples)
**Best Configuration:** T=0.10, max_tokens=56
- Edit Similarity: 0.6513 ❌ (target: 0.70)
- Exact Match: 0.26 ✅ (target: 0.25)
- pass@1: 0.26 ❌ (target: 0.55)
- Balanced Score: 0.8143

### Java (2,139 samples)
**Best Configuration:** T=0.20, max_tokens=48
- Edit Similarity: 0.7304 ✅ (target: 0.70)
- Exact Match: 0.30 ✅ (target: 0.25)
- pass@1: 0.30 ❌ (target: 0.55)
- Balanced Score: 0.9296

### Key Findings
1. **Java outperforms Python** across all metrics (~12% higher ES, ~15% higher EM)
2. **pass@1 bottleneck**: Both languages struggle with the combined requirement (ES ≥ 0.7 AND EM = True)
3. **Temperature sensitivity**: Lower temperatures (0.10-0.20) perform better than ultra-low (0.01-0.03)
4. **Token limits**: Shorter generation lengths (48-56 tokens) work best for these tasks

## Architecture

```
├── api_client.py              # API client for model inference
├── eval_python_all.py         # Python full dataset evaluation
├── eval_java_all.py           # Java full dataset evaluation
├── config.json                # Model and API configuration
├── requirements.txt           # Python dependencies
├── datasets/                  # Input datasets (gitignored)
├── results/                   # Evaluation outputs (gitignored)
└── logs/                      # Execution logs (gitignored)
```

## API Client

The `ModelAPIClient` class handles:
- FIM template formatting
- Stop token management
- Retry logic with exponential backoff
- Temperature and max_tokens parameter control
- Multiple completions (n parameter)

## Logs

Execution logs are saved to `logs/`:
- `python_full.log` - Python evaluation progress
- `java_full.log` - Java evaluation progress

Logs include progress bars, metrics, and any errors encountered.

## Notes

- Evaluations can take several hours for full datasets (~1-2 seconds per sample)
- Results are saved incrementally (can resume if interrupted)
- All sensitive data (API tokens, credentials) are excluded from git
- The model used: Qwen/Qwen2.5-Coder-3B (or as configured)

## Future Work

1. **Improve pass@1**: Current ceiling at ~26-30%, needs investigation
2. **Alternative extraction strategies**: May improve ES for Python
3. **Larger temperature grid search**: Explore 0.15-0.25 range
4. **Longer token limits**: Test 64-128 tokens for complex completions
5. **n>1 sampling**: Use multiple samples and select best
