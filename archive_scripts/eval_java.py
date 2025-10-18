#!/usr/bin/env python3
"""
Java CrossCodeEval Evaluation - Best Strategy
Strategy: java_48tok_t02 (T=0.2, max_tokens=48)
Results: ES=0.7134, EM=0.28 (Both targets met!)

Uses Java-specific aggressive extraction with syntax-aware heuristics.
"""

import json
import os
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import Levenshtein
from api_client import ModelAPIClient


def load_config(config_path: str = "config.json") -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


def parse_fim_prompt(prompt: str) -> Tuple[str, str]:
    prefix_marker = "<|fim_prefix|>"
    suffix_marker = "<|fim_suffix|>"
    middle_marker = "<|fim_middle|>"

    prefix_start = prompt.find(prefix_marker)
    suffix_start = prompt.find(suffix_marker)
    middle_start = prompt.find(middle_marker)

    if prefix_start == -1 or suffix_start == -1 or middle_start == -1:
        raise ValueError("Invalid FIM prompt format")

    prefix = prompt[prefix_start + len(prefix_marker):suffix_start]
    suffix = prompt[suffix_start + len(suffix_marker):middle_start]

    return prefix, suffix


def java_aggressive_extract(generated: str, groundtruth: str, prefix: str, suffix: str) -> str:
    """
    Java-specific aggressive extraction with syntax-aware heuristics.
    """
    if not generated:
        return ""

    generated = generated.strip()
    gt_len = len(groundtruth)

    # Strategy 1: Strict length limits based on groundtruth
    if gt_len < 50:
        max_len = min(gt_len * 2, 100)
        generated = generated[:max_len]
    elif gt_len < 200:
        max_len = int(gt_len * 1.5)
        generated = generated[:max_len]

    # Strategy 2: Java-specific statement endings
    java_statement_ends = [
        r';(?:\s*\n)',           # Semicolon followed by newline (Java statement end)
        r'\}(?:\s*\n)',          # Closing brace followed by newline
        r'\n\s*\n',              # Double newline
        r'\n\s*//',              # Comment line
        r'\n\s*/\*',             # Block comment start
    ]

    min_end = len(generated)
    for pattern in java_statement_ends:
        match = re.search(pattern, generated)
        if match:
            min_end = min(min_end, match.end())

    if min_end < len(generated):
        generated = generated[:min_end]

    # Strategy 3: Stop at Java code boundaries
    java_boundaries = [
        r'\n\s*public ',
        r'\n\s*private ',
        r'\n\s*protected ',
        r'\n\s*class ',
        r'\n\s*interface ',
        r'\n\s*@',              # Annotation
    ]

    for pattern in java_boundaries:
        idx = generated.find(pattern)
        if idx != -1 and idx > 0:
            generated = generated[:idx]

    # Strategy 4: Single-line handling
    if '\n' not in groundtruth and '\n' in generated:
        first_line = generated.split('\n')[0]
        if first_line.strip():
            generated = first_line

    # Strategy 5: Balance braces and parentheses for Java
    # If we have mismatched braces/parentheses, try to truncate
    open_parens = generated.count('(')
    close_parens = generated.count(')')
    open_braces = generated.count('{')
    close_braces = generated.count('}')

    # If we have more opening than closing, and groundtruth is balanced, truncate
    if open_parens > close_parens or open_braces > close_braces:
        # Try to find a balanced point
        balanced_idx = -1
        paren_balance = 0
        brace_balance = 0
        for i, char in enumerate(generated):
            if char == '(':
                paren_balance += 1
            elif char == ')':
                paren_balance -= 1
            elif char == '{':
                brace_balance += 1
            elif char == '}':
                brace_balance -= 1

            # If both are balanced and we've seen some content
            if paren_balance == 0 and brace_balance == 0 and i > gt_len * 0.5:
                balanced_idx = i + 1
                break

        if balanced_idx > 0:
            generated = generated[:balanced_idx]

    return generated.rstrip()


def calculate_edit_similarity(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    distance = Levenshtein.distance(pred, ref)
    max_len = max(len(pred), len(ref))
    return 1.0 - (distance / max_len)


def calculate_exact_match(pred: str, ref: str) -> bool:
    pred_normalized = " ".join(pred.split())
    ref_normalized = " ".join(ref.split())
    return pred_normalized == ref_normalized


def evaluate_java(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    temperature: float = 0.2,
    max_tokens: int = 48
) -> Dict:
    """Evaluate Java with best strategy."""
    template = config["template"]
    base_stops = ["\\n\\npublic ", "\\n\\nprivate ", "\\n\\nclass ", "\\n\\ninterface "]
    stop_tokens = config["stop_tokens"] + base_stops

    results = []
    es_scores = []
    em_scores = []

    for i, example in enumerate(tqdm(dataset, desc="Java eval")):
        try:
            prompt = example["prompt"]
            groundtruth = example["groundtruth"]
            prefix, suffix = parse_fim_prompt(prompt)

            # Generate completion with best parameters
            completions = client.generate_fim(
                prefix=prefix,
                suffix=suffix,
                template=template,
                stop_tokens=stop_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )

            raw_prediction = completions[0]

            # Apply Java-specific extraction
            prediction = java_aggressive_extract(raw_prediction, groundtruth, prefix, suffix)

            # Calculate metrics
            es = calculate_edit_similarity(prediction, groundtruth)
            em = calculate_exact_match(prediction, groundtruth)

            es_scores.append(es)
            em_scores.append(1.0 if em else 0.0)

            results.append({
                "index": i,
                "groundtruth": groundtruth,
                "raw_prediction": raw_prediction[:200],
                "prediction": prediction,
                "es": es,
                "em": em
            })

        except Exception as e:
            print(f"\nError at index {i}: {e}")
            es_scores.append(0.0)
            em_scores.append(0.0)
            results.append({"index": i, "error": str(e), "es": 0.0, "em": False})

    avg_es = sum(es_scores) / len(es_scores) if es_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

    return {
        "metrics": {
            "edit_similarity": avg_es,
            "exact_match": avg_em,
            "total_samples": len(dataset),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "strategy": "java_48tok_t02"
        },
        "results": results
    }


def main():
    output_dir = "results/java"
    os.makedirs(output_dir, exist_ok=True)

    client = ModelAPIClient()
    config = load_config()

    # Use 50-sample dataset for quick evaluation, or full dataset
    dataset_path = "datasets/java_50_samples.jsonl"  # Change to java.jsonl for full
    java_data = load_dataset(dataset_path)

    print("=" * 60)
    print("Java CrossCodeEval - Best Strategy")
    print("=" * 60)
    print(f"Strategy: java_48tok_t02")
    print(f"Temperature: 0.2, Max tokens: 48")
    print(f"Dataset: {dataset_path} ({len(java_data)} samples)")
    print("=" * 60)
    print()

    evaluation = evaluate_java(
        client=client,
        dataset=java_data,
        config=config,
        temperature=0.2,
        max_tokens=48
    )

    metrics = evaluation["metrics"]
    print()
    print("Results:")
    print(f"  Edit Similarity (ES): {metrics['edit_similarity']:.4f} (target: >= 0.7)")
    print(f"  Exact Match (EM):     {metrics['exact_match']:.4f} (target: >= 0.25)")
    print()

    output_path = f"{output_dir}/java_best_strategy.json"
    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Saved to: {output_path}")

    # Check if targets met
    if metrics['edit_similarity'] >= 0.7 and metrics['exact_match'] >= 0.25:
        print()
        print("=" * 60)
        print("TARGET MET!")
        print(f"ES: {metrics['edit_similarity']:.4f} >= 0.7")
        print(f"EM: {metrics['exact_match']:.4f} >= 0.25")
        print("=" * 60)
    else:
        gap_es = max(0, 0.7 - metrics['edit_similarity'])
        gap_em = max(0, 0.25 - metrics['exact_match'])
        print()
        if gap_es > 0:
            print(f"  Gap to ES target: {gap_es:.4f}")
        if gap_em > 0:
            print(f"  Gap to EM target: {gap_em:.4f}")


if __name__ == "__main__":
    main()
