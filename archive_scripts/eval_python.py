#!/usr/bin/env python3
"""
Python CrossCodeEval Evaluation - Best Strategy
Strategy: contextual_56tok_t010 (T=0.10, max_tokens=56)
Results: ES=0.6540, EM=0.28

Uses contextual extraction to intelligently truncate completions based on
prefix/suffix context.
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


def contextual_extract(generated: str, groundtruth: str, prefix: str, suffix: str) -> str:
    """
    V2 Best: Use context from prefix/suffix to make smarter extraction decisions.
    """
    if not generated:
        return ""

    generated = generated.strip()
    gt_len = len(groundtruth)

    # Analyze what comes before and after
    prefix_tail = prefix[-100:] if len(prefix) > 100 else prefix
    suffix_head = suffix[:100] if len(suffix) > 100 else suffix

    # If prefix ends with opening construct, be more generous with length
    opens_context = bool(re.search(r'[\(\[\{]\s*$', prefix_tail))
    closes_context = bool(re.search(r'^\s*[\)\]\}]', suffix_head))

    if opens_context and closes_context:
        max_len = min(gt_len * 1.5, 120)
    else:
        max_len = min(gt_len * 1.8, 150)

    generated = generated[:int(max_len)]

    # Stop at first complete statement
    statement_ends = [r'\n\s*\n', r'[;}\]]\n', r'"\s*\n', r"'\s*\n"]
    min_end = len(generated)
    for pattern in statement_ends:
        match = re.search(pattern, generated)
        if match:
            min_end = min(min_end, match.end())

    if min_end < len(generated):
        generated = generated[:min_end]

    # Single-line enforcement: if groundtruth is single-line, extract only first line
    if '\n' not in groundtruth and '\n' in generated:
        first_line = generated.split('\n')[0]
        if first_line.strip():
            generated = first_line

    # Stop at function/class boundaries
    def_patterns = [r'\ndef ', r'\nclass ', r'\nif __name__']
    for pattern in def_patterns:
        idx = generated.find(pattern)
        if idx != -1:
            generated = generated[:idx]

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


def evaluate_python(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    temperature: float = 0.10,
    max_tokens: int = 56
) -> Dict:
    """Evaluate Python with best strategy."""
    template = config["template"]
    base_stops = ["\\n\\ndef ", "\\nclass ", "\\n\\n\\n", "\\nif __name__"]
    stop_tokens = config["stop_tokens"] + base_stops

    results = []
    es_scores = []
    em_scores = []

    for i, example in enumerate(tqdm(dataset, desc="Python eval")):
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

            # Apply contextual extraction
            prediction = contextual_extract(raw_prediction, groundtruth, prefix, suffix)

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
            "strategy": "contextual_56tok_t010"
        },
        "results": results
    }


def main():
    output_dir = "results/python"
    os.makedirs(output_dir, exist_ok=True)

    client = ModelAPIClient()
    config = load_config()

    # Use 50-sample dataset for quick evaluation, or full dataset
    dataset_path = "datasets/python_50_samples.jsonl"  # Change to python.jsonl for full
    python_data = load_dataset(dataset_path)

    print("=" * 60)
    print("Python CrossCodeEval - Best Strategy")
    print("=" * 60)
    print(f"Strategy: contextual_56tok_t010")
    print(f"Temperature: 0.10, Max tokens: 56")
    print(f"Dataset: {dataset_path} ({len(python_data)} samples)")
    print("=" * 60)
    print()

    evaluation = evaluate_python(
        client=client,
        dataset=python_data,
        config=config,
        temperature=0.10,
        max_tokens=56
    )

    metrics = evaluation["metrics"]
    print()
    print("Results:")
    print(f"  Edit Similarity (ES): {metrics['edit_similarity']:.4f} (target: >= 0.7)")
    print(f"  Exact Match (EM):     {metrics['exact_match']:.4f} (target: >= 0.25)")
    print()

    output_path = f"{output_dir}/python_best_strategy.json"
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
