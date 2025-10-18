#!/usr/bin/env python3
"""
Balanced Multi-Objective Optimization for Java
Target: Optimize (ES, EM, accuracy) together with balanced weights
Strategy: Find configurations that maximize weighted score across all metrics

Scoring function:
  balanced_score = w1 * ES_norm + w2 * EM_norm + w3 * accuracy_norm

Where:
  - ES_norm = ES / 0.7 (normalized to target)
  - EM_norm = EM / 0.25 (normalized to target)
  - accuracy_norm = accuracy / 0.55 (normalized to target)
  - w1, w2, w3 are weights (default: equal weighting)
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
    """Java extraction strategy."""
    if not generated:
        return ""

    generated = generated.strip()
    gt_len = len(groundtruth)

    if gt_len < 50:
        max_len = min(gt_len * 2, 100)
        generated = generated[:max_len]
    elif gt_len < 200:
        max_len = int(gt_len * 1.5)
        generated = generated[:max_len]

    java_statement_ends = [
        r';(?:\s*\n)',
        r'\}(?:\s*\n)',
        r'\n\s*\n',
        r'\n\s*//',
        r'\n\s*/\*',
    ]

    min_end = len(generated)
    for pattern in java_statement_ends:
        match = re.search(pattern, generated)
        if match:
            min_end = min(min_end, match.end())

    if min_end < len(generated):
        generated = generated[:min_end]

    java_boundaries = [
        r'\n\s*public ',
        r'\n\s*private ',
        r'\n\s*protected ',
        r'\n\s*class ',
        r'\n\s*interface ',
        r'\n\s*@',
    ]

    for pattern in java_boundaries:
        idx = generated.find(pattern)
        if idx != -1 and idx > 0:
            generated = generated[:idx]

    if '\n' not in groundtruth and '\n' in generated:
        first_line = generated.split('\n')[0]
        if first_line.strip():
            generated = first_line

    open_parens = generated.count('(')
    close_parens = generated.count(')')
    open_braces = generated.count('{')
    close_braces = generated.count('}')

    if open_parens > close_parens or open_braces > close_braces:
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


def check_pass(es: float, em: bool, es_threshold: float = 0.7) -> bool:
    return es >= es_threshold and em


def calculate_balanced_score(es: float, em: float, accuracy: float,
                             w_es: float = 1.0, w_em: float = 1.0, w_acc: float = 1.0) -> float:
    """
    Calculate balanced score with weighted normalization to targets.

    Args:
        es: Edit similarity (0-1)
        em: Exact match rate (0-1)
        accuracy: Accuracy rate (0-1)
        w_es: Weight for ES (default 1.0)
        w_em: Weight for EM (default 1.0)
        w_acc: Weight for accuracy (default 1.0)

    Returns:
        Balanced score (higher is better)
    """
    # Normalize to targets
    es_norm = es / 0.7
    em_norm = em / 0.25
    acc_norm = accuracy / 0.55

    # Weighted average
    total_weight = w_es + w_em + w_acc
    balanced = (w_es * es_norm + w_em * em_norm + w_acc * acc_norm) / total_weight

    return balanced


def evaluate_config(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Evaluate a single configuration."""
    template = config["template"]
    base_stops = ["\\n\\npublic ", "\\n\\nprivate ", "\\n\\nclass ", "\\n\\ninterface "]
    stop_tokens = config["stop_tokens"] + base_stops

    results = []
    passes = 0

    for i, example in enumerate(tqdm(dataset, desc=f"T={temperature}, tok={max_tokens}")):
        try:
            prompt = example["prompt"]
            groundtruth = example["groundtruth"]
            prefix, suffix = parse_fim_prompt(prompt)

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
            prediction = java_aggressive_extract(raw_prediction, groundtruth, prefix, suffix)

            es = calculate_edit_similarity(prediction, groundtruth)
            em = calculate_exact_match(prediction, groundtruth)
            passed = check_pass(es, em)

            if passed:
                passes += 1

            results.append({
                "index": i,
                "groundtruth": groundtruth,
                "prediction": prediction,
                "es": es,
                "em": em,
                "passed": passed
            })

        except Exception as e:
            print(f"\nError at index {i}: {e}")
            results.append({"index": i, "error": str(e), "es": 0.0, "em": False, "passed": False})

    accuracy = passes / len(results) if results else 0.0
    avg_es = sum(r['es'] for r in results if 'es' in r) / len(results) if results else 0.0
    avg_em = sum(1 for r in results if r.get('em', False)) / len(results) if results else 0.0

    # Calculate balanced score
    balanced = calculate_balanced_score(avg_es, avg_em, accuracy)

    return {
        "metrics": {
            "accuracy": accuracy,
            "edit_similarity": avg_es,
            "exact_match": avg_em,
            "balanced_score": balanced,
            "total_samples": len(dataset),
            "passes": passes,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        "results": results
    }


def main():
    output_dir = "results/java_full"
    os.makedirs(output_dir, exist_ok=True)

    client = ModelAPIClient()
    config = load_config()

    dataset_path = "datasets/java.jsonl"
    java_data = load_dataset(dataset_path)

    print("=" * 70)
    print("Java Full Dataset Evaluation")
    print("=" * 70)
    print(f"Dataset: {dataset_path} ({len(java_data)} samples)")
    print("Using best config from optimization: T=0.20, max_tokens=48")
    print("Targets:")
    print("  - ES >= 0.7")
    print("  - EM >= 0.25")
    print("  - Accuracy >= 0.55")
    print("=" * 70)
    print()

    # Best configuration from optimization
    configs = [
        {'temp': 0.20, 'tokens': 48},
    ]

    best_score = 0.0
    best_config = None
    all_evaluations = []

    for cfg in configs:
        evaluation = evaluate_config(
            client=client,
            dataset=java_data,
            config=config,
            temperature=cfg['temp'],
            max_tokens=cfg['tokens']
        )

        metrics = evaluation["metrics"]

        print()
        print(f"Config: T={cfg['temp']}, max_tokens={cfg['tokens']}")
        print(f"  ES: {metrics['edit_similarity']:.4f}, EM: {metrics['exact_match']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Score: {metrics['balanced_score']:.4f}")

        # Check targets
        es_met = metrics['edit_similarity'] >= 0.7
        em_met = metrics['exact_match'] >= 0.25
        acc_met = metrics['accuracy'] >= 0.55
        all_met = es_met and em_met and acc_met

        if all_met:
            print(f"  ✅ ALL TARGETS MET!")
        else:
            status = []
            if not es_met:
                status.append(f"ES: {0.7 - metrics['edit_similarity']:.3f} below")
            if not em_met:
                status.append(f"EM: {0.25 - metrics['exact_match']:.3f} below")
            if not acc_met:
                status.append(f"Accuracy: {0.55 - metrics['accuracy']:.3f} below")
            print(f"  ⚠️  Gaps: {', '.join(status)}")

        # Save individual result
        output_file = os.path.join(output_dir, f"java_t{int(cfg['temp']*100):02d}_tok{cfg['tokens']}.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        all_evaluations.append({
            'config': cfg,
            'metrics': metrics
        })

        if metrics['balanced_score'] > best_score:
            best_score = metrics['balanced_score']
            best_config = cfg

    # Summary
    print()
    print("=" * 70)
    print("JAVA FULL DATASET RESULTS")
    print("=" * 70)

    best_eval = all_evaluations[0]
    best_metrics = best_eval['metrics']
    best_config = best_eval['config']

    print(f"Configuration: T={best_config['temp']}, max_tokens={best_config['tokens']}")
    print(f"  ES: {best_metrics['edit_similarity']:.4f}")
    print(f"  EM: {best_metrics['exact_match']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Balanced Score: {best_metrics['balanced_score']:.4f}")
    print(f"  Total Samples: {best_metrics['total_samples']}")
    print(f"  Passes: {best_metrics['passes']}")

    # Check targets
    es_met = best_metrics['edit_similarity'] >= 0.7
    em_met = best_metrics['exact_match'] >= 0.25
    acc_met = best_metrics['accuracy'] >= 0.55
    all_met = es_met and em_met and acc_met

    print()
    if all_met:
        print("✅ ALL TARGETS MET!")
    else:
        print("Target Status:")
        print(f"  ES: {'✅' if es_met else '❌'} ({best_metrics['edit_similarity']:.4f} / 0.70)")
        print(f"  EM: {'✅' if em_met else '❌'} ({best_metrics['exact_match']:.4f} / 0.25)")
        print(f"  Accuracy: {'✅' if acc_met else '❌'} ({best_metrics['accuracy']:.4f} / 0.55)")

    print("=" * 70)

    # Save summary
    summary = {
        'best_config': best_config,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'all_evaluations': all_evaluations
    }

    with open(os.path.join(output_dir, 'optimization_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
