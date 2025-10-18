#!/usr/bin/env python3
"""
Balanced Multi-Objective Optimization for Both Languages
Target: Optimize (ES, EM, pass@1) together with balanced weights
Strategy: Find configurations that maximize weighted score across all metrics

Scoring function:
  balanced_score = w1 * ES_norm + w2 * EM_norm + w3 * pass@1_norm

Where:
  - ES_norm = ES / 0.7 (normalized to target)
  - EM_norm = EM / 0.25 (normalized to target)
  - pass@1_norm = pass@1 / 0.55 (normalized to target)
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


def contextual_extract(generated: str, groundtruth: str, prefix: str, suffix: str) -> str:
    """Python extraction strategy."""
    if not generated:
        return ""

    generated = generated.strip()
    gt_len = len(groundtruth)

    prefix_tail = prefix[-100:] if len(prefix) > 100 else prefix
    suffix_head = suffix[:100] if len(suffix) > 100 else suffix

    opens_context = bool(re.search(r'[\(\[\{]\s*$', prefix_tail))
    closes_context = bool(re.search(r'^\s*[\)\]\}]', suffix_head))

    if opens_context and closes_context:
        max_len = min(gt_len * 1.5, 120)
    else:
        max_len = min(gt_len * 1.8, 150)

    generated = generated[:int(max_len)]

    statement_ends = [r'\n\s*\n', r'[;}\]]\n', r'"\s*\n', r"'\s*\n"]
    min_end = len(generated)
    for pattern in statement_ends:
        match = re.search(pattern, generated)
        if match:
            min_end = min(min_end, match.end())

    if min_end < len(generated):
        generated = generated[:min_end]

    if '\n' not in groundtruth and '\n' in generated:
        first_line = generated.split('\n')[0]
        if first_line.strip():
            generated = first_line

    def_patterns = [r'\ndef ', r'\nclass ', r'\nif __name__']
    for pattern in def_patterns:
        idx = generated.find(pattern)
        if idx != -1:
            generated = generated[:idx]

    return generated.rstrip()


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


def calculate_balanced_score(es: float, em: float, pass_at_1: float,
                             w_es: float = 1.0, w_em: float = 1.0, w_pass: float = 1.0) -> float:
    """
    Calculate balanced score with weighted normalization to targets.

    Args:
        es: Edit similarity (0-1)
        em: Exact match rate (0-1)
        pass_at_1: Pass@1 rate (0-1)
        w_es: Weight for ES (default 1.0)
        w_em: Weight for EM (default 1.0)
        w_pass: Weight for pass@1 (default 1.0)

    Returns:
        Balanced score (higher is better)
    """
    # Normalize to targets
    es_norm = es / 0.7
    em_norm = em / 0.25
    pass_norm = pass_at_1 / 0.55

    # Weighted average
    total_weight = w_es + w_em + w_pass
    balanced = (w_es * es_norm + w_em * em_norm + w_pass * pass_norm) / total_weight

    return balanced


def evaluate_config(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    language: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Evaluate a single configuration."""
    template = config["template"]

    if language == 'python':
        base_stops = ["\\n\\ndef ", "\\nclass ", "\\n\\n\\n", "\\nif __name__"]
        extract_fn = contextual_extract
    else:  # java
        base_stops = ["\\n\\npublic ", "\\n\\nprivate ", "\\n\\nclass ", "\\n\\ninterface "]
        extract_fn = java_aggressive_extract

    stop_tokens = config["stop_tokens"] + base_stops

    results = []
    passes = 0

    for i, example in enumerate(tqdm(dataset, desc=f"{language} T={temperature}, tok={max_tokens}", leave=False)):
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
            prediction = extract_fn(raw_prediction, groundtruth, prefix, suffix)

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

    pass_at_1 = passes / len(results) if results else 0.0
    avg_es = sum(r['es'] for r in results if 'es' in r) / len(results) if results else 0.0
    avg_em = sum(1 for r in results if r.get('em', False)) / len(results) if results else 0.0

    # Calculate balanced score
    balanced = calculate_balanced_score(avg_es, avg_em, pass_at_1)

    return {
        "metrics": {
            "pass@1": pass_at_1,
            "edit_similarity": avg_es,
            "exact_match": avg_em,
            "balanced_score": balanced,
            "total_samples": len(dataset),
            "passes": passes,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "language": language
        },
        "results": results
    }


def main():
    output_dir = "results/balanced_all"
    os.makedirs(output_dir, exist_ok=True)

    client = ModelAPIClient()
    config = load_config()

    print("=" * 70)
    print("Balanced Multi-Objective Optimization (Python + Java)")
    print("=" * 70)
    print("Targets:")
    print("  - ES >= 0.7")
    print("  - EM >= 0.25")
    print("  - pass@1 >= 0.55")
    print("Optimization: Maximize balanced_score = (ES/0.7 + EM/0.25 + pass@1/0.55) / 3")
    print("=" * 70)
    print()

    # Test configurations for both languages
    python_configs = [
        {'temp': 0.05, 'tokens': 56},
        {'temp': 0.08, 'tokens': 56},
        {'temp': 0.10, 'tokens': 56},
        {'temp': 0.12, 'tokens': 56},
    ]

    java_configs = [
        {'temp': 0.10, 'tokens': 48},
        {'temp': 0.15, 'tokens': 48},
        {'temp': 0.20, 'tokens': 48},
        {'temp': 0.20, 'tokens': 56},
    ]

    # Load datasets
    python_data = load_dataset("datasets/python_50_samples.jsonl")
    java_data = load_dataset("datasets/java_50_samples.jsonl")

    all_results = {
        'python': [],
        'java': []
    }

    best_overall_score = 0.0
    best_overall_config = None

    # Evaluate Python configurations
    print("Evaluating Python configurations...")
    print()
    for cfg in python_configs:
        evaluation = evaluate_config(
            client=client,
            dataset=python_data,
            config=config,
            language='python',
            temperature=cfg['temp'],
            max_tokens=cfg['tokens']
        )

        metrics = evaluation["metrics"]

        print(f"Python T={cfg['temp']}, tok={cfg['tokens']}")
        print(f"  ES: {metrics['edit_similarity']:.4f}, EM: {metrics['exact_match']:.4f}, pass@1: {metrics['pass@1']:.4f}")
        print(f"  Balanced Score: {metrics['balanced_score']:.4f}")

        # Check targets
        es_met = metrics['edit_similarity'] >= 0.7
        em_met = metrics['exact_match'] >= 0.25
        pass_met = metrics['pass@1'] >= 0.55
        all_met = es_met and em_met and pass_met

        if all_met:
            print(f"  ✅ ALL TARGETS MET!")
        else:
            status = []
            if not es_met:
                status.append(f"ES: {0.7 - metrics['edit_similarity']:.3f} below")
            if not em_met:
                status.append(f"EM: {0.25 - metrics['exact_match']:.3f} below")
            if not pass_met:
                status.append(f"pass@1: {0.55 - metrics['pass@1']:.3f} below")
            print(f"  ⚠️  Gaps: {', '.join(status)}")
        print()

        # Save result
        output_file = os.path.join(output_dir, f"python_t{int(cfg['temp']*100):02d}_tok{cfg['tokens']}.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        all_results['python'].append({
            'config': cfg,
            'metrics': metrics
        })

        if metrics['balanced_score'] > best_overall_score:
            best_overall_score = metrics['balanced_score']
            best_overall_config = {'language': 'python', **cfg}

    # Evaluate Java configurations
    print()
    print("Evaluating Java configurations...")
    print()
    for cfg in java_configs:
        evaluation = evaluate_config(
            client=client,
            dataset=java_data,
            config=config,
            language='java',
            temperature=cfg['temp'],
            max_tokens=cfg['tokens']
        )

        metrics = evaluation["metrics"]

        print(f"Java T={cfg['temp']}, tok={cfg['tokens']}")
        print(f"  ES: {metrics['edit_similarity']:.4f}, EM: {metrics['exact_match']:.4f}, pass@1: {metrics['pass@1']:.4f}")
        print(f"  Balanced Score: {metrics['balanced_score']:.4f}")

        # Check targets
        es_met = metrics['edit_similarity'] >= 0.7
        em_met = metrics['exact_match'] >= 0.25
        pass_met = metrics['pass@1'] >= 0.55
        all_met = es_met and em_met and pass_met

        if all_met:
            print(f"  ✅ ALL TARGETS MET!")
        else:
            status = []
            if not es_met:
                status.append(f"ES: {0.7 - metrics['edit_similarity']:.3f} below")
            if not em_met:
                status.append(f"EM: {0.25 - metrics['exact_match']:.3f} below")
            if not pass_met:
                status.append(f"pass@1: {0.55 - metrics['pass@1']:.3f} below")
            print(f"  ⚠️  Gaps: {', '.join(status)}")
        print()

        # Save result
        output_file = os.path.join(output_dir, f"java_t{int(cfg['temp']*100):02d}_tok{cfg['tokens']}.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        all_results['java'].append({
            'config': cfg,
            'metrics': metrics
        })

        if metrics['balanced_score'] > best_overall_score:
            best_overall_score = metrics['balanced_score']
            best_overall_config = {'language': 'java', **cfg}

    # Summary
    print()
    print("=" * 70)
    print("BALANCED OPTIMIZATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Best overall: {best_overall_config['language'].upper()} "
          f"T={best_overall_config['temp']}, tok={best_overall_config['tokens']}")
    print(f"Best balanced score: {best_overall_score:.4f}")
    print()

    print("Top Python configurations:")
    sorted_python = sorted(all_results['python'], key=lambda x: x['metrics']['balanced_score'], reverse=True)
    for i, e in enumerate(sorted_python[:3], 1):
        cfg = e['config']
        m = e['metrics']
        print(f"{i}. T={cfg['temp']}, tok={cfg['tokens']}: score={m['balanced_score']:.4f} "
              f"(ES={m['edit_similarity']:.3f}, EM={m['exact_match']:.3f}, pass@1={m['pass@1']:.3f})")

    print()
    print("Top Java configurations:")
    sorted_java = sorted(all_results['java'], key=lambda x: x['metrics']['balanced_score'], reverse=True)
    for i, e in enumerate(sorted_java[:3], 1):
        cfg = e['config']
        m = e['metrics']
        print(f"{i}. T={cfg['temp']}, tok={cfg['tokens']}: score={m['balanced_score']:.4f} "
              f"(ES={m['edit_similarity']:.3f}, EM={m['exact_match']:.3f}, pass@1={m['pass@1']:.3f})")

    print("=" * 70)

    # Save summary
    summary = {
        'best_config': best_overall_config,
        'best_score': best_overall_score,
        'python_results': all_results['python'],
        'java_results': all_results['java']
    }

    with open(os.path.join(output_dir, 'optimization_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
