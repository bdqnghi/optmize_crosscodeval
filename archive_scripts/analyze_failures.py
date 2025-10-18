#!/usr/bin/env python3
"""
Analyze pass@1 failure patterns to identify improvement opportunities.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(result_file):
    """Load evaluation results."""
    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_failures(results, pass_threshold=0.7):
    """Analyze why samples fail the pass criteria."""

    analysis = {
        'total_samples': 0,
        'passed': 0,
        'failed': 0,
        'near_miss_es': 0,  # ES in [0.65, 0.70)
        'low_es': 0,  # ES < 0.5
        'em_only_fail': 0,  # ES >= 0.7 but EM = False
        'es_only_fail': 0,  # EM = True but ES < 0.7
        'both_fail': 0,  # Both ES < 0.7 AND EM = False
        'es_distribution': [],
        'em_distribution': [],
        'failed_samples': [],
        'length_analysis': {
            'gt_too_short': [],  # groundtruth < 30 chars
            'gt_medium': [],  # 30 <= groundtruth < 100
            'gt_long': []  # groundtruth >= 100
        }
    }

    for item in results['results']:
        analysis['total_samples'] += 1

        es = item['es']
        em = item['em']
        gt = item['groundtruth']
        pred = item['prediction']
        gt_len = len(gt)

        analysis['es_distribution'].append(es)
        analysis['em_distribution'].append(1.0 if em else 0.0)

        # Classify by pass criteria
        passed = (es >= pass_threshold and em)

        if passed:
            analysis['passed'] += 1
        else:
            analysis['failed'] += 1

            # Categorize failure type
            if es >= pass_threshold and not em:
                analysis['em_only_fail'] += 1
                failure_type = 'EM_FAIL'
            elif es < pass_threshold and em:
                analysis['es_only_fail'] += 1
                failure_type = 'ES_FAIL'
            else:
                analysis['both_fail'] += 1
                failure_type = 'BOTH_FAIL'

            # Near miss detection
            if 0.65 <= es < pass_threshold:
                analysis['near_miss_es'] += 1
                failure_type += '_NEAR_MISS'
            elif es < 0.5:
                analysis['low_es'] += 1
                failure_type += '_LOW_ES'

            # Store failed sample info
            analysis['failed_samples'].append({
                'es': es,
                'em': em,
                'failure_type': failure_type,
                'gt_length': gt_len,
                'pred_length': len(pred),
                'groundtruth': gt,
                'prediction': pred
            })

            # Length-based analysis
            if gt_len < 30:
                analysis['length_analysis']['gt_too_short'].append(es)
            elif gt_len < 100:
                analysis['length_analysis']['gt_medium'].append(es)
            else:
                analysis['length_analysis']['gt_long'].append(es)

    # Calculate statistics
    analysis['pass_rate'] = analysis['passed'] / analysis['total_samples']
    analysis['near_miss_rate'] = analysis['near_miss_es'] / analysis['failed']
    analysis['em_only_fail_rate'] = analysis['em_only_fail'] / analysis['failed']

    # ES statistics by length
    for key in ['gt_too_short', 'gt_medium', 'gt_long']:
        samples = analysis['length_analysis'][key]
        if samples:
            analysis['length_analysis'][f'{key}_avg_es'] = np.mean(samples)
            analysis['length_analysis'][f'{key}_count'] = len(samples)

    return analysis


def print_analysis(analysis, language):
    """Print detailed analysis report."""
    print("=" * 70)
    print(f"{language.upper()} FAILURE PATTERN ANALYSIS")
    print("=" * 70)
    print()

    print(f"Total samples: {analysis['total_samples']}")
    print(f"Passed: {analysis['passed']} ({analysis['pass_rate']:.1%})")
    print(f"Failed: {analysis['failed']} ({1-analysis['pass_rate']:.1%})")
    print()

    print("Failure Breakdown:")
    print(f"  Near-miss (ES 0.65-0.69): {analysis['near_miss_es']} ({analysis['near_miss_es']/analysis['failed']:.1%} of failures)")
    print(f"  Low ES (<0.5): {analysis['low_es']} ({analysis['low_es']/analysis['failed']:.1%} of failures)")
    print(f"  EM-only fail (ES>=0.7, EM=False): {analysis['em_only_fail']} ({analysis['em_only_fail_rate']:.1%} of failures)")
    print(f"  ES-only fail (ES<0.7, EM=True): {analysis['es_only_fail']} ({analysis['es_only_fail']/analysis['failed']:.1%} of failures)")
    print(f"  Both fail: {analysis['both_fail']} ({analysis['both_fail']/analysis['failed']:.1%} of failures)")
    print()

    print("Length-based Performance:")
    for key, label in [('gt_too_short', 'Short (<30 chars)'),
                       ('gt_medium', 'Medium (30-100 chars)'),
                       ('gt_long', 'Long (>=100 chars)')]:
        avg_key = f'{key}_avg_es'
        count_key = f'{key}_count'
        if avg_key in analysis['length_analysis']:
            avg_es = analysis['length_analysis'][avg_key]
            count = analysis['length_analysis'][count_key]
            print(f"  {label}: {count} samples, avg ES={avg_es:.3f}")
    print()

    # Top failure patterns
    print("Top 5 Worst Failures (lowest ES):")
    sorted_failures = sorted(analysis['failed_samples'], key=lambda x: x['es'])[:5]
    for i, failure in enumerate(sorted_failures, 1):
        print(f"\n{i}. ES={failure['es']:.3f}, EM={failure['em']}, Type={failure['failure_type']}")
        print(f"   GT length: {failure['gt_length']}, Pred length: {failure['pred_length']}")
        print(f"   Groundtruth: {failure['groundtruth'][:80]}...")
        print(f"   Prediction:  {failure['prediction'][:80]}...")
    print()

    # Improvement opportunities
    print("=" * 70)
    print("IMPROVEMENT OPPORTUNITIES:")
    print("=" * 70)

    if analysis['near_miss_es'] > analysis['failed'] * 0.2:
        print(f"1. 🎯 HIGH PRIORITY: {analysis['near_miss_es']} near-miss samples (ES 0.65-0.69)")
        print("   → Small extraction/generation tweaks could push these over 0.7")
        print("   → Potential gain: +{:.1%} pass rate".format(analysis['near_miss_es']/analysis['total_samples']))

    if analysis['em_only_fail'] > analysis['failed'] * 0.2:
        print(f"\n2. 🔧 WHITESPACE ISSUES: {analysis['em_only_fail']} samples with ES>=0.7 but EM=False")
        print("   → Normalization/formatting could fix these")
        print("   → Potential gain: +{:.1%} pass rate".format(analysis['em_only_fail']/analysis['total_samples']))

    if analysis['low_es'] > analysis['failed'] * 0.3:
        print(f"\n3. ⚠️  MODEL CAPABILITY: {analysis['low_es']} samples with ES<0.5")
        print("   → These are fundamental generation failures")
        print("   → May need model upgrade or few-shot examples")

    print()


def main():
    results_dir = Path('results/python')

    # Analyze Python results
    python_result_file = results_dir / 'python_best_strategy.json'
    if python_result_file.exists():
        print("Analyzing Python results...")
        results = load_results(python_result_file)
        analysis = analyze_failures(results)
        print_analysis(analysis, 'Python')

        # Save detailed analysis
        with open(results_dir / 'failure_analysis.json', 'w') as f:
            # Remove large sample data for saving
            analysis_copy = analysis.copy()
            analysis_copy['failed_samples'] = analysis_copy['failed_samples'][:10]  # Keep top 10
            json.dump(analysis_copy, f, indent=2)

    # Analyze Java results
    java_results_dir = Path('results/java')
    java_result_file = java_results_dir / 'java_best_strategy.json'
    if java_result_file.exists():
        print("\n")
        print("Analyzing Java results...")
        results = load_results(java_result_file)
        analysis = analyze_failures(results)
        print_analysis(analysis, 'Java')

        # Save detailed analysis
        with open(java_results_dir / 'failure_analysis.json', 'w') as f:
            analysis_copy = analysis.copy()
            analysis_copy['failed_samples'] = analysis_copy['failed_samples'][:10]
            json.dump(analysis_copy, f, indent=2)


if __name__ == '__main__':
    main()
