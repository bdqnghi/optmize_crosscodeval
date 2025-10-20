#!/usr/bin/env python3
"""
Extract Pass@k Metrics from Evalplus Results
Parses the evalplus evaluation JSON and extracts pass@k metrics into a simple summary file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def calculate_pass_at_k(results: Dict) -> Dict:
    """
    Calculate pass@k metrics from evalplus results

    Args:
        results: Evalplus evaluation results dictionary

    Returns:
        Dictionary with pass@k metrics
    """
    eval_data = results.get("eval", {})

    # Count passes and total
    base_pass = 0
    plus_pass = 0
    total = len(eval_data)

    for task_id, completions in eval_data.items():
        # For pass@1, we just check if the first completion passes
        if completions and len(completions) > 0:
            first_completion = completions[0]
            if first_completion.get("base_status") == "pass":
                base_pass += 1
            if first_completion.get("plus_status") == "pass":
                plus_pass += 1

    # Calculate percentages
    base_pass_rate = (base_pass / total * 100) if total > 0 else 0
    plus_pass_rate = (plus_pass / total * 100) if total > 0 else 0

    return {
        "base": {
            "pass": base_pass,
            "total": total,
            "pass@1": round(base_pass_rate, 2)
        },
        "plus": {
            "pass": plus_pass,
            "total": total,
            "pass@1": round(plus_pass_rate, 2)
        }
    }


def extract_metadata(results: Dict) -> Dict:
    """Extract metadata from evalplus results"""
    return {
        "date": results.get("date", ""),
        "hash": results.get("hash", "")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract pass@k metrics from evalplus results"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Input evalplus results JSON file"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output summary JSON file (default: input_file with _summary.json suffix)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Replace _eval_results.json with _summary.json
        if input_path.name.endswith("_eval_results.json"):
            output_path = input_path.parent / input_path.name.replace("_eval_results.json", "_summary.json")
        else:
            output_path = input_path.parent / f"{input_path.stem}_summary.json"

    # Load evalplus results
    print(f"Loading evalplus results from: {input_path}")
    with open(input_path, 'r') as f:
        results = json.load(f)

    # Extract metrics
    print("Calculating pass@k metrics...")
    metrics = calculate_pass_at_k(results)
    metadata = extract_metadata(results)

    # Create summary
    summary = {
        "file": input_path.name,
        "date": metadata["date"],
        "hash": metadata["hash"],
        "metrics": metrics
    }

    # Save summary
    print(f"Saving summary to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary to console
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"File: {input_path.name}")
    print(f"Date: {metadata['date']}")
    print(f"Hash: {metadata['hash']}")
    print()
    print(f"{'Metric':<20} {'Base':>10} {'Plus':>10}")
    print("-" * 42)
    print(f"{'pass@1':<20} {metrics['base']['pass@1']:>9.2f}% {metrics['plus']['pass@1']:>9.2f}%")
    print(f"{'Passed':<20} {metrics['base']['pass']:>10} {metrics['plus']['pass']:>10}")
    print(f"{'Total':<20} {metrics['base']['total']:>10} {metrics['plus']['total']:>10}")
    print("=" * 70)
    print()
    print(f"✅ Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
