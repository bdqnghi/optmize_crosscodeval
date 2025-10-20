#!/usr/bin/env python3
"""
HumanEval Results Evaluation Script
Reads completion files and evaluates them to calculate pass@1
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class TimeoutError(Exception):
    """Raised when code execution times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Code execution timed out")


def evaluate_completions(samples: List[Dict], verbose: bool = True, timeout: int = 5) -> Dict:
    """
    Evaluate completions using HumanEval+ test cases

    Args:
        samples: List of completion samples with task_id and completion
        verbose: Whether to print detailed progress
        timeout: Maximum seconds per test execution (default: 5)

    Returns:
        Dictionary with evaluation results and metrics
    """
    if verbose:
        print("\nEvaluating completions...")

    # Import evalplus components
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError:
        print("Error: evalplus library not found. Install with: pip install evalplus")
        sys.exit(1)

    # Get test cases
    problems = get_human_eval_plus()

    # Evaluate
    results = {}
    passed = 0
    failed = 0
    errors_detail = []
    total = len(samples)

    iterator = tqdm(samples, desc="Evaluating") if verbose else samples

    for idx, sample in enumerate(iterator):
        task_id = sample.get("task_id")
        completion = sample.get("completion", "")

        if not task_id:
            print(f"\nWarning: Sample {idx} missing task_id")
            continue

        if task_id not in problems:
            print(f"\nWarning: {task_id} not found in HumanEval+ problems")
            continue

        problem = problems[task_id]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        # Construct full code
        full_code = prompt + completion

        # Execute test with timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(test, exec_globals)
            exec(f"check({entry_point})", exec_globals)

            results[task_id] = {
                "passed": True,
                "completion": completion
            }
            passed += 1

        except TimeoutError as e:
            error_msg = f"Timeout after {timeout}s (infinite loop or slow code)"
            results[task_id] = {
                "passed": False,
                "error": error_msg,
                "completion": completion
            }
            failed += 1

            errors_detail.append({
                "task_id": task_id,
                "error": error_msg,
                "error_type": "TimeoutError"
            })

        except Exception as e:
            error_msg = str(e)[:200]
            results[task_id] = {
                "passed": False,
                "error": error_msg,
                "completion": completion
            }
            failed += 1

            errors_detail.append({
                "task_id": task_id,
                "error": error_msg,
                "error_type": type(e).__name__
            })

        finally:
            # Cancel the alarm
            signal.alarm(0)

    # Calculate pass@1
    pass_at_1 = passed / total if total > 0 else 0.0

    return {
        "eval": {
            "pass@1": pass_at_1,
            "passed": passed,
            "failed": failed,
            "total": total
        },
        "results": results,
        "errors": errors_detail
    }


def load_completions_from_jsonl(file_path: Path) -> List[Dict]:
    """Load completions from JSONL file (one JSON object per line)"""
    completions = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                completions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    return completions


def load_completions_from_json(file_path: Path) -> List[Dict]:
    """Load completions from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Try to find completions in common keys
        if "results" in data:
            return data["results"]
        elif "completions" in data:
            return data["completions"]
        elif "samples" in data:
            return data["samples"]
        else:
            print("Error: Could not find completions in JSON structure")
            print("Expected keys: 'results', 'completions', or 'samples'")
            sys.exit(1)
    else:
        print("Error: Unexpected JSON structure")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HumanEval completion results and calculate pass@1"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file containing completions (.jsonl or .json)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save evaluation results (default: adds '_eval_results.json' suffix)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show detailed error information"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for each test execution (default: 5)"
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Print header
    if not args.quiet:
        print("=" * 70)
        print("HumanEval Results Evaluation")
        print("=" * 70)
        print(f"Input file: {input_file}")
        print("=" * 70)
        print()

    # Load completions
    if not args.quiet:
        print(f"Loading completions from {input_file}...")

    if input_file.suffix == ".jsonl":
        completions = load_completions_from_jsonl(input_file)
    elif input_file.suffix == ".json":
        completions = load_completions_from_json(input_file)
    else:
        print(f"Warning: Unknown file extension '{input_file.suffix}', attempting to parse as JSON...")
        try:
            completions = load_completions_from_json(input_file)
        except:
            print("Failed to parse as JSON, trying JSONL format...")
            completions = load_completions_from_jsonl(input_file)

    if not args.quiet:
        print(f"Loaded {len(completions)} completions")
        print()

    # Evaluate
    results = evaluate_completions(completions, verbose=not args.quiet, timeout=args.timeout)

    # Print results
    if not args.quiet:
        print()
        print("=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)

    eval_data = results["eval"]
    print(f"pass@1:  {eval_data['pass@1']:.6f} ({eval_data['pass@1']*100:.2f}%)")
    print(f"Passed:  {eval_data['passed']}/{eval_data['total']}")
    print(f"Failed:  {eval_data['failed']}/{eval_data['total']}")

    if not args.quiet:
        print("=" * 70)

    # Show errors if requested
    if args.show_errors and results.get("errors"):
        print()
        print("=" * 70)
        print(f"ERROR DETAILS ({len(results['errors'])} errors)")
        print("=" * 70)
        for error in results["errors"][:10]:  # Show first 10 errors
            print(f"\n{error['task_id']}:")
            print(f"  Type: {error['error_type']}")
            print(f"  Message: {error['error']}")

        if len(results["errors"]) > 10:
            print(f"\n... and {len(results['errors']) - 10} more errors")
        print("=" * 70)

    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Auto-generate output filename
        output_file = input_file.parent / f"{input_file.stem}_eval_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if not args.quiet:
        print(f"\nResults saved to: {output_file}")
        print()


if __name__ == "__main__":
    main()
