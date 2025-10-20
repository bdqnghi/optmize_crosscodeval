#!/usr/bin/env python3
"""
FIM-based Evaluation Script
For CrossCode Python and CrossCode Java datasets
"""

import argparse
import json
import os
from typing import Dict, List
from tqdm import tqdm

from api_client import ModelAPIClient
from eval_utils import (
    load_config,
    load_dataset,
    parse_fim_prompt,
    contextual_extract_python,
    contextual_extract_java,
    post_process_prediction,
    calculate_edit_similarity,
    calculate_exact_match,
    check_pass,
    calculate_metrics
)


def save_batch_results(
    batch_results: List[Dict],
    batch_errors: List[Dict],
    batch_num: int,
    output_dir: str,
    base_filename: str,
    language: str,
    temperature: float,
    max_tokens: int,
    post_process: bool,
    model_name: str
):
    """
    Save results and errors for a single batch

    Args:
        batch_results: List of successful results in this batch
        batch_errors: List of errors in this batch
        batch_num: Batch number (1-indexed)
        output_dir: Directory to save files
        base_filename: Base name for output files
        language: Programming language
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        post_process: Whether post-processing was applied
        model_name: Full model name
    """
    # Save batch results
    batch_metrics = calculate_metrics(batch_results)
    batch_metrics.update({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "language": language,
        "dataset": f"crosscode_{language}",
        "post_process": post_process,
        "model": model_name,
        "batch_number": batch_num,
        "batch_size": len(batch_results)
    })

    batch_data = {
        "metrics": batch_metrics,
        "results": batch_results
    }

    batch_file = os.path.join(output_dir, f"{base_filename}_batch_{batch_num}.json")
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    print(f"\n✓ Saved batch {batch_num} results to: {batch_file}")

    # Save batch errors if any
    if batch_errors:
        error_file = os.path.join(output_dir, f"{base_filename}_error_batch_{batch_num}.json")
        error_data = {
            "batch_number": batch_num,
            "error_count": len(batch_errors),
            "errors": batch_errors
        }
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        print(f"✓ Saved batch {batch_num} errors to: {error_file}")


def evaluate_fim_dataset(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    language: str,
    temperature: float,
    max_tokens: int,
    post_process: bool = False,
    output_dir: str = None,
    base_filename: str = None,
    model_name: str = None,
    batch_size: int = 100
) -> Dict:
    """
    Evaluate FIM-based dataset (CrossCode Python or Java) with batch logging

    Args:
        client: API client for model inference
        dataset: List of evaluation samples
        config: Model configuration
        language: Programming language ('python' or 'java')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        post_process: Whether to apply post-processing
        output_dir: Directory to save batch results
        base_filename: Base filename for batch files
        model_name: Full model name
        batch_size: Number of records per batch (default: 100)

    Returns:
        Dictionary with metrics and results
    """
    template = config["template"]

    # Language-specific stop tokens and extraction function
    if language == "python":
        base_stops = ["\\n\\ndef ", "\\nclass ", "\\n\\n\\n", "\\nif __name__"]
        extract_fn = contextual_extract_python
    elif language == "java":
        base_stops = ["\\n\\npublic ", "\\n\\nprivate ", "\\n\\nclass ", "\\n\\ninterface "]
        extract_fn = contextual_extract_java
    else:
        raise ValueError(f"Unknown language: {language}")

    stop_tokens = config["stop_tokens"] + base_stops

    all_results = []
    all_errors = []
    batch_results = []
    batch_errors = []
    passes = 0
    current_batch = 1

    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, example in enumerate(tqdm(dataset, desc=f"{language.upper()} T={temperature}, tok={max_tokens}")):
        try:
            prompt = example["prompt"]
            groundtruth = example["groundtruth"]
            prefix, suffix = parse_fim_prompt(prompt)

            # Generate completion
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

            # Extract relevant code
            prediction = extract_fn(raw_prediction, groundtruth, prefix, suffix)

            # Optional post-processing (without using ground truth)
            if post_process:
                prediction = post_process_prediction(prediction, language)

            # Calculate metrics
            es = calculate_edit_similarity(prediction, groundtruth)
            em = calculate_exact_match(prediction, groundtruth)
            passed = check_pass(es, em)

            if passed:
                passes += 1

            result = {
                "index": i,
                "groundtruth": groundtruth,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "es": es,
                "em": em,
                "passed": passed
            }

            batch_results.append(result)
            all_results.append(result)

        except Exception as e:
            error_msg = f"Error at index {i}: {e}"
            print(f"\n{error_msg}")

            error_record = {
                "index": i,
                "error": str(e),
                "error_type": type(e).__name__,
                "prompt": example.get("prompt", "")[:200] + "..." if len(example.get("prompt", "")) > 200 else example.get("prompt", "")
            }

            batch_errors.append(error_record)
            all_errors.append(error_record)

            # Still add to results with error info
            result = {
                "index": i,
                "error": str(e),
                "es": 0.0,
                "em": False,
                "passed": False
            }
            batch_results.append(result)
            all_results.append(result)

        # Save batch when reaching batch_size
        if (i + 1) % batch_size == 0:
            if output_dir and base_filename:
                save_batch_results(
                    batch_results, batch_errors, current_batch,
                    output_dir, base_filename, language,
                    temperature, max_tokens, post_process, model_name
                )

            # Reset batch
            batch_results = []
            batch_errors = []
            current_batch += 1

    # Save remaining records in final batch
    if batch_results and output_dir and base_filename:
        save_batch_results(
            batch_results, batch_errors, current_batch,
            output_dir, base_filename, language,
            temperature, max_tokens, post_process, model_name
        )

    # Calculate overall metrics
    metrics = calculate_metrics(all_results)
    metrics.update({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "language": language,
        "dataset": f"crosscode_{language}",
        "post_process": post_process,
        "total_batches": current_batch,
        "total_errors": len(all_errors)
    })

    return {
        "metrics": metrics,
        "results": all_results,
        "errors": all_errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="FIM-based evaluation for CrossCode datasets"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["crosscode_python", "crosscode_java"],
        help="Dataset to evaluate"
    )

    parser.add_argument(
        "--dataset-path",
        help="Path to dataset file (auto-detected if not provided)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        default="3b",
        help="Model to use (3b, 7b, 14b, etc.) - default: 3b"
    )

    # Model parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )

    # Post-processing
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Apply post-processing to predictions"
    )

    # Batch processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records per batch (default: 100)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        help="Output directory (auto: results/crosscode/{language}/{model}/)"
    )

    parser.add_argument(
        "--output-name",
        help="Output filename (auto-generated if not provided)"
    )

    args = parser.parse_args()

    # Extract language from dataset name
    language = args.dataset.replace("crosscode_", "")

    # Auto-detect dataset path
    if not args.dataset_path:
        args.dataset_path = f"datasets/{args.dataset}.jsonl"

    # Load configuration first to get model name
    config = load_config()

    # Get full model name from config
    if args.model in config.get("models", {}):
        model_full_name = config["models"][args.model]["name"]
    else:
        # If not in config, use the provided model string
        model_full_name = args.model

    # Auto-detect output directory with model subfolder
    if not args.output_dir:
        args.output_dir = f"results/crosscode/{language}/{model_full_name}"

    # Generate base filename for batch files
    temp_str = f"t{int(args.temperature * 100):02d}"
    tokens_str = f"tok{args.max_tokens}"
    pp_str = "_pp" if args.post_process else ""
    base_filename = f"{language}_{temp_str}_{tokens_str}{pp_str}"

    # Print configuration
    print("=" * 70)
    print(f"Evaluating {args.dataset.upper()}")
    print("=" * 70)
    print(f"Model: {model_full_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Language: {language.upper()}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Post-processing: {args.post_process}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)
    print()

    client = ModelAPIClient(model_size=args.model)
    dataset = load_dataset(args.dataset_path, args.max_samples)

    print(f"Loaded {len(dataset)} samples")
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    print(f"Will process in {total_batches} batches of {args.batch_size} records")
    print()

    # Run evaluation with batch logging
    evaluation = evaluate_fim_dataset(
        client=client,
        dataset=dataset,
        config=config,
        language=language,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        post_process=args.post_process,
        output_dir=args.output_dir,
        base_filename=base_filename,
        model_name=model_full_name,
        batch_size=args.batch_size
    )

    # Add model name to metrics
    evaluation["metrics"]["model"] = model_full_name

    # Print results
    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    metrics = evaluation["metrics"]
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Accuracy':<20} {metrics['accuracy']:>10.4f}")
    print(f"{'Edit Similarity':<20} {metrics['edit_similarity']:>10.4f}")
    print(f"{'Exact Match':<20} {metrics['exact_match']:>10.4f}")
    print(f"{'Passes':<20} {metrics['passes']:>10}")
    print(f"{'Total Samples':<20} {metrics['total_samples']:>10}")
    print(f"{'Total Batches':<20} {metrics.get('total_batches', 0):>10}")
    print(f"{'Total Errors':<20} {metrics.get('total_errors', 0):>10}")
    print("-" * 32)
    print(f"{'Temperature':<20} {metrics['temperature']:>10}")
    print(f"{'Max Tokens':<20} {metrics['max_tokens']:>10}")
    print(f"{'Post-process':<20} {str(metrics['post_process']):>10}")

    print("=" * 70)

    # Save overall results
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name:
        output_file = os.path.join(args.output_dir, args.output_name)
    else:
        # Auto-generate filename for overall results
        output_file = os.path.join(
            args.output_dir,
            f"{base_filename}_overall.json"
        )

    with open(output_file, 'w') as f:
        json.dump(evaluation, f, indent=2)

    print(f"\nOverall results saved to: {output_file}")

    # Save overall errors if any
    if evaluation.get("errors"):
        error_file = os.path.join(args.output_dir, f"{base_filename}_all_errors.json")
        error_summary = {
            "total_errors": len(evaluation["errors"]),
            "errors": evaluation["errors"]
        }
        with open(error_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        print(f"Overall errors saved to: {error_file}")

    # Print batch files summary
    print()
    print("=" * 70)
    print("BATCH FILES CREATED")
    print("=" * 70)
    for batch_num in range(1, metrics.get('total_batches', 0) + 1):
        batch_file = f"{base_filename}_batch_{batch_num}.json"
        print(f"  Batch {batch_num}: {batch_file}")
        error_batch_file = os.path.join(args.output_dir, f"{base_filename}_error_batch_{batch_num}.json")
        if os.path.exists(error_batch_file):
            print(f"  Errors {batch_num}: {base_filename}_error_batch_{batch_num}.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
