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


def evaluate_fim_dataset(
    client: ModelAPIClient,
    dataset: List[Dict],
    config: Dict,
    language: str,
    temperature: float,
    max_tokens: int,
    post_process: bool = False
) -> Dict:
    """
    Evaluate FIM-based dataset (CrossCode Python or Java)

    Args:
        client: API client for model inference
        dataset: List of evaluation samples
        config: Model configuration
        language: Programming language ('python' or 'java')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        post_process: Whether to apply post-processing

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

    results = []
    passes = 0

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

            results.append({
                "index": i,
                "groundtruth": groundtruth,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "es": es,
                "em": em,
                "passed": passed
            })

        except Exception as e:
            print(f"\nError at index {i}: {e}")
            results.append({
                "index": i,
                "error": str(e),
                "es": 0.0,
                "em": False,
                "passed": False
            })

    # Calculate overall metrics
    metrics = calculate_metrics(results)
    metrics.update({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "language": language,
        "dataset": f"crosscode_{language}",
        "post_process": post_process
    })

    return {
        "metrics": metrics,
        "results": results
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
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)
    print()

    client = ModelAPIClient()
    dataset = load_dataset(args.dataset_path, args.max_samples)

    print(f"Loaded {len(dataset)} samples")
    print()

    # Run evaluation
    evaluation = evaluate_fim_dataset(
        client, dataset, config, language,
        args.temperature, args.max_tokens, args.post_process
    )

    # Add model name to metrics
    evaluation["metrics"]["model"] = model_full_name

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    metrics = evaluation["metrics"]
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Accuracy':<20} {metrics['accuracy']:>10.4f}")
    print(f"{'Edit Similarity':<20} {metrics['edit_similarity']:>10.4f}")
    print(f"{'Exact Match':<20} {metrics['exact_match']:>10.4f}")
    print(f"{'Passes':<20} {metrics['passes']:>10}")
    print(f"{'Total Samples':<20} {metrics['total_samples']:>10}")
    print("-" * 32)
    print(f"{'Temperature':<20} {metrics['temperature']:>10}")
    print(f"{'Max Tokens':<20} {metrics['max_tokens']:>10}")
    print(f"{'Post-process':<20} {str(metrics['post_process']):>10}")

    print("=" * 70)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name:
        output_file = os.path.join(args.output_dir, args.output_name)
    else:
        # Auto-generate filename
        temp_str = f"t{int(args.temperature * 100):02d}"
        tokens_str = f"tok{args.max_tokens}"
        pp_str = "_pp" if args.post_process else ""
        output_file = os.path.join(
            args.output_dir,
            f"{language}_{temp_str}_{tokens_str}{pp_str}.json"
        )

    with open(output_file, 'w') as f:
        json.dump(evaluation, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
