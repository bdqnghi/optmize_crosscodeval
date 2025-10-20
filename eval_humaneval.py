#!/usr/bin/env python3
"""
HumanEval Completion Generation Script
Generates completions for HumanEval benchmark
Use eval_humaneval.sh for full evaluation pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Install with: pip install datasets")
    sys.exit(1)

from api_client import ModelAPIClient
from eval_utils import load_config


def generate_completions(
    client: ModelAPIClient,
    problems: List[Dict],
    config: Dict,
    temperature: float,
    max_tokens: int,
    num_samples: int = 1,
    use_chat_format: bool = False,
    top_p: float = None
) -> List[Dict]:
    """Generate completions for HumanEval problems following evalplus structure"""

    # Base stop tokens from evalplus
    base_stops = ["<|endoftext|>", "</s>", "\nif __name__", "\ndef main(", "\nprint("]
    # HumanEval-specific stops (prevent generating multiple functions)
    humaneval_stops = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    stop_tokens = base_stops + humaneval_stops

    completions = []

    for problem in tqdm(problems, desc="Generating completions"):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        for sample_idx in range(num_samples):
            try:
                if use_chat_format:
                    # Chat-based format (instruction-tuned models)
                    instruction = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
                    user_message = f"{instruction}\n```python\n{prompt.strip()}\n```"

                    # Pre-fill assistant response to guide format
                    # Note: We can't actually pre-fill with our API, so we just prompt properly
                    final_prompt = user_message
                else:
                    # Direct completion (base models) - just use raw prompt
                    final_prompt = prompt.strip()

                # Generate completion
                outputs = client.generate(
                    prompt=final_prompt,
                    stop=stop_tokens,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    top_p=top_p
                )

                raw_completion = outputs[0]

                # Extract code from markdown if present (common with chat models)
                completion = extract_code_from_markdown(raw_completion)

                # For direct completion models, the output is just the function body
                # For chat models, it might include the full function
                # We save the raw completion as-is for now
                completions.append({
                    "task_id": task_id,
                    "completion": completion
                })

            except Exception as e:
                print(f"\nError on {task_id} (sample {sample_idx}): {e}")
                completions.append({
                    "task_id": task_id,
                    "completion": ""
                })

    return completions


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks"""
    # Remove markdown code block markers
    text = text.strip()

    # Pattern 1: ```python ... ```
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    # Pattern 2: ``` ... ```
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate HumanEval completions"
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
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per problem for pass@k (default: 1)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter (default: None)"
    )

    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Force direct completion format (disable chat format even for instruct models)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        help="Output directory (auto: results/humaneval/{model}/)"
    )

    parser.add_argument(
        "--output-name",
        help="Output filename (auto-generated if not provided)"
    )

    args = parser.parse_args()

    # Load configuration first to get model name
    config = load_config()

    # Get full model name and detect if instruct model
    if args.model in config.get("models", {}):
        model_config = config["models"][args.model]
        model_full_name = model_config["name"]
        is_instruct = model_config.get("is_instruct", False)
    else:
        model_full_name = args.model
        is_instruct = "instruct" in args.model.lower()

    # Auto-detect output directory with model subfolder
    if not args.output_dir:
        args.output_dir = f"results/humaneval/{model_full_name}"

    # Determine use_chat_format based on model type and user override
    # For instruct models: default to chat format unless --no-context is specified
    # For base models: default to direct completion unless user forces chat format
    if args.no_context:
        use_chat_format = False
    else:
        use_chat_format = is_instruct

    # Print configuration
    print("=" * 70)
    print("HumanEval Completion Generation")
    print("=" * 70)
    print(f"Model: {model_full_name}")
    print(f"Model Type: {'Instruct' if is_instruct else 'Base'}")
    print(f"Prompt Format: {'Chat' if use_chat_format else 'Direct'}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p if args.top_p is not None else 'None'}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)
    print()

    # Load HumanEval dataset
    print("Loading HumanEval dataset from Hugging Face...")
    try:
        ds = load_dataset("openai/openai_humaneval")
        problems = ds["test"]
        print(f"Loaded {len(problems)} problems")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure you have internet connection and datasets library installed:")
        print("  pip install datasets")
        sys.exit(1)

    print()

    # Initialize client
    client = ModelAPIClient(model_size=args.model)

    # Generate completions
    completions = generate_completions(
        client=client,
        problems=problems,
        config=config,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        use_chat_format=use_chat_format,
        top_p=args.top_p
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename
    if args.output_name:
        output_file = Path(args.output_dir) / args.output_name
    else:
        temp_str = f"t{int(args.temperature * 100):02d}"
        samples_str = f"n{args.num_samples}"
        output_file = Path(args.output_dir) / f"samples_{temp_str}_{samples_str}.jsonl"

    # Save completions
    print(f"\nSaving {len(completions)} completions to {output_file}")
    with open(output_file, 'w') as f:
        for completion in completions:
            f.write(json.dumps(completion) + '\n')

    print()
    print("=" * 70)
    print("✅ Completion generation finished!")
    print("=" * 70)
    print(f"Completions saved to: {output_file}")
    print()
    print("To evaluate, use the shell script:")
    print(f"  EVAL_ONLY=\"{output_file}\" ./eval_humaneval.sh")
    print()


if __name__ == "__main__":
    main()
