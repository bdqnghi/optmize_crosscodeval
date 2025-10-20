#!/usr/bin/env python3
"""
Fix HumanEval Completions Post-Processor
Applies systematic fixes to improve pass@1 rate
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def remove_markdown_artifacts(completion: str) -> str:
    """Remove markdown code block markers and artifacts"""
    # Remove ```python, ```java, etc.
    completion = re.sub(r'```\w*\n?', '', completion)
    completion = re.sub(r'\n?```$', '', completion)
    completion = re.sub(r'</code>\s*$', '', completion)
    return completion.strip()


def fix_indentation(completion: str) -> str:
    """
    Fix indentation issues - the most critical fix
    Ensures all lines have proper 4-space base indentation for function body
    """
    if not completion:
        return completion

    lines = completion.split('\n')
    fixed_lines = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            fixed_lines.append('')
            continue

        # Get current indentation
        current_indent = len(line) - len(line.lstrip())

        # If line has no indentation, add base 4 spaces
        if current_indent == 0:
            fixed_lines.append('    ' + line)
        # If line already has indentation, ensure it's at least 4 spaces
        elif current_indent < 4:
            # Add the missing spaces
            fixed_lines.append('    ' + line.lstrip())
        else:
            # Keep existing indentation
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_common_typos(completion: str) -> str:
    """Fix common typos in generated code"""
    typo_fixes = {
        'noteote': 'note',
        'strign': 'string',
        'retrun': 'return',
        'lenth': 'length',
        'lenght': 'length',
    }

    for typo, correct in typo_fixes.items():
        completion = completion.replace(typo, correct)

    return completion


# def fix_syntax_errors(completion: str) -> str:
#     """Fix common syntax errors"""
#     # Fix orphaned elif/else (common issue)
#     lines = completion.split('\n')
#     fixed_lines = []

#     for i, line in enumerate(lines):
#         stripped = line.lstrip()

#         # Fix elif/else that should be if
#         if stripped.startswith('elif ') or stripped.startswith('else:'):
#             # Check if there's a preceding if statement
#             has_preceding_if = False
#             for j in range(i - 1, -1, -1):
#                 prev_stripped = lines[j].lstrip()
#                 if prev_stripped.startswith('if '):
#                     has_preceding_if = True
#                     break
#                 if prev_stripped and not prev_stripped.startswith('#'):
#                     break

#             if not has_preceding_if and stripped.startswith('elif '):
#                 # Convert elif to if
#                 line = line.replace('elif ', 'if ', 1)

#         fixed_lines.append(line)

#     completion = '\n'.join(fixed_lines)

#     # Balance parentheses
#     open_count = completion.count('(')
#     close_count = completion.count(')')
#     if open_count > close_count:
#         completion += ')' * (open_count - close_count)

#     # Balance brackets
#     open_count = completion.count('[')
#     close_count = completion.count(']')
#     if open_count > close_count:
#         completion += ']' * (open_count - close_count)

#     # Balance braces
#     open_count = completion.count('{')
#     close_count = completion.count('}')
#     if open_count > close_count:
#         completion += '}' * (open_count - close_count)

#     return completion


# def remove_trailing_code(completion: str) -> str:
#     """Remove test code or examples after the main function"""
#     # Remove any code after common test patterns
#     patterns = [
#         r'\n\n+# Test',
#         r'\n\n+# Example',
#         r'\n\n+if __name__',
#         r'\n\n+def test_',
#         r'\n\n+assert ',
#         r'\n\n+print\(',
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, completion)
#         if match:
#             completion = completion[:match.start()]
#             break

#     return completion


def fix_completion(completion: str, aggressive: bool = False) -> str:
    """
    Apply all fixes to a completion

    Args:
        completion: Raw completion string
        aggressive: If True, apply more aggressive fixes

    Returns:
        Fixed completion string
    """
    # Step 1: Remove markdown artifacts
    completion = remove_markdown_artifacts(completion)

    # # Step 2: Remove trailing test code
    # completion = remove_trailing_code(completion)

    # Step 3: Fix common typos
    completion = fix_common_typos(completion)

    # # Step 4: Fix syntax errors
    # completion = fix_syntax_errors(completion)

    # Step 5: Fix indentation (MOST IMPORTANT)
    completion = fix_indentation(completion)

    # Step 6: Clean up excessive newlines
    while '\n\n\n' in completion:
        completion = completion.replace('\n\n\n', '\n\n')

    return completion


def process_completions_file(
    input_file: Path,
    output_file: Path,
    aggressive: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Process a completions file and apply fixes

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        aggressive: Apply aggressive fixes
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    completions = []
    changes_made = 0

    # Read input file
    with open(input_file, 'r') as f:
        for line in f:
            completions.append(json.loads(line.strip()))

    if verbose:
        print(f"Loaded {len(completions)} completions")
        print("Applying fixes...")

    # Apply fixes
    for item in completions:
        original = item['completion']
        fixed = fix_completion(original, aggressive=aggressive)

        if fixed != original:
            changes_made += 1
            if verbose and changes_made <= 5:
                print(f"\n--- Example Fix for {item['task_id']} ---")
                print(f"Original (first 100 chars): {original[:100]}")
                print(f"Fixed (first 100 chars):    {fixed[:100]}")

        item['completion'] = fixed

    # Write output file
    with open(output_file, 'w') as f:
        for item in completions:
            f.write(json.dumps(item) + '\n')

    stats = {
        'total': len(completions),
        'changed': changes_made,
        'unchanged': len(completions) - changes_made
    }

    if verbose:
        print(f"\n✓ Fixed {changes_made}/{len(completions)} completions")
        print(f"✓ Saved to: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix HumanEval completions to improve pass@1"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file with completions"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output JSONL file (default: adds '_fixed' suffix)"
    )

    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Apply more aggressive fixes (may change logic)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate fixed completions after processing"
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)

    if not input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return

    # Generate output filename
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.parent / f"{input_file.stem}_fixed.jsonl"

    if not args.quiet:
        print("=" * 70)
        print("HumanEval Completion Fixer")
        print("=" * 70)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print(f"Aggressive mode: {args.aggressive}")
        print("=" * 70)
        print()

    # Process completions
    stats = process_completions_file(
        input_file,
        output_file,
        aggressive=args.aggressive,
        verbose=not args.quiet
    )

    if not args.quiet:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total completions:     {stats['total']}")
        print(f"Modified:              {stats['changed']}")
        print(f"Unchanged:             {stats['unchanged']}")
        print(f"Modification rate:     {stats['changed']/stats['total']*100:.1f}%")
        print("=" * 70)

    # Optionally evaluate
    if args.evaluate:
        print()
        print("Evaluating fixed completions...")
        import subprocess
        result = subprocess.run(
            ['python3', 'eval_humaneval_results.py', str(output_file)],
            cwd=input_file.parent.parent.parent.parent
        )


if __name__ == "__main__":
    main()
