#!/usr/bin/env python3
"""
Evaluation Utilities
Utilities for FIM-based code completion evaluation (CrossCode datasets)
Includes metrics, FIM parsing, post-processing, and extraction strategies

For HumanEval evaluation, use eval_humaneval.sh which uses the official OpenAI library
"""

import json
import re
from typing import List, Dict, Tuple

# Try to import Levenshtein, fall back to simple implementation
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

    def simple_edit_distance(s1: str, s2: str) -> int:
        """Fallback edit distance calculation"""
        if len(s1) < len(s2):
            return simple_edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# ========== DATA LOADING ==========

def load_config(config_path: str = "config.json") -> Dict:
    """Load model configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load dataset from JSONL file"""
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


# ========== FIM PARSING ==========

def parse_fim_prompt(prompt: str) -> Tuple[str, str]:
    """
    Parse FIM (Fill-In-the-Middle) prompt to extract prefix and suffix

    Args:
        prompt: FIM-formatted prompt string

    Returns:
        Tuple of (prefix, suffix)
    """
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


# ========== EXTRACTION STRATEGIES ==========

def contextual_extract_python(generated: str, groundtruth: str, prefix: str, suffix: str) -> str:
    """Python extraction strategy - extract relevant code from generated completion"""
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


def contextual_extract_java(generated: str, groundtruth: str, prefix: str, suffix: str) -> str:
    """Java extraction strategy - extract relevant code from generated completion"""
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


# ========== POST-PROCESSING (Without Ground Truth) ==========

def remove_markdown_artifacts(text: str) -> str:
    """Remove markdown code block markers"""
    # Remove ```python, ```java, etc.
    text = re.sub(r'^```\w*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    return text


def fix_obvious_syntax_errors(code: str) -> str:
    """Fix obvious syntax errors that would prevent code from running"""
    if not code:
        return code

    # Balance parentheses
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        code += ')' * (open_parens - close_parens)

    # Balance square brackets
    open_brackets = code.count('[')
    close_brackets = code.count(']')
    if open_brackets > close_brackets:
        code += ']' * (open_brackets - close_brackets)

    # Balance curly braces
    open_braces = code.count('{')
    close_braces = code.count('}')
    if open_braces > close_braces:
        code += '}' * (open_braces - close_braces)

    return code


def fix_indentation(completion: str, full_code: str = "") -> str:
    """Fix indentation for function bodies"""
    if 'def ' in full_code and '"""' in full_code:
        lines = completion.split('\n')
        fixed_lines = []

        for line in lines:
            if line.strip() and not line.startswith('    '):
                # Add proper indentation
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)

        completion = '\n'.join(fixed_lines)

    return completion


def post_process_prediction(prediction: str, language: str = "python") -> str:
    """
    Apply post-processing to prediction (without using ground truth)

    Args:
        prediction: The model's prediction
        language: Programming language ('python' or 'java')

    Returns:
        Post-processed prediction
    """
    # Remove markdown artifacts
    processed = remove_markdown_artifacts(prediction)

    # Fix syntax errors
    processed = fix_obvious_syntax_errors(processed)

    # Fix indentation if needed
    if language == "python":
        processed = fix_indentation(processed)

    return processed


# ========== METRICS ==========

def calculate_edit_similarity(pred: str, ref: str) -> float:
    """Calculate edit similarity between prediction and reference"""
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    if LEVENSHTEIN_AVAILABLE:
        distance = Levenshtein.distance(pred, ref)
    else:
        distance = simple_edit_distance(pred, ref)

    max_len = max(len(pred), len(ref))
    return 1.0 - (distance / max_len)


def calculate_exact_match(pred: str, ref: str) -> bool:
    """Check exact match after normalization"""
    pred_normalized = " ".join(pred.split())
    ref_normalized = " ".join(ref.split())
    return pred_normalized == ref_normalized


def check_pass(es: float, em: bool, es_threshold: float = 0.7) -> bool:
    """Check if sample passes based on ES and EM thresholds"""
    return es >= es_threshold and em


def calculate_accuracy(results: List[Dict], es_threshold: float = 0.7) -> float:
    """Calculate accuracy (percentage of passing samples)"""
    if not results:
        return 0.0
    passes = sum(1 for r in results if check_pass(r.get('es', 0), r.get('em', False), es_threshold))
    return passes / len(results)


def calculate_metrics(results: List[Dict], es_threshold: float = 0.7) -> Dict:
    """Calculate all metrics for a set of results"""
    if not results:
        return {
            'accuracy': 0.0,
            'edit_similarity': 0.0,
            'exact_match': 0.0,
            'total_samples': 0,
            'passes': 0
        }

    passes = sum(1 for r in results if check_pass(r.get('es', 0), r.get('em', False), es_threshold))
    avg_es = sum(r.get('es', 0) for r in results) / len(results)
    avg_em = sum(1 for r in results if r.get('em', False)) / len(results)

    return {
        'accuracy': passes / len(results),
        'edit_similarity': avg_es,
        'exact_match': avg_em,
        'total_samples': len(results),
        'passes': passes
    }
