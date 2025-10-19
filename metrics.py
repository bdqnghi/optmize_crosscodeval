#!/usr/bin/env python3
"""
Metric Calculations for Code Completion Evaluation
These are the EXACT metric functions from eval_python_all.py and eval_java_all.py
"""

# Try to import Levenshtein, fall back to simple implementation if not available
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


# ========== METRIC CALCULATIONS (Exact copies from eval scripts) ==========

def calculate_edit_similarity(pred: str, ref: str) -> float:
    """
    Calculate edit similarity - EXACT copy from eval_python_all.py and eval_java_all.py

    Args:
        pred: Predicted text
        ref: Reference/ground truth text

    Returns:
        Float between 0 and 1, where 1 means perfect match
    """
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
    """
    Check exact match after normalization - EXACT copy from eval scripts

    Args:
        pred: Predicted text
        ref: Reference/ground truth text

    Returns:
        True if normalized strings match exactly
    """
    pred_normalized = " ".join(pred.split())
    ref_normalized = " ".join(ref.split())
    return pred_normalized == ref_normalized


def check_pass(es: float, em: bool, es_threshold: float = 0.7) -> bool:
    """
    Check if sample passes - EXACT copy from eval scripts

    Args:
        es: Edit similarity score
        em: Exact match boolean
        es_threshold: Threshold for edit similarity (default: 0.7)

    Returns:
        True if both conditions are met (es >= threshold AND em == True)
    """
    return es >= es_threshold and em


def calculate_pass_at_1(results: list, es_threshold: float = 0.7) -> float:
    """
    Calculate pass@1 metric for a list of results

    Args:
        results: List of result dictionaries with 'es' and 'em' keys
        es_threshold: Threshold for edit similarity

    Returns:
        Pass@1 rate as a float between 0 and 1
    """
    if not results:
        return 0.0

    passes = sum(1 for r in results if check_pass(r.get('es', 0), r.get('em', False), es_threshold))
    return passes / len(results)


def calculate_average_edit_similarity(results: list) -> float:
    """
    Calculate average edit similarity across all results

    Args:
        results: List of result dictionaries with 'es' key

    Returns:
        Average edit similarity
    """
    if not results:
        return 0.0

    return sum(r.get('es', 0) for r in results) / len(results)


def calculate_exact_match_rate(results: list) -> float:
    """
    Calculate exact match rate across all results

    Args:
        results: List of result dictionaries with 'em' key

    Returns:
        Exact match rate as a float between 0 and 1
    """
    if not results:
        return 0.0

    return sum(1 for r in results if r.get('em', False)) / len(results)


def calculate_all_metrics(results: list, es_threshold: float = 0.7) -> dict:
    """
    Calculate all metrics at once

    Args:
        results: List of result dictionaries
        es_threshold: Threshold for edit similarity

    Returns:
        Dictionary with all metrics
    """
    return {
        'pass@1': calculate_pass_at_1(results, es_threshold),
        'edit_similarity': calculate_average_edit_similarity(results),
        'exact_match': calculate_exact_match_rate(results),
        'total_samples': len(results),
        'passes': sum(1 for r in results if check_pass(r.get('es', 0), r.get('em', False), es_threshold))
    }