"""
Evaluation metrics for the RDE-LMC system.
"""
import os
import sys
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_article_id(article_id: str) -> str:
    """Normalize article ID to canonical form: 《法律名》第X条"""
    article_id = article_id.strip()
    # Add brackets if missing
    if not article_id.startswith('《'):
        # Try to detect if it's "法律名 第X条" format
        import re
        m = re.match(r'^(.+?)\s+第(.+?)条$', article_id)
        if m:
            article_id = f"《{m.group(1)}》第{m.group(2)}条"
    return article_id


def numerical_accuracy(
    predicted: List[float],
    ground_truth: List[float],
    tolerance: float = 0.01
) -> float:
    """
    Check if predicted numerical answers match ground truth within tolerance.

    For single answers: exact match within relative tolerance.
    For multi-answers: all ground truth values must appear in predictions.

    Args:
        predicted: List of predicted numerical values
        ground_truth: List of ground truth numerical values
        tolerance: Relative tolerance (default 1%)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    def is_close(pred: float, truth: float) -> bool:
        """Check if two values are close within relative tolerance."""
        if truth == 0:
            return abs(pred) <= tolerance
        rel_err = abs(pred - truth) / abs(truth)
        return rel_err <= tolerance

    # For each ground truth value, check if it appears in predictions
    for truth_val in ground_truth:
        found = any(is_close(pred, truth_val) for pred in predicted)
        if not found:
            return 0.0

    return 1.0


def article_f1(
    predicted_articles: List[str],
    ground_truth_articles: List[str]
) -> float:
    """
    Compute F1 score for article retrieval.

    Args:
        predicted_articles: List of predicted article IDs
        ground_truth_articles: List of ground truth article IDs

    Returns:
        F1 score (0.0 to 1.0)
    """
    if not ground_truth_articles:
        return 1.0 if not predicted_articles else 0.0

    if not predicted_articles:
        return 0.0

    # Normalize article IDs
    pred_set = {_normalize_article_id(a) for a in predicted_articles}
    truth_set = {_normalize_article_id(a) for a in ground_truth_articles}

    intersection = pred_set & truth_set

    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(truth_set)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def article_exact_match(
    predicted_articles: List[str],
    ground_truth_articles: List[str]
) -> float:
    """
    Check if predicted article sets exactly match ground truth.

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_set = {_normalize_article_id(a) for a in predicted_articles}
    truth_set = {_normalize_article_id(a) for a in ground_truth_articles}
    return 1.0 if pred_set == truth_set else 0.0


def compute_all_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    tolerance: float = 0.01
) -> Dict:
    """
    Compute aggregate metrics across all samples.

    Args:
        predictions: List of {numerical_answer: [...], article_answer: [...]}
        ground_truths: List of {numerical_answer: [...], article_answer: [...]}
        tolerance: Numerical tolerance for accuracy check

    Returns:
        Dict with aggregate metrics
    """
    n = len(predictions)
    if n == 0:
        return {"n_samples": 0}

    num_acc_scores = []
    art_f1_scores = []
    art_em_scores = []
    both_correct = []
    n_failed = 0

    for pred, truth in zip(predictions, ground_truths):
        pred_num = pred.get("numerical_answer", [])
        pred_art = pred.get("article_answer", [])
        truth_num = truth.get("numerical_answer", [])
        truth_art = truth.get("article_answer", [])

        # Check if pipeline failed (no output)
        if not pred_num and not pred_art:
            n_failed += 1

        num_score = numerical_accuracy(pred_num, truth_num, tolerance)
        art_f1_score = article_f1(pred_art, truth_art)
        art_em_score = article_exact_match(pred_art, truth_art)
        both_score = 1.0 if num_score == 1.0 and art_em_score == 1.0 else 0.0

        num_acc_scores.append(num_score)
        art_f1_scores.append(art_f1_score)
        art_em_scores.append(art_em_score)
        both_correct.append(both_score)

    return {
        "n_samples": n,
        "n_failed": n_failed,
        "numerical_accuracy": sum(num_acc_scores) / n,
        "article_f1": sum(art_f1_scores) / n,
        "article_exact_match": sum(art_em_scores) / n,
        "both_correct": sum(both_correct) / n,
    }
