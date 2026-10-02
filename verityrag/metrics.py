"""Evaluation metrics: SQuAD EM/F1 plus retrieval metrics.

The SQuAD metric logic (normalization + token F1) is implemented here from
the official definition, not imported.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return 1.0 if any(norm_pred == normalize_answer(g) for g in ground_truths) else 0.0


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == gold_id:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
