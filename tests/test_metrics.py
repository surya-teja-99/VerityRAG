"""Tests for the SQuAD EM/F1 metric implementation."""

from verityrag.metrics import exact_match, f1_score, mrr_at_k, normalize_answer, recall_at_k


def test_normalize_answer():
    assert normalize_answer("  The   quick, brown fox! ") == "quick brown fox"
    assert normalize_answer("An apple") == "apple"


def test_exact_match():
    assert exact_match("Paris", ["Paris", "Paris, France"]) == 1.0
    assert exact_match("the Paris", ["Paris"]) == 1.0  # article stripped
    assert exact_match("London", ["Paris"]) == 0.0
    assert exact_match("", ["Paris"]) == 0.0


def test_f1_score():
    assert f1_score("Paris", ["Paris"]) == 1.0
    # Partial overlap: pred "Paris France", gold "Paris" -> P=0.5, R=1.0, F1=2/3
    assert abs(f1_score("Paris France", ["Paris"]) - 2 / 3) < 1e-9
    assert f1_score("London", ["Paris"]) == 0.0
    assert f1_score("", ["Paris"]) == 0.0
    # Takes the max over multiple gold answers.
    assert f1_score("Paris", ["London", "Paris"]) == 1.0


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "b", 2) == 1.0
    assert recall_at_k(ranked, "c", 2) == 0.0
    assert recall_at_k(ranked, "c", 3) == 1.0


def test_mrr_at_k():
    assert mrr_at_k(["a", "b", "c"], "b", 10) == 0.5
    assert mrr_at_k(["a", "b", "c"], "z", 10) == 0.0
    assert mrr_at_k(["a", "b", "c"], "c", 2) == 0.0  # outside k
