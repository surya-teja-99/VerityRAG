"""Tests for RRF fusion and hybrid retrieval plumbing."""

from verityrag.retrieval import rrf_fuse


def test_rrf_fusion_ordering():
    # b appears in both lists (once at rank 1) -> should outrank a and c.
    fused = rrf_fuse([["a", "b", "c"], ["b"]], k=60)
    ids = [d for d, _ in fused]
    assert ids == ["b", "a", "c"]
    # Exact RRF scores: b = 1/62 + 1/61, a = 1/61, c = 1/63.
    scores = dict(fused)
    assert abs(scores["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(scores["a"] - 1 / 61) < 1e-12


def test_rrf_deterministic_tiebreak():
    # Identical scores -> ids sorted ascending for determinism.
    fused = rrf_fuse([["b"], ["a"]], k=60)
    ids = [d for d, _ in fused]
    assert ids == ["a", "b"]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []
