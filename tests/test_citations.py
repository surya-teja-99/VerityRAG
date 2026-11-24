"""Tests for citation parsing and question decomposition."""

from verityrag.agent import decompose, extract_citations


def test_extract_citations_valid():
    valid, invalid = extract_citations("Paris is the capital [1] of France [2].", 3)
    assert valid == [1, 2]
    assert invalid == []


def test_extract_citations_out_of_range():
    valid, invalid = extract_citations("Claim [1] and another [9].", 3)
    assert valid == [1]
    assert invalid == [9]


def test_extract_citations_none():
    assert extract_citations("No citations here.", 3) == ([], [])


def test_decompose_single_clause_is_noop():
    q = "What is the capital of France?"
    assert decompose(q) == [q]


def test_decompose_splits_conjunction():
    subs = decompose("Who wrote Hamlet and when was it published?")
    assert len(subs) >= 2
    assert subs[0] == "Who wrote Hamlet and when was it published?"
    # Every sub-question ends with a question mark.
    assert all(s.endswith("?") for s in subs)


def test_decompose_keeps_second_interrogative_word():
    # Previously rebuilt as the mangled "Who when was it published?".
    assert decompose("Who wrote Hamlet and when was it published?") == [
        "Who wrote Hamlet and when was it published?",
        "when was it published?",
    ]


def test_decompose_no_split_without_interrogative():
    # "and Germany" is part of one question, not a second question.
    q = "What is the capital of France and Germany?"
    assert decompose(q) == [q]


def test_decompose_no_split_on_or_without_interrogative():
    q = "Is Paris in France or Italy?"
    assert decompose(q) == [q]


def test_decompose_splits_on_semicolon_before_interrogative():
    assert decompose("Who wrote Hamlet; when was it published?") == [
        "Who wrote Hamlet; when was it published?",
        "when was it published?",
    ]
