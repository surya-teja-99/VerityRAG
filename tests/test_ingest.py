"""Tests for ingestion determinism and deduplication."""

from verityrag.ingest import build_corpus

FIXTURE = {
    "data": [
        {"title": "Article A", "paragraphs": [
            {"context": "First paragraph text.",
             "qas": [{"id": "q1", "question": "What is first?",
                      "answers": [{"text": "First paragraph text."}]}]},
            {"context": "First paragraph text.",  # duplicate paragraph
             "qas": [{"id": "q2", "question": "Repeat?",
                      "answers": [{"text": "First paragraph text."}]}]},
            {"context": "Second paragraph text.",
             "qas": [{"id": "q3", "question": "What is second?",
                      "answers": [{"text": "Second paragraph text."}]}]},
        ]},
    ]
}


def test_dedupe_collapses_duplicate_paragraphs():
    passages, questions = build_corpus(FIXTURE)
    assert len(passages) == 2
    assert len(questions) == 3
    # Both questions from the duplicated paragraph map to the same gold id.
    gold_ids = {q["id"]: q["gold_passage_id"] for q in questions}
    assert gold_ids["q1"] == gold_ids["q2"]


def test_ingest_is_deterministic():
    first = build_corpus(FIXTURE)
    second = build_corpus(FIXTURE)
    assert first == second
    assert [p["id"] for p in first[0]] == ["Article A::0", "Article A::1"]
