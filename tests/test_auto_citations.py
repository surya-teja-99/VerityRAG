"""Tests for auto-citation assignment (VerityAgent.assign_citations).

The small local generator almost never emits [n] citations, so uncited
sentences are scored against the retrieved passages and given the
best-supported citation when one clears the threshold. No models are
needed here: the scorer is injected, as in test_verifier.py.
"""

from verityrag.agent import VerityAgent

PASSAGES = [
    {"id": "p1", "title": "T", "text": "Paris is the capital of France."},
    {"id": "p2", "title": "T", "text": "The Eiffel Tower is in Paris."},
]

SUPPORTS = 0.9
CONTRADICTS = 0.05


def keyword_scorer(sentence: str, passage_text: str) -> float:
    # A sentence is supported only when its key claim appears in the passage.
    # "Eiffel ... in London" is supported by neither passage: p2 mentions the
    # Eiffel Tower but says Paris, not London.
    if "capital" in sentence and "capital" in passage_text:
        return SUPPORTS
    if "Eiffel" in sentence and "Eiffel" in passage_text and "London" not in sentence:
        return SUPPORTS
    return CONTRADICTS


def make_agent(threshold: float = 0.35) -> VerityAgent:
    return VerityAgent(retriever=None, reranker=None, llm=None, threshold=threshold)


def test_assigns_citation_to_supported_uncited_sentence():
    agent = make_agent()
    out = agent.assign_citations(
        "Paris is the capital of France.", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "Paris is the capital of France [1]."


def test_assigns_best_passage_not_first():
    agent = make_agent()
    out = agent.assign_citations(
        "The Eiffel Tower is in Paris.", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "The Eiffel Tower is in Paris [2]."


def test_leaves_unsupported_uncited_sentence_uncited():
    agent = make_agent()
    out = agent.assign_citations(
        "The Eiffel Tower is in London.", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "The Eiffel Tower is in London."


def test_preserves_existing_citations():
    agent = make_agent()
    out = agent.assign_citations(
        "Paris is the capital of France [1].", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "Paris is the capital of France [1]."


def test_preserves_out_of_range_citation():
    agent = make_agent()
    out = agent.assign_citations(
        "Paris is nice [7].", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "Paris is nice [7]."


def test_threshold_blocks_weak_support():
    agent = make_agent(threshold=0.95)
    out = agent.assign_citations(
        "Paris is the capital of France.", PASSAGES, score_fn=keyword_scorer
    )
    assert out == "Paris is the capital of France."


def test_assign_then_verify_marks_supported_answer_verified():
    agent = make_agent()
    cited = agent.assign_citations(
        "Paris is the capital of France.", PASSAGES, score_fn=keyword_scorer
    )
    _, overall = agent.verify(cited, PASSAGES, score_fn=keyword_scorer)
    assert overall == "VERIFIED"
