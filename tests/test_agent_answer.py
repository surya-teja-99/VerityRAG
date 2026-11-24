"""End-to-end tests for VerityAgent.answer() with stub backends.

No models are loaded: the retriever, reranker, and LLM are fakes, so
these tests cover the full loop — retrieve -> rerank -> synthesize ->
auto-cite -> verify — including the wiring added in commit 15a282d.
"""

import math

from verityrag.agent import VerityAgent

PASSAGES = [
    {"id": "p1", "title": "T", "text": "Paris is the capital of France."},
    {"id": "p2", "title": "T", "text": "The Eiffel Tower is in Paris."},
]


class StubRetriever:
    def __init__(self, passages):
        self._by_id = {p["id"]: p for p in passages}

    def hybrid_search(self, query, top_k=30):
        return [(p["id"], 1.0) for p in self._by_id.values()]

    def get(self, doc_id):
        return self._by_id[doc_id]


class StubReranker:
    """score() returns a cross-encoder-style logit; 2.0 -> sigmoid ~0.88."""

    def rerank(self, query, docs, top_n=5):
        return [(d, 1.0) for d in docs[:top_n]]

    def score(self, query, texts):
        return [2.0 for _ in texts]


class StubLLM:
    def __init__(self, answer):
        self._answer = answer

    def generate(self, prompt, max_new_tokens=64):
        return self._answer


def make_agent(answer="Paris is the capital of France.", passages=None):
    return VerityAgent(
        retriever=StubRetriever(passages if passages is not None else PASSAGES),
        reranker=StubReranker(),
        llm=StubLLM(answer),
    )


def test_answer_auto_cites_and_verifies():
    agent = make_agent()
    res = agent.answer("What is the capital of France?")
    assert res.answer == "Paris is the capital of France [1]."
    assert res.verdict == "VERIFIED"
    assert len(res.sentences) == 1
    assert res.sentences[0].citations == [1]
    assert not res.sentences[0].flagged
    assert res.passages == PASSAGES


def test_answer_keeps_generator_citation():
    agent = make_agent(answer="Paris is the capital of France [1].")
    res = agent.answer("What is the capital of France?")
    assert res.answer == "Paris is the capital of France [1]."
    assert res.verdict == "VERIFIED"


def test_answer_no_passages_is_no_answer():
    agent = make_agent(passages=[])
    res = agent.answer("What is the capital of France?")
    assert res.answer == "I don't know."
    assert res.verdict == "NO ANSWER"
    assert res.sentences == []


def test_sigmoid_of_stub_logit_clears_threshold():
    # Sanity-check the stub: the default score_fn maps logit 2.0 to ~0.88,
    # above the 0.35 verifier threshold, so auto-citation can fire.
    assert 1.0 / (1.0 + math.exp(-2.0)) > 0.35
