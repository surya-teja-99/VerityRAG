"""Tests for the verifier agent using a synthetic scoring function."""

from verityrag.agent import VerityAgent

PASSAGES = [
    {"id": "p1", "title": "T", "text": "Paris is the capital of France."},
    {"id": "p2", "title": "T", "text": "The Eiffel Tower is in Paris."},
]

SUPPORTS = 0.9
CONTRADICTS = 0.05


def fake_scorer(sentence: str, passage_text: str) -> float:
    # Synthetic ground truth: sentences mentioning the Eiffel Tower are
    # unsupported by p1 but supported by p2; everything else is supported.
    if "Eiffel" in sentence:
        return SUPPORTS if "Eiffel" in passage_text else CONTRADICTS
    return SUPPORTS


def make_agent() -> VerityAgent:
    # score_fn is injected, so no models are needed.
    return VerityAgent(retriever=None, reranker=None, llm=None)


def test_verifier_accepts_supported_claim():
    agent = make_agent()
    verdicts, overall = agent.verify("Paris is the capital of France [1].",
                                     PASSAGES, score_fn=fake_scorer)
    assert len(verdicts) == 1
    assert not verdicts[0].flagged
    assert overall == "VERIFIED"


def test_verifier_flags_unsupported_claim():
    agent = make_agent()
    verdicts, overall = agent.verify(
        "The Eiffel Tower is in London [1].", PASSAGES, score_fn=fake_scorer)
    assert len(verdicts) == 1
    assert verdicts[0].flagged
    assert "support score" in verdicts[0].reason


def test_verifier_flags_uncited_sentence():
    agent = make_agent()
    verdicts, _ = agent.verify("Paris is nice.", PASSAGES, score_fn=fake_scorer)
    assert verdicts[0].flagged
    assert verdicts[0].reason == "no citation"


def test_verifier_flags_out_of_range_citation():
    agent = make_agent()
    verdicts, _ = agent.verify("Paris is nice [7].", PASSAGES, score_fn=fake_scorer)
    assert verdicts[0].flagged
    assert "out-of-range" in verdicts[0].reason


def test_verifier_marks_heavily_flagged_low_confidence():
    agent = make_agent()
    answer = ("The Eiffel Tower is in London [1]. "
              "The Eiffel Tower was built in 1066 [1]. "
              "Paris is the capital of France [1].")
    verdicts, overall = agent.verify(answer, PASSAGES, score_fn=fake_scorer)
    assert sum(v.flagged for v in verdicts) == 2
    assert overall == "LOW CONFIDENCE"


def test_verifier_partial():
    agent = make_agent()
    answer = "Paris is the capital of France [1]. The Eiffel Tower is in London [1]."
    _, overall = agent.verify(answer, PASSAGES, score_fn=fake_scorer)
    assert overall == "PARTIALLY VERIFIED"
