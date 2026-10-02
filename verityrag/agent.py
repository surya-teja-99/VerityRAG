"""The VerityRAG agent loop.

Pipeline:
  1. Decompose the question into sub-questions (simple, honest heuristic:
     split on conjunctions/semicolons and rebuild each fragment as a question).
  2. Retrieve per sub-question with hybrid search and merge the results.
  3. Rerank to the top-5 passages.
  4. Synthesize an answer with [n] citations via the LLM backend.
  5. Verify: score every answer sentence against its cited passages with the
     cross-encoder. Sentences below the threshold (or with no citation) are
     flagged; heavily-flagged answers are marked LOW CONFIDENCE.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SPLIT_RE = re.compile(r"\s+(?:and|or|but)\s+|;")

# Sigmoid of the cross-encoder logit. Heuristic: below this, a sentence is not
# considered supported by the passage it cites.
VERIFIER_THRESHOLD = 0.35

PASSAGE_CHARS = 500  # truncate passages in the synthesis prompt (CPU budget)


def decompose(question: str) -> list[str]:
    """Heuristic question decomposition.

    Splits on conjunctions/semicolons; each extra fragment is rebuilt as a
    question using the original interrogative word. This is a deliberately
    simple heuristic — it helps multi-hop-ish questions and is a no-op for
    single-clause questions.
    """
    q = question.strip()
    parts = [p.strip(" ,") for p in SPLIT_RE.split(q) if p.strip(" ,")]
    if len(parts) <= 1:
        return [q]
    first_word = q.split()[0] if q.split() else "What"
    subqs = [q]
    for part in parts[1:]:
        frag = part.rstrip("?")
        if not frag.lower().startswith(first_word.lower()):
            frag = f"{first_word} {frag[0].lower() + frag[1:]}" if frag else frag
        subqs.append(frag.rstrip("?") + "?")
    # Deduplicate, preserve order.
    seen, out = set(), []
    for s in subqs:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def extract_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    """Return (valid citation indices, out-of-range indices) from text."""
    valid, invalid = [], []
    for m in CITATION_RE.finditer(text):
        idx = int(m.group(1))
        (valid if 1 <= idx <= n_passages else invalid).append(idx)
    return valid, invalid


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


@dataclass
class SentenceVerdict:
    text: str
    citations: list[int]
    score: float | None
    flagged: bool
    reason: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str
    passages: list[dict] = field(default_factory=list)
    sentences: list[SentenceVerdict] = field(default_factory=list)
    verdict: str = "NO ANSWER"
    latency_s: float = 0.0
    n_subquestions: int = 1


class VerityAgent:
    def __init__(self, retriever, reranker, llm, threshold: float = VERIFIER_THRESHOLD):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.threshold = threshold

    # -- retrieval ------------------------------------------------------
    def retrieve(self, question: str, per_query_k: int = 30, top_n: int = 5):
        subqs = decompose(question)
        merged: dict[str, float] = {}
        for sq in subqs:
            for doc_id, score in self.retriever.hybrid_search(sq, top_k=per_query_k):
                merged[doc_id] = max(merged.get(doc_id, 0.0), score)
        # Deterministic order before rerank: score desc, id asc.
        candidates = [self.retriever.get(d) for d, _ in
                      sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))]
        ranked = self.reranker.rerank(question, candidates[:per_query_k], top_n=top_n)
        return [doc for doc, _ in ranked], len(subqs)

    # -- synthesis ------------------------------------------------------
    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        ctx = "\n".join(
            f"[{i + 1}] {p['text'][:PASSAGE_CHARS]}" for i, p in enumerate(passages)
        )
        return (
            "Answer the question using ONLY the passages below. "
            "Reply in 1-3 sentences. Cite every factual claim with the passage "
            "number like [1] or [2]. If the passages do not contain the answer, "
            "say \"I don't know\".\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )

    # -- verification ---------------------------------------------------
    def verify(
        self,
        answer: str,
        passages: list[dict],
        score_fn=None,
    ) -> tuple[list[SentenceVerdict], str]:
        """Score each sentence against its cited passages.

        ``score_fn(sentence, passage_text) -> float`` defaults to the
        cross-encoder with a sigmoid applied. Returns (sentence verdicts,
        overall verdict).
        """
        if score_fn is None:
            def score_fn(sent: str, ptext: str) -> float:
                # (reranker.score truncates long inputs for CPU speed;
                # see CrossEncoderReranker.)
                logit = self.reranker.score(sent, [ptext])[0]
                return 1.0 / (1.0 + math.exp(-logit))

        verdicts: list[SentenceVerdict] = []
        for sent in split_sentences(answer):
            cites, bad = extract_citations(sent, len(passages))
            if bad:
                verdicts.append(SentenceVerdict(sent, cites, None, True,
                                                f"cites out-of-range passages {bad}"))
                continue
            if not cites:
                verdicts.append(SentenceVerdict(sent, [], None, True, "no citation"))
                continue
            best = max(score_fn(sent, passages[i - 1]["text"]) for i in cites)
            flagged = best < self.threshold
            verdicts.append(SentenceVerdict(
                sent, cites, best, flagged,
                "" if not flagged else f"support score {best:.2f} < {self.threshold}",
            ))

        if not verdicts:
            overall = "NO ANSWER"
        else:
            frac = sum(v.flagged for v in verdicts) / len(verdicts)
            if frac > 0.5:
                overall = "LOW CONFIDENCE"
            elif frac > 0:
                overall = "PARTIALLY VERIFIED"
            else:
                overall = "VERIFIED"
        return verdicts, overall

    # -- full loop ------------------------------------------------------
    def answer(self, question: str) -> AgentResult:
        t0 = time.time()
        passages, n_sub = self.retrieve(question)
        if not passages:
            return AgentResult(question, "I don't know.", [], [], "NO ANSWER",
                               time.time() - t0, n_sub)
        prompt = self._build_prompt(question, passages)
        raw = self.llm.generate(prompt, max_new_tokens=64)
        sentences, verdict = self.verify(raw, passages)
        return AgentResult(
            question=question,
            answer=raw,
            passages=passages,
            sentences=sentences,
            verdict=verdict,
            latency_s=time.time() - t0,
            n_subquestions=n_sub,
        )
