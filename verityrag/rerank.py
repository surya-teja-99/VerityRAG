"""Cross-encoder reranking of first-stage retrieval results."""

from __future__ import annotations

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL, batch_size: int = 32,
                 max_chars: int = 800):
        # max_chars: truncate rerank inputs. The cross-encoder's attention cost
        # is quadratic in sequence length; on CPU, full SQuAD paragraphs
        # (~630 chars avg, long tail) make bulk reranking ~8x slower for only
        # marginal ranking changes. Documented trade-off, not a silent one.
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Raw cross-encoder relevance scores for (query, text) pairs."""
        truncated = [t[: self.max_chars] for t in texts]
        return [float(s) for s in self.model.predict(
            [(query, t) for t in truncated], batch_size=self.batch_size
        )]

    def rerank(
        self, query: str, docs: list[dict], top_n: int = 5
    ) -> list[tuple[dict, float]]:
        """Rerank ``docs`` (each with a 'text' field) to the top_n by relevance."""
        if not docs:
            return []
        scores = self.score(query, [d["text"] for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return ranked[:top_n]
