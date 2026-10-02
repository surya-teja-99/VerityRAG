"""Hybrid retrieval: BM25 + dense embeddings fused with Reciprocal Rank Fusion.

The dense index is a brute-force cosine search over L2-normalized embeddings
with numpy — fine for a ~2k-passage corpus. Embeddings are cached on disk so
reruns are instant.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over ranked id lists.

    score(d) = sum over lists of 1 / (k + rank), rank starting at 1.
    Returns (id, score) sorted by descending score; ties broken by id for
    determinism.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    def __init__(self, passages: list[dict], cache_dir: Path | str = "data"):
        self.passages = passages
        self.ids = [p["id"] for p in passages]
        self.by_id = {p["id"]: p for p in passages}
        self.cache_dir = Path(cache_dir)
        self.texts = [p["text"] for p in passages]

        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self._model = None
        self._embeddings: np.ndarray | None = None

    # -- dense index -----------------------------------------------------
    @property
    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _embeddings_path(self) -> Path:
        return self.cache_dir / "embeddings.npy"

    def build_dense_index(self, force: bool = False) -> None:
        """Encode the corpus (or load the cached matrix) and L2-normalize it."""
        path = self._embeddings_path()
        if path.exists() and not force:
            self._embeddings = np.load(path)
            print(f"Loaded cached embeddings: {path} {self._embeddings.shape}")
            return
        print(f"Encoding {len(self.texts)} passages with {EMBEDDING_MODEL} ...")
        emb = self._encoder.encode(
            self.texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        self._embeddings = emb
        print(f"Saved embeddings to {path}")

    def dense_search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        if self._embeddings is None:
            self.build_dense_index()
        q = self._encoder.encode([query], convert_to_numpy=True)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        scores = (self._embeddings @ q[0])
        top = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[i], float(scores[i])) for i in top]

    # -- sparse index ----------------------------------------------------
    def bm25_search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[i], float(scores[i])) for i in top]

    # -- hybrid ----------------------------------------------------------
    def hybrid_search(
        self, query: str, top_k: int = 30, rrf_k: int = 60
    ) -> list[tuple[str, float]]:
        bm25_ids = [doc_id for doc_id, _ in self.bm25_search(query, top_k)]
        dense_ids = [doc_id for doc_id, _ in self.dense_search(query, top_k)]
        return rrf_fuse([bm25_ids, dense_ids], k=rrf_k)[:top_k]

    def get(self, doc_id: str) -> dict:
        return self.by_id[doc_id]
