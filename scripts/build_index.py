"""One-time setup: ingest SQuAD, build the corpus, cache dense embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verityrag.ingest import DATA_DIR, ingest, load_corpus  # noqa: E402
from verityrag.retrieval import HybridRetriever  # noqa: E402


def main() -> None:
    ingest(DATA_DIR)
    corpus = load_corpus()
    retriever = HybridRetriever(corpus, cache_dir=DATA_DIR)
    retriever.build_dense_index()
    print("Setup complete.")


if __name__ == "__main__":
    main()
