"""VerityRAG demo: 2 sample Q&A with citations + verifier verdicts, then metrics.

Runs retrieval ablations on 100 SQuAD dev questions (fast, no generation) and
the full agent pipeline on 10 questions. Target: under ~10 minutes on CPU.
The 2 showcased Q&A are the first 2 of the 10 full-pipeline questions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verityrag.agent import VerityAgent  # noqa: E402
from verityrag.eval import print_table, run_full_agent, run_retrieval_eval, sample_questions  # noqa: E402
from verityrag.ingest import (  # noqa: E402
    CORPUS_PATH, DATA_DIR, QUESTIONS_PATH, ingest, load_corpus, load_questions,
)
from verityrag.llm import LocalBackend  # noqa: E402
from verityrag.metrics import exact_match, f1_score  # noqa: E402
from verityrag.rerank import CrossEncoderReranker  # noqa: E402
from verityrag.retrieval import HybridRetriever  # noqa: E402


def show_qa(agent: VerityAgent, question: dict) -> None:
    res = agent.answer(question["question"])
    print(f"\nQ: {question['question']}")
    print(f"A: {res.answer}")
    print(f"Verdict: {res.verdict} | EM={exact_match(res.answer, question['answers']):.0f} "
          f"F1={f1_score(res.answer, question['answers']):.2f} | {res.latency_s:.1f}s")
    for s in res.sentences:
        mark = "FLAGGED" if s.flagged else "ok"
        score = f"{s.score:.2f}" if s.score is not None else "n/a"
        print(f"  [{mark} score={score}] {s.text}  cites={s.citations} {s.reason}")
    print("  Passages:")
    for i, p in enumerate(res.passages, 1):
        print(f"    [{i}] {p['id']} :: {p['text'][:120]}...")


def main() -> None:
    t0 = time.time()
    print("=== VerityRAG demo ===\n")
    if not CORPUS_PATH.exists() or not QUESTIONS_PATH.exists():
        ingest()
    corpus = load_corpus()
    questions = load_questions()
    print(f"Corpus: {len(corpus)} passages | {len(questions)} questions available\n")

    retriever = HybridRetriever(corpus, cache_dir=DATA_DIR)
    retriever.build_dense_index()
    reranker = CrossEncoderReranker()
    agent = VerityAgent(retriever, reranker, LocalBackend(num_threads=2))

    full_qs = sample_questions(questions, 10)
    print("--- Sample Q&A (with verifier) ---")
    for q in full_qs[:2]:
        show_qa(agent, q)

    print("\n--- Retrieval ablations: 100 questions (no generation) ---")
    ret_qs = sample_questions(questions, 100)
    retrieval = run_retrieval_eval(ret_qs, retriever, reranker)
    retrieval["full_agent"] = retrieval["rerank"]

    print("\n--- Full agent: 10 questions ---")
    agent_metrics, details = run_full_agent(full_qs, agent)

    results = {
        "n_retrieval": len(ret_qs),
        "n_full": len(full_qs),
        "n_passages": len(corpus),
        "retrieval": retrieval,
        "full_agent": agent_metrics,
        "details": details,
    }
    print_table(results)

    out = Path(__file__).resolve().parent.parent / "results" / "demo_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out}")
    print(f"\nTotal demo wall-clock: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
