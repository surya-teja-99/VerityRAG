"""Eval harness: retrieval ablations + full-agent metrics on SQuAD v1.1 dev.

Ablations (retrieval metrics: Recall@5, Recall@20, MRR@10):
  1. BM25-only            — sparse baseline, top-20
  2. dense-only           — all-MiniLM-L6-v2 cosine, top-20
  3. hybrid (RRF)         — BM25 + dense fused with RRF, top-20
  4. hybrid+rerank        — hybrid top-30 reranked to top-5 with the cross-encoder
  5. full agent           — same retrieval as (4), plus generation + verifier

Config 5 additionally reports: Exact Match, token F1 (SQuAD logic), % of
answers with >=1 verifier-flagged sentence, % marked LOW CONFIDENCE, and mean
latency. Note: configs 4-5 return only 5 passages, so their Recall@20 equals
their Recall@5 by construction.

Usage:
    python -m verityrag.eval --n_retrieval 100 --n_full 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .agent import VerityAgent
from .ingest import CORPUS_PATH, DATA_DIR, QUESTIONS_PATH, ingest, load_corpus, load_questions
from .llm import LocalBackend
from .metrics import exact_match, f1_score, mean, mrr_at_k, recall_at_k
from .rerank import CrossEncoderReranker
from .retrieval import HybridRetriever, rrf_fuse, tokenize

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RETRIEVAL_K = 20   # list size for pure-retrieval ablations
RERANK_CANDIDATES = 30
RERANK_TOP_N = 5


def sample_questions(questions: list[dict], n: int) -> list[dict]:
    """Deterministic strided sample spread across the dataset."""
    if n >= len(questions):
        return questions
    return [questions[round(i * (len(questions) - 1) / (n - 1))] for i in range(n)]


def batched_dense_scores(retriever: HybridRetriever, queries: list[str]) -> np.ndarray:
    """Cosine scores for all (query, passage) pairs; shape (n_q, n_docs)."""
    q = retriever._encoder.encode(queries, batch_size=64, convert_to_numpy=True,
                                  show_progress_bar=False)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q @ retriever._embeddings.T


def run_retrieval_eval(questions: list[dict], retriever: HybridRetriever,
                       reranker: CrossEncoderReranker) -> dict:
    queries = [q["question"] for q in questions]
    golds = [q["gold_passage_id"] for q in questions]
    ids = retriever.ids

    print("  BM25 + dense scoring ...")
    dense_scores = batched_dense_scores(retriever, queries)

    lists: dict[str, list[list[str]]] = {
        "bm25": [], "dense": [], "hybrid": [], "rerank": [],
    }
    t0 = time.time()
    for qi, query in enumerate(queries):
        bm25_scores = retriever.bm25.get_scores(tokenize(query))
        bm25_ids = [ids[i] for i in np.argsort(-bm25_scores, kind="stable")[:RETRIEVAL_K]]
        dense_ids = [ids[i] for i in np.argsort(-dense_scores[qi], kind="stable")[:RETRIEVAL_K]]
        hybrid_ids = [d for d, _ in rrf_fuse([bm25_ids, dense_ids])[:RETRIEVAL_K]]
        lists["bm25"].append(bm25_ids)
        lists["dense"].append(dense_ids)
        lists["hybrid"].append(hybrid_ids)
    print(f"  first-stage retrieval: {time.time() - t0:.1f}s for {len(queries)} questions")

    print("  reranking top-30 -> top-5 ...")
    t0 = time.time()
    for qi, query in enumerate(queries):
        toks = tokenize(query)
        bm30 = [ids[i] for i in np.argsort(
            -retriever.bm25.get_scores(toks), kind="stable")[:RERANK_CANDIDATES]]
        de30 = [ids[i] for i in np.argsort(-dense_scores[qi], kind="stable")[:RERANK_CANDIDATES]]
        fused30 = [d for d, _ in rrf_fuse([bm30, de30])[:RERANK_CANDIDATES]]
        docs = [retriever.get(d) for d in fused30]
        reranked = reranker.rerank(query, docs, top_n=RERANK_TOP_N)
        lists["rerank"].append([d["id"] for d, _ in reranked])
    print(f"  reranking: {time.time() - t0:.1f}s for {len(queries)} questions")

    out = {}
    for name, ranked in lists.items():
        out[name] = {
            "recall@5": mean([recall_at_k(r, g, 5) for r, g in zip(ranked, golds)]),
            "recall@20": mean([recall_at_k(r, g, 20) for r, g in zip(ranked, golds)]),
            "mrr@10": mean([mrr_at_k(r, g, 10) for r, g in zip(ranked, golds)]),
        }
    return out


def run_full_agent(questions: list[dict], agent: VerityAgent) -> tuple[dict, list[dict]]:
    em, f1, lat, flagged_any, low_conf, details = [], [], [], [], [], []
    for i, q in enumerate(questions):
        res = agent.answer(q["question"])
        em.append(exact_match(res.answer, q["answers"]))
        f1.append(f1_score(res.answer, q["answers"]))
        lat.append(res.latency_s)
        any_flag = any(s.flagged for s in res.sentences)
        flagged_any.append(1.0 if any_flag else 0.0)
        low_conf.append(1.0 if res.verdict == "LOW CONFIDENCE" else 0.0)
        details.append({
            "id": q["id"], "question": q["question"],
            "gold_answers": q["answers"], "answer": res.answer,
            "verdict": res.verdict,
            "sentences": [{"text": s.text, "citations": s.citations,
                           "score": s.score, "flagged": s.flagged,
                           "reason": s.reason} for s in res.sentences],
            "passage_ids": [p["id"] for p in res.passages],
            "latency_s": round(res.latency_s, 2),
            "em": em[-1], "f1": round(f1[-1], 3),
        })
        print(f"  [{i + 1}/{len(questions)}] {res.verdict} "
              f"({res.latency_s:.1f}s) F1={f1[-1]:.2f}")
    metrics = {
        "em": mean(em), "f1": mean(f1),
        "pct_answers_flagged": 100.0 * mean(flagged_any),
        "pct_low_confidence": 100.0 * mean(low_conf),
        "mean_latency_s": mean(lat),
    }
    return metrics, details


def run_eval(n_retrieval: int = 100, n_full: int = 10,
             out_path: Path | None = None) -> dict:
    if not CORPUS_PATH.exists() or not QUESTIONS_PATH.exists():
        ingest()
    corpus = load_corpus()
    questions = load_questions()
    print(f"Corpus: {len(corpus)} passages | Questions: {len(questions)}")

    retriever = HybridRetriever(corpus, cache_dir=DATA_DIR)
    retriever.build_dense_index()
    reranker = CrossEncoderReranker()
    llm = LocalBackend(num_threads=2)
    agent = VerityAgent(retriever, reranker, llm)

    ret_qs = sample_questions(questions, n_retrieval)
    print(f"\nRetrieval eval on {len(ret_qs)} questions ...")
    retrieval = run_retrieval_eval(ret_qs, retriever, reranker)

    full_qs = sample_questions(questions, n_full)
    print(f"\nFull-agent eval on {len(full_qs)} questions ...")
    agent_metrics, details = run_full_agent(full_qs, agent)
    # Retrieval metrics for the agent's own pipeline (hybrid+rerank path).
    retrieval["full_agent"] = retrieval["rerank"]

    results = {
        "n_retrieval": len(ret_qs), "n_full": len(full_qs),
        "n_passages": len(corpus),
        "retrieval": retrieval,
        "full_agent": agent_metrics,
        "details": details,
    }
    if out_path is None:
        out_path = RESULTS_DIR / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")
    return results


def print_table(results: dict) -> None:
    print("\n================ RETRIEVAL (n={}) ================".format(results["n_retrieval"]))
    print(f"{'config':<16}{'R@5':>8}{'R@20':>8}{'MRR@10':>8}")
    for name in ["bm25", "dense", "hybrid", "rerank"]:
        m = results["retrieval"][name]
        label = {"bm25": "BM25-only", "dense": "dense-only",
                 "hybrid": "hybrid (RRF)", "rerank": "hybrid+rerank"}[name]
        print(f"{label:<16}{m['recall@5']:>8.3f}{m['recall@20']:>8.3f}{m['mrr@10']:>8.3f}")
    fa = results["full_agent"]
    print("\n================ FULL AGENT (n={}) ================".format(results["n_full"]))
    print(f"Exact Match:            {fa['em']:.3f}")
    print(f"F1:                     {fa['f1']:.3f}")
    print(f"Answers flagged (>=1):  {fa['pct_answers_flagged']:.1f}%")
    print(f"Marked LOW CONFIDENCE:  {fa['pct_low_confidence']:.1f}%")
    print(f"Mean latency:           {fa['mean_latency_s']:.1f}s/question")
    print("Note: reranked lists contain 5 passages, so R@20 == R@5 there by construction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_retrieval", type=int, default=100)
    ap.add_argument("--n_full", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    results = run_eval(args.n_retrieval, args.n_full,
                       Path(args.out) if args.out else None)
    print_table(results)


if __name__ == "__main__":
    main()
