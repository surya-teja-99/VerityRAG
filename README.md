# VerityRAG

An **agentic retrieval-augmented generation (RAG)** pipeline with a **verifier agent**,
evaluated honestly on the SQuAD v1.1 dev set. It retrieves with hybrid search
(BM25 + dense embeddings fused by Reciprocal Rank Fusion), reranks with a
cross-encoder, synthesizes a cited answer with a small local instruction model,
and then *verifies* every cited sentence against its sources — flagging
unsupported claims instead of letting them through silently.

## Architecture

```
 ┌────────────┐
 │  Question  │
 └─────┬──────┘
       ▼
 ┌─────────────┐   heuristic: split on
 │  Decompose  │── conjunctions ("and"/"or"/";")
 └─────┬───────┘
       ▼
 ┌─────────────────────────────────┐
 │  Hybrid retrieval (per sub-q)   │◄── BM25 (rank-bm25)
 │  BM25 + dense, fused with RRF   │◄── dense: all-MiniLM-L6-v2 (cached .npy)
 └─────────────┬───────────────────┘
               ▼  merge + dedupe
 ┌─────────────────────────────┐
 │  Rerank top-30 → top-5      │◄── cross-encoder ms-marco-MiniLM-L-6-v2
 └─────────────┬───────────────┘
               ▼
 ┌─────────────────────────────┐
 │  Synthesize answer + [n]    │◄── SmolLM2-360M-Instruct (CPU, greedy)
 │  citations                  │
 └─────────────┬───────────────┘
               ▼
 ┌─────────────────────────────┐
 │  Verifier agent             │── cross-encoder scores each sentence
 │  VERIFIED / PARTIALLY       │   vs. its cited passages; below-threshold
 │  VERIFIED / LOW CONFIDENCE  │   or uncited sentences are flagged
 └─────────────────────────────┘
```

## Quickstart

```bash
make setup   # install deps, download SQuAD dev, cache embeddings
make test    # pytest unit tests
make demo    # 2 sample Q&A + retrieval ablations (100 q) + full agent (10 q); <10 min on CPU
make eval    # larger eval: 500 retrieval + 50 full-agent questions (slower)
```

`make demo` saves machine-readable results to `results/demo_results.json`.

## Results

Measured with `make demo` on 2026-10-02 on this machine (CPU-only, 2 threads):
100 SQuAD v1.1 dev questions for retrieval, 10 for the full agent.
Total demo wall-clock: **425 s (7.1 min)**. No numbers below are invented —
they are the literal output of the demo run (`results/demo_results.json`).

### Retrieval ablations

| config | Recall@5 | Recall@20 | MRR@10 |
|---|---|---|---|
| BM25-only | 0.870 | 0.940 | 0.795 |
| dense-only | 0.850 | 0.940 | 0.726 |
| hybrid (RRF) | 0.910 | 0.970 | 0.811 |
| hybrid + rerank (top-5) | 0.920 | 0.920 | 0.874 |

Hybrid fusion beats either retriever alone, and the cross-encoder rerank
lifts MRR@10 from 0.811 → 0.874.

### Full agent

| metric | value |
|---|---|
| Exact Match | 0.000 |
| F1 | 0.086 |
| Answers with ≥1 flagged sentence | 100.0% |
| Marked LOW CONFIDENCE | 100.0% |
| Mean latency | 10.6 s/question |

### Sample output

Actual `make demo` output (trimmed):

```
Q: Which NFL team represented the AFC at Super Bowl 50?
A: The AFC team that represented the AFC at Super Bowl 50 was the Denver Broncos.
Verdict: LOW CONFIDENCE | EM=0 F1=0.29 | 27.2s
  [FLAGGED score=n/a] ... cites=[] no citation
```

The answer is factually right ("Denver Broncos") and the gold passage was
retrieved at rank 1 — but the 360M model emitted no `[n]` citation, so the
verifier flagged the sentence as uncited and the answer as LOW CONFIDENCE.
That is the verifier working as designed: it catches exactly the failure
mode small generators have.

## Limitations (read before judging the numbers)

- **CPU-only, tiny generator.** Answer synthesis uses SmolLM2-360M-Instruct on
  CPU with greedy decoding and 64 new tokens. On the 10-question demo sample
  it scored EM 0.000 / F1 0.086 — that reflects the model's size, not the
  retrieval quality. Retrieval metrics (n=100) are the fair comparison.
- **The generator ignores citation instructions.** SmolLM2-360M almost never
  emits `[n]` citations despite being told to, so the verifier flags nearly
  every sentence as uncited (100% LOW CONFIDENCE on the demo sample). The
  verifier is behaving correctly; a stronger generator (or the documented
  `OpenAICompatibleBackend`) would change this picture.
- **Rerank inputs are truncated to 800 chars** for CPU speed (full paragraphs
  make the cross-encoder ~8x slower on this machine). Measured ranking
  agreement vs. full-text reranking is high but not perfect — a documented
  speed/quality trade-off.
- **Heuristic decomposition.** Sub-question splitting is a regex over
  conjunctions, not a learned planner. It is a no-op for single-clause
  questions (most of SQuAD).
- **SQuAD-only eval.** The corpus is the SQuAD v1.1 dev paragraphs themselves,
  so the gold passage is always in the index. This measures retrieval/rerank
  quality, not open-domain generalization.
- **Verifier threshold is a heuristic** (sigmoid(cross-encoder logit) < 0.35),
  chosen as a reasonable default, not tuned on this data.
- **Small eval sizes** (100 retrieval / 10 full-agent) keep the CPU demo under
  10 minutes; use `make eval` for tighter estimates.

## Project layout

```
VerityRAG/
├── verityrag/
│   ├── ingest.py      # download SQuAD v1.1 dev, dedupe into passage corpus
│   ├── retrieval.py   # BM25 + dense (MiniLM) fused with Reciprocal Rank Fusion
│   ├── rerank.py      # cross-encoder reranking top-30 → top-5
│   ├── llm.py         # pluggable backends: local SmolLM2 (default) / OpenAI-compatible
│   ├── agent.py       # decompose → retrieve → rerank → synthesize → verify
│   ├── eval.py        # ablations + SQuAD EM/F1 + verifier stats
│   └── metrics.py     # SQuAD normalization, EM/F1, recall/MRR
├── scripts/
│   ├── demo.py        # sample Q&A + metrics, <10 min on CPU
│   └── build_index.py # one-time ingest + embedding cache
├── tests/             # pytest: metrics, RRF, citations, verifier, ingest
├── data/              # corpus, questions, cached embeddings (git-ignored)
└── results/           # eval JSON output (git-ignored)
```

## License

All rights reserved.
