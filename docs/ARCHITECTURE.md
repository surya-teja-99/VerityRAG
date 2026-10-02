# VerityRAG architecture

How the pieces fit together and why they are built this way.

## Pipeline

```
question
  → decompose (heuristic split on conjunctions / ";")
  → hybrid retrieval per sub-question (BM25 + dense, RRF fusion, top-30)
  → merge + dedupe
  → cross-encoder rerank (top-30 → top-5)
  → answer synthesis with [n] citations (local instruction model)
  → verifier: every sentence scored against its cited passages
  → VERIFIED / PARTIALLY VERIFIED / LOW CONFIDENCE
```

## Components

| Module | Responsibility | Key choice |
|---|---|---|
| `ingest.py` | Download SQuAD v1.1 dev, dedupe contexts into ~2k passages | Deterministic; cached under `data/` |
| `retrieval.py` | BM25 (`rank-bm25`) + dense (`all-MiniLM-L6-v2`) fused with Reciprocal Rank Fusion (k=60) | Hybrid beats either alone on Recall@5 (0.91 vs 0.87 / 0.85) |
| `rerank.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks top-30 → top-5 | Inputs truncated to 800 chars: ~8x faster on CPU, high rank agreement with full text |
| `llm.py` | Pluggable backends | Default: `SmolLM2-360M-Instruct` on CPU (greedy, 64 tokens). `OpenAICompatibleBackend` (e.g. Azure OpenAI) documented for stronger generation |
| `agent.py` | Decompose → retrieve → rerank → synthesize → verify | Verifier flags sentences with no citation or cross-encoder support < 0.35 sigmoid |
| `eval.py` / `metrics.py` | Ablations + SQuAD EM/F1 + verifier stats | Retrieval on 100 q, full agent on 10 q (`make eval` scales up) |

## Design decisions

- **Verifier as a separate pass, not a prompt trick.** Citation compliance is
  checked mechanically after generation, so a weak generator can't talk its
  way past verification. On the demo run the 360M model ignored citation
  instructions and the verifier correctly marked 100% of answers LOW
  CONFIDENCE — the failure is visible instead of silent.
- **RRF instead of score fusion.** BM25 and dense scores live on different
  scales; rank fusion needs no calibration.
- **Cached dense embeddings** (`data/embeddings.npy`) so reruns skip the
  slowest CPU step.
- **Small, honest eval.** 100 retrieval / 10 full-agent questions keeps
  `make demo` under 10 minutes on CPU; limitations are documented in the
  README rather than hidden.

## Known trade-offs

- Heuristic decomposition is a no-op for single-clause questions (most of SQuAD).
- The corpus is the SQuAD dev paragraphs themselves, so this measures
  retrieval/rerank quality, not open-domain generalization.
- Verifier threshold (0.35) is a reasonable default, not tuned on this data.
