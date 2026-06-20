# Semantic Search RAG vs. Moment RAG

Two retrieval-augmented pipelines built over the transcript of a single YouTube video, then
compared — the Module 3 (Agentic RAG) assignment. Everything lives in **`moment_rag.ipynb`**.

- **Part 1 — Baseline semantic-search RAG:** fixed-size token chunks → OpenAI embeddings →
  cosine top-k → answer.
- **Part 2 — Moment RAG:** semantically-segmented **moments** with timestamps → ingestion
  enrichment (HyDE-style hypothetical questions, gist, keywords) → query **decomposition** →
  **hybrid** retrieval (dense text + dense questions + BM25, **RRF-fused**) → **cross-encoder
  re-rank** → moment rollup with **click-to-timestamp citations**.
- **Part 3 — Comparison & analysis.**

The Moment RAG design mirrors the `Moment_RAG/` reference codebase from the course (query-side
intelligence, HyDE-at-ingestion, RRF fusion, cross-encoder rerank, moment rollup) on a lighter,
fully transparent stack (numpy + `rank-bm25` + a small cross-encoder) instead of Qdrant/fastembed.

**Video:** *How does a Vector Database work?* — https://www.youtube.com/watch?v=VVNYQKDLY5s

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name semantic-moment-rag --display-name "Python 3.12 (moment-rag)"

cp .env.example .env        # then paste your OPENAI_API_KEY (https://platform.openai.com/api-keys)

jupyter lab moment_rag.ipynb   # select the "Python 3.12 (moment-rag)" kernel, run top-to-bottom
```

Requires **Python 3.12** (the ML wheels don't all have 3.14 builds yet). The first Moment RAG run
downloads a small cross-encoder model (~80 MB, one-time). Each run makes OpenAI calls
(embeddings + `gpt-4o-mini`) — a video this size costs a few cents.
