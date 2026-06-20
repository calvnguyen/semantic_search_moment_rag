# Semantic Search RAG vs. Moment RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/calvnguyen/semantic_search_moment_rag/blob/main/moment_rag.ipynb)

Two retrieval-augmented-generation pipelines built over the transcript of a single YouTube video,
then compared. Everything lives in one runnable notebook: **[`moment_rag.ipynb`](moment_rag.ipynb)**.

**Run it instantly:** click the **Open in Colab** badge → *Runtime → Run all*. The first cell
installs dependencies, pulls the cached transcript, and prompts for your `OPENAI_API_KEY`.

**Slides:** a 25-slide walkthrough of the build and comparison is in
[`presentation.pdf`](presentation.pdf) (open [`presentation.html`](presentation.html) to present;
press **S** for speaker notes).

- **Part 1 — Baseline semantic-search RAG:** fixed-size token chunks → embeddings → cosine top-k → answer.
- **Part 2 — Moment RAG:** semantically-segmented **moments** with timestamps → ingestion enrichment
  (HyDE-style hypothetical questions, gist, keywords) → query **decomposition** → **hybrid** retrieval
  (dense text + dense questions + BM25, **RRF-fused**) → **cross-encoder re-rank** → moment rollup
  with **click-to-timestamp citations**.
- **Part 3 — Comparison & analysis.**
- **Appendix — Qdrant:** a real **server-side hybrid** (dense + BM25 sparse, RRF-fused *inside the database*).

**Source video:** *What is a Vector Database? Powering Semantic Search & AI Applications* — https://www.youtube.com/watch?v=gl1r1XV0SLw

The Moment RAG design mirrors the course's `Moment_RAG/` reference codebase (query-side intelligence,
HyDE-at-ingestion, RRF fusion, cross-encoder rerank, moment rollup). The core pipelines run on a
lighter, **fully transparent stack** (NumPy + `rank-bm25` + a small cross-encoder) so every retrieval
step is readable — and the appendix reproduces the hybrid on the reference's own stack
(**Qdrant + fastembed**).

---

## Why two pipelines?

A baseline RAG cuts the transcript into arbitrary fixed windows and does a single embedding lookup.
Moment RAG moves the intelligence to the **query side** and makes retrieval land on coherent,
timestamped *moments*. The contrast is the whole point:

```
BASELINE                                  MOMENT RAG
transcript                                transcript
   │                                         │
fixed ~256-token chunks                   semantic moments  (start_ms/end_ms)
   │                                         │  + enrich: hypothetical questions, gist, keywords
embed (OpenAI)                            embed text  +  embed questions (HyDE)  +  BM25
   │                                         │
cosine top-k  ── NumPy (exact)            decompose query → dense + dense-Q + BM25 → RRF fuse
   │                                         │
                                          cross-encoder re-rank
   │                                         │
gpt-4o-mini answer                        gpt-4o-mini cited answer  +  &t= timestamp deep-links
```

---

## What each pipeline does

**Part 1 — baseline semantic-search RAG**

| Capability | How it works | Key code |
|---|---|---|
| Obtain the transcript from a YouTube video | `youtube-transcript-api` → 109 timed cues, cached to `data/transcript.json` | `fetch_transcript()` |
| Split into chunks | ~256-token windows, 25% overlap | `fixed_chunks()` |
| Generate embeddings | OpenAI `text-embedding-3-small` → `(9, 1536)` matrix | `embed_texts()` |
| Store in a vector DB / similarity index | In-memory NumPy cosine index (the Appendix adds a real **Qdrant** hybrid) | `cosine_topk()`; Qdrant in Appendix |
| Accept queries, retrieve relevant chunks | top-k cosine retrieval | `baseline_rag()` |
| Generate an answer from context | context-grounded `gpt-4o-mini` synthesis | `baseline_rag()` |

**Part 2 — Moment RAG**

| Capability | How it works | Key code |
|---|---|---|
| Identify meaningful "moments" | semantic-breakpoint segmentation → 10 moments with real timestamps | `segment_moments()` |
| Structure segments around moments | per-moment enrichment (hypothetical questions / gist / keywords); indexed by text + question vectors + BM25 | `enrich_moment()` |
| Improve retrieval with moment-level context | decompose → dense + dense-questions + BM25 → RRF → cross-encoder rerank | `moment_retrieve()`, `rerank()`, `moment_rag()` |
| Compare against the baseline | 3-query head-to-head + summary table | Part 3, `unit_summary()` |
| Explain how Moment RAG improves answers | written analysis of each mechanism + tradeoffs | "How Moment RAG changes answer quality" |

---

## Headline result

On *"What is an embedding?"* the **baseline failed** — *"The context does not explicitly define what
an embedding is"* — because its fixed chunks split the definition away from the cue that names it.
**Moment RAG answered correctly** — *"an array of numbers that captures the semantic essence of data,
where similar items sit close together in vector space"* — with six timestamped citations, because
query **decomposition** + **HyDE question-indexing** surfaced the right moment. The multi-faceted Q3
shows the same effect: the baseline answers one facet, Moment RAG covers both.

---

## Vector storage — NumPy (default) vs. Qdrant (server-side hybrid)

The core pipelines retrieve from in-memory **NumPy** + `rank-bm25`, so every step is readable — the
cosine math (`M @ q`) and the RRF fusion are right there in the notebook. The **appendix** then
reproduces Moment RAG's hybrid on **Qdrant**, a real vector database run **in-memory**
(`QdrantClient(":memory:")` — no server, works on Colab), which does the one thing an in-process
array can't: **dense + BM25-sparse retrieval with RRF fusion _inside the database_**.

### How Qdrant is used

```python
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

bm25 = SparseTextEmbedding("Qdrant/bm25")
qc = QdrantClient(":memory:")                              # in-process; path="data/qdrant" to persist
qc.create_collection(
    "moments_hybrid",
    vectors_config={"text": models.VectorParams(size=1536, distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
qc.upsert("moments_hybrid", points=[...])                 # reuse Part 2 moment_vecs + BM25 sparse

res = qc.query_points(                                     # dense + sparse, fused server-side
    "moments_hybrid",
    prefetch=[
        models.Prefetch(query=dense_q,  using="text", limit=8),
        models.Prefetch(query=sparse_q, using="bm25", limit=8),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),   # <- RRF happens in the DB
    limit=8,
)
```

### Why NumPy is the default — and why Qdrant for the hybrid

- **Transparency first.** On a 9–39 vector corpus, NumPy brute-force cosine is *exact*, instant, and
  fully readable. That clarity is the point of the core notebook.
- **Qdrant earns its place by doing real hybrid.** It stores dense + sparse vectors together and fuses
  them with RRF **server-side** — the architecture the course's `Moment_RAG/` reference uses, and what
  you'd actually run at scale (plus persistence, metadata filters, millions of vectors).
- **Still Colab-friendly.** `QdrantClient(":memory:")` runs in-process — no server to deploy.
- **What stays in Python.** Query **decomposition** (LLM) and the **cross-encoder re-rank** — Qdrant
  stores and fuses; it doesn't decompose or cross-encode.
- **No extra embedding cost.** The Qdrant index reuses the Part 2 `moment_vecs`; only the BM25 sparse
  vectors are added (computed locally by `fastembed`).

---

## Setup & run

Requires **Python 3.12** (some ML wheels — torch, onnxruntime — don't yet ship 3.14 builds).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name semantic-moment-rag --display-name "Python 3.12 (moment-rag)"

cp .env.example .env        # then paste your OPENAI_API_KEY  (https://platform.openai.com/api-keys)

jupyter lab moment_rag.ipynb   # select the "Python 3.12 (moment-rag)" kernel → Run All
```

The first Moment RAG run downloads a small cross-encoder model (`ms-marco-MiniLM-L-6-v2`, ~80 MB,
one-time). The cross-encoder and BM25 run **locally** (free); only embeddings and `gpt-4o-mini`
hit the API.

### Cost

A full top-to-bottom run is **~1–3 cents**: embeddings (`text-embedding-3-small`, $0.02/1M tokens)
plus `gpt-4o-mini` calls (one enrichment call per moment + decomposition + synthesis). OpenAI
requires a one-time **$5 minimum** account top-up to enable the API.

---

## Repository layout

| Path | Purpose |
|---|---|
| `moment_rag.ipynb` | The project — Parts 1–3 + Qdrant hybrid appendix (run with outputs) |
| `presentation.html` / `presentation.pdf` | 25-slide deck explaining the build & comparison (dark theme, speaker notes) |
| `data/transcript.json` | Cached transcript, committed so the Colab demo runs without a live YouTube fetch |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Template for your `OPENAI_API_KEY` |
| `.gitignore` | Excludes `.env`, `.venv/`, and `data/` (except the committed transcript) |

`.env` (your key) and the rest of `data/` are **not** committed. The cached `data/transcript.json`
**is** committed so the Colab demo works even when YouTube blocks the runtime's IP; locally the
notebook reuses it (or fetches live if it's absent).

---

## Design notes

- **NumPy vs. a vector database.** See *[Vector storage — NumPy vs. Qdrant](#vector-storage--numpy-default-vs-qdrant-server-side-hybrid)*
  above: brute-force cosine for transparency on a tiny corpus, with Qdrant (server-side dense+BM25 RRF) as the production path.
- **HyDE lives at ingestion.** Each moment is pre-tagged with the questions it answers, embedded as a
  separate retrieval branch — so a user's question can match a stored *question*, keeping query-time
  latency low. This mirrors the reference codebase.
- **Hybrid + RRF.** Dense vectors catch paraphrase; BM25 catches exact terms (e.g. "cosine", "HNSW");
  Reciprocal Rank Fusion blends the rankings so neither dominates.
- **Tunable knobs:** `USE_RERANK`, the segmentation `percentile`/`min_tokens` in `segment_moments`,
  `top_n`, and `k` in retrieval.
