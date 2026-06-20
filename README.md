# Semantic Search RAG vs. Moment RAG

Two retrieval-augmented-generation pipelines built over the transcript of a single YouTube video,
then compared. Everything lives in one runnable notebook: **[`moment_rag.ipynb`](moment_rag.ipynb)**.

- **Part 1 — Baseline semantic-search RAG:** fixed-size token chunks → embeddings → cosine top-k → answer.
- **Part 2 — Moment RAG:** semantically-segmented **moments** with timestamps → ingestion enrichment
  (HyDE-style hypothetical questions, gist, keywords) → query **decomposition** → **hybrid** retrieval
  (dense text + dense questions + BM25, **RRF-fused**) → **cross-encoder re-rank** → moment rollup
  with **click-to-timestamp citations**.
- **Part 3 — Comparison & analysis.**
- **Appendix — ChromaDB:** the same baseline backed by a real vector database (HNSW index).

**Source video:** *What is a Vector Database? Powering Semantic Search & AI Applications* — https://www.youtube.com/watch?v=gl1r1XV0SLw

The Moment RAG design mirrors the course's `Moment_RAG/` reference codebase (query-side intelligence,
HyDE-at-ingestion, RRF fusion, cross-encoder rerank, moment rollup) on a lighter, **fully transparent
stack** (NumPy + `rank-bm25` + a small cross-encoder) instead of Qdrant/fastembed — so every retrieval
step is readable in the notebook.

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
cosine top-k  ── NumPy / ChromaDB         decompose query → dense + dense-Q + BM25 → RRF fuse
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
| Store in a vector DB / similarity index | In-memory NumPy cosine index **and** a real **ChromaDB** collection (Appendix) | `cosine_topk()`, `collection.add/query` |
| Accept queries, retrieve relevant chunks | top-k cosine retrieval | `baseline_rag()`, `baseline_rag_chroma()` |
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

## Vector storage — NumPy index vs. ChromaDB

The pipelines retrieve from an in-memory **NumPy** matrix (an exact brute-force cosine index). The
notebook's **appendix** additionally stores the same Part 1 embeddings in **ChromaDB** — a real
vector database — and serves retrieval from it via `baseline_rag_chroma()`, proving the pipeline
works against a production-style store, not just an in-process array.

### How ChromaDB is used

```python
import chromadb

chroma = chromadb.EphemeralClient()                       # in-memory; PersistentClient(path="data/chroma") to persist
try: chroma.delete_collection("baseline_chunks")          # idempotent on re-run
except Exception: pass
collection = chroma.create_collection(
    "baseline_chunks", metadata={"hnsw:space": "cosine"},  # cosine-distance HNSW index
)

collection.add(                                            # reuse Part 1 embeddings -> no extra API cost
    ids=[c["chunk_id"] for c in base_chunks],
    embeddings=base_matrix.tolist(),
    documents=[c["text"] for c in base_chunks],
    metadatas=[{"start_ms": c["start_ms"], "start": fmt_ts(c["start_ms"])} for c in base_chunks],
)

res = collection.query(query_embeddings=[qv], n_results=4) # ANN retrieval -> documents + distances
```

### Why ChromaDB (and why NumPy stays the default)

- **Covers both options of the brief.** The task says store embeddings in a *vector database **or**
  similarity index*. NumPy satisfies the similarity-index option; ChromaDB satisfies the
  vector-database option literally — so the project demonstrates both.
- **What a vector DB adds.** An **HNSW** approximate-nearest-neighbor index (sub-linear search vs.
  NumPy scanning every vector), on-disk **persistence**, metadata storage, and filtering — the things
  that matter once you outgrow one short transcript.
- **Why ChromaDB specifically.** Zero-config, pure-`pip` install, runs **embedded in-process** (no
  server to stand up), and a tiny `add`/`query` API — the lightest real vector DB to show the concept
  in a notebook. Swapping in Qdrant or Pinecone is the same shape.
- **Why NumPy is still the default.** On a 9–39 vector corpus, brute-force cosine is *exact*, instant,
  and fully transparent. ChromaDB returns the **same top result** here — distance `0.268` ↔ cosine
  `0.732`, matching the NumPy score — so it validates the baseline rather than changing it.
- **No extra cost.** Indexing reuses the embeddings already in `base_matrix`; only the query embedding
  is computed at search time.

> Chroma reports a **distance** (`cosine distance = 1 − cosine similarity`), so *lower = closer* —
> compare it to the cosine *scores* the NumPy baseline prints.

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
| `moment_rag.ipynb` | The project — Parts 1–3 + ChromaDB appendix (run with outputs) |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Template for your `OPENAI_API_KEY` |
| `.gitignore` | Excludes `.env`, `.venv/`, and the cached `data/` |

`data/` (the cached transcript) and `.env` (your key) are **not** committed — the notebook
re-fetches the transcript on first run.

---

## Design notes

- **NumPy vs. a vector database.** See *[Vector storage — NumPy index vs. ChromaDB](#vector-storage--numpy-index-vs-chromadb)*
  above: brute-force cosine for transparency on a tiny corpus, with ChromaDB (HNSW) as the production path.
- **HyDE lives at ingestion.** Each moment is pre-tagged with the questions it answers, embedded as a
  separate retrieval branch — so a user's question can match a stored *question*, keeping query-time
  latency low. This mirrors the reference codebase.
- **Hybrid + RRF.** Dense vectors catch paraphrase; BM25 catches exact terms (e.g. "cosine", "HNSW");
  Reciprocal Rank Fusion blends the rankings so neither dominates.
- **Tunable knobs:** `USE_RERANK`, the segmentation `percentile`/`min_tokens` in `segment_moments`,
  `top_n`, and `k` in retrieval.
