# 🔍 Day 3 — Hybrid Search: BM25 + Vector Retrieval

> Part of the **RAG From Scratch** learning series  
> Day 1: Basic RAG Pipeline · Day 2: Chunking & Sentence Transformers · **Day 3: Hybrid Search** · Day 4: Re-ranking *(coming soon)*

---

## Objectives

- Why neither keyword search nor semantic search is enough on its own
- How BM25 ranks documents using term frequency and inverse document frequency
- How dense vector embeddings enable semantic, concept-level retrieval
- How to fuse both signals using **Reciprocal Rank Fusion (RRF)**
- How to evaluate retrieval quality with Precision, Recall, MRR, and HitRate

---

## The Core Problem

Neither approach is perfect in isolation:

| Scenario | BM25 (Keyword) | Vector (Semantic) |
|---|---|---|
| Query: `"BM25 k1 parameter"` | ✅ Exact term match wins | ✅ Also handles it |
| Query: `"how neural networks learn"` | ❌ Misses if phrasing differs | ✅ Understands the concept |
| Query: `"sparse index for fast lookup"` | ⚠️ Partial match | ✅ Semantic overlap |
| Rare product code / error message | ✅ IDF highly rewards rare terms | ❌ May dilute in embedding space |

**Hybrid search gets the best of both worlds.**

---

## Project Structure

```
day3-hybrid-search/
├── main.py                  # End-to-end demo — run this
├── requirements.txt
├── data/
│   └── documents.py         # 15-document corpus (varied topics)
└── src/
    ├── bm25_search.py       # Keyword retrieval using BM25
    ├── vector_search.py     # Semantic retrieval using dense embeddings
    ├── hybrid_search.py     # RRF + Linear Combination fusion
    └── evaluation.py        # Precision@K, Recall@K, MRR, HitRate
```

---

## Quickstart

```bash
# 1. Clone / navigate to the folder
cd hybrid-search

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

**Requirements:**
```
rank-bm25==0.2.2
sentence-transformers>=2.7.0   # or: scikit-learn + numpy (lightweight fallback)
numpy>=1.24.0
```

> **No GPU?** The code includes a drop-in lightweight fallback using `TF-IDF + Truncated SVD (LSA)` from scikit-learn. Same API, no torch needed. Swap in `SentenceTransformer` when you have GPU for production-quality embeddings.

---

## How It Works

### 1. BM25 — Keyword Retrieval

BM25 scores each document for a query using three ideas:

```
score(doc, query) = Σ  IDF(term) × TF_saturated(term, doc)
```

- **IDF**: rare terms score higher than common ones (`"BM25"` > `"the"`)
- **TF saturation**: the 10th occurrence of a word is worth much less than the 1st — controlled by `k1`
- **Length normalisation**: long documents don't unfairly dominate — controlled by `b`

```python
from src.bm25_search import BM25Retriever

retriever = BM25Retriever(k1=1.5, b=0.75)
retriever.index(documents)
results = retriever.search("BM25 k1 parameter term frequency", top_k=5)

# Debug: see which terms contributed most
explanation = retriever.explain("BM25 k1 parameter", doc_id="doc_002")
```

**When BM25 wins:** exact technical terms, product codes, error messages, rare domain-specific vocabulary.

---

### 2. Vector Search — Semantic Retrieval

Documents and queries are encoded as dense vectors. Retrieval = finding the nearest vectors by cosine similarity.

```
cosine_similarity(q, d) = (q · d) / (|q| × |d|)
```

Similar *meaning* clusters together in vector space, even when the exact words differ.

```python
from src.vector_search import VectorRetriever

retriever = VectorRetriever(model_name="all-MiniLM-L6-v2")
retriever.index(documents)
results = retriever.search("how neural networks learn during training", top_k=5)
```

**When vector search wins:** paraphrases, synonyms, conceptual questions, cross-lingual queries.

---

### 3. Hybrid Fusion — Combining Both Signals

#### Reciprocal Rank Fusion (RRF) ✅ Recommended

```
rrf_score(doc) = Σ  1 / (k + rank_in_list)
```

Each retriever contributes `1 / (60 + rank)` per document. Documents that rank high in **both** lists dominate. `k=60` prevents a single top-ranked result from drowning everything else.

**Why RRF is great:**
- No score normalisation needed (rank-based, not score-based)
- Works with 2, 3, or N retrievers without tuning
- Robust to outlier scores

#### Linear Combination

```
final_score = α × norm_vector_score + (1 - α) × norm_bm25_score
```

Both scores are min-max normalised before combining. Use `alpha=0.7` to trust semantic search more, `alpha=0.3` to trust keyword search more.

```python
from src.hybrid_search import HybridRetriever

# RRF fusion (recommended)
retriever = HybridRetriever(fusion="rrf")
retriever.index(documents)
output = retriever.search("retrieval augmented generation LLM context", top_k=5)

print(output["bm25_results"])    # keyword results
print(output["vector_results"])  # semantic results
print(output["hybrid_results"])  # merged results

# Linear fusion
retriever = HybridRetriever(fusion="linear", alpha=0.7)
```

---

### 4. Evaluation Metrics

```python
from src.evaluation import compare_retrievers, print_comparison_table

comparison = compare_retrievers(
    retriever_fns={
        "BM25":   lambda q: bm25_ret.search(q, top_k=5),
        "Vector": lambda q: vector_ret.search(q, top_k=5),
        "Hybrid": lambda q: hybrid_ret.search(q, top_k=5)["hybrid_results"],
    },
    test_queries=test_queries,
    k=5,
)
print_comparison_table(comparison)
```

| Metric | Measures |
|---|---|
| **Precision@K** | Of the top-K retrieved docs, how many are relevant? |
| **Recall@K** | Of all relevant docs, how many did we find in top-K? |
| **MRR** | `1 / rank_of_first_relevant_doc` — rewards finding the best answer early |
| **HitRate@K** | Binary: did *any* relevant doc appear in top-K? |

---

## Sample Output

```
══════════════════════════════════════════════════════════════════════
  STEP 2: Query Comparisons
══════════════════════════════════════════════════════════════════════

  Query: "how neural networks learn during training"
  [Semantic-friendly: conceptual paraphrase]

  ── BM25 (Keyword) ──
    [1] BM25=4.775    | doc_006 | Gradient Descent Optimization
    [2] BM25=3.806    | doc_014 | Attention Mechanism in Deep Learning
    [3] BM25=3.206    | doc_001 | Introduction to Transformers

  ── Vector (Semantic) ──
    [1] Vec=0.865     | doc_014 | Attention Mechanism in Deep Learning
    [2] Vec=0.479     | doc_006 | Gradient Descent Optimization
    [3] Vec=0.267     | doc_001 | Introduction to Transformers

  ── Hybrid (RRF) ──
    [1] RRF=0.032522  | doc_006 | Gradient Descent Optimization   ← consensus #1
    [2] RRF=0.032522  | doc_014 | Attention Mechanism in Deep Learning
    [3] RRF=0.031746  | doc_001 | Introduction to Transformers

══════════════════════════════════════════════════════════════════════
Retriever      Precision@5      Recall@5           MRR     HitRate@5
══════════════════════════════════════════════════════════════════════
BM25                0.3200        0.8667        1.0000        1.0000
Vector              0.3200        0.8667        1.0000        1.0000
Hybrid              0.3200        0.8667        1.0000        1.0000
══════════════════════════════════════════════════════════════════════
```

---

## When to Use Each Retriever

```
┌─────────────────────┬──────────────────────────────────┬────────────────────────────────┐
│                     │ BM25 (Keyword)                   │ Vector (Semantic)              │
├─────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ Strengths           │ Exact term match                 │ Synonyms & paraphrases         │
│                     │ Rare/specific terms              │ Conceptual similarity          │
│                     │ Fast, no GPU required            │ Cross-lingual queries          │
├─────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ Weaknesses          │ Vocabulary mismatch              │ Can miss exact keywords        │
│                     │ No semantic understanding        │ Requires GPU for large corpora │
└─────────────────────┴──────────────────────────────────┴────────────────────────────────┘

→ Hybrid (RRF) is the default choice for production RAG systems.
```

---

## Production Upgrade Path

The `VectorRetriever` in this repo uses a lightweight TF-IDF + LSA fallback. To upgrade to full sentence embeddings:

```python
# In src/vector_search.py, swap _build_embeddings():

# BEFORE (lightweight, no GPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# AFTER (production quality)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)
```

Everything else — the fusion logic, evaluation, and `HybridRetriever` API — stays **identical**.

For large-scale deployment, consider:
- [FAISS](https://github.com/facebookresearch/faiss) — approximate nearest neighbour for millions of vectors
- [Elasticsearch](https://www.elastic.co/) — built-in BM25 + kNN hybrid search
- [Qdrant](https://qdrant.tech/) / [Weaviate](https://weaviate.io/) — vector DBs with hybrid search support


---

## References

- [BM25: Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion — Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Sentence-Transformers](https://www.sbert.net/)
- [rank-bm25 library](https://github.com/dorianbrown/rank_bm25)
