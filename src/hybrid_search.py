"""
hybrid_search.py
----------------
Hybrid retrieval: combine BM25 (keyword) + Vector (semantic) search results.

WHY HYBRID?
-----------
Neither BM25 nor vector search alone is perfect:

  ┌─────────────────────┬─────────────────────────────────┬────────────────────────────────┐
  │                     │ BM25 (Keyword)                  │ Vector (Semantic)              │
  ├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
  │ Strengths           │ Exact term match                │ Synonyms & paraphrases         │
  │                     │ Rare/specific terms (BM25 code) │ Conceptual similarity          │
  │                     │ No model needed                 │ Cross-lingual (multilingual)   │
  ├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
  │ Weaknesses          │ Can't handle synonyms           │ Can miss exact keywords        │
  │                     │ Vocabulary mismatch             │ Computationally expensive      │
  │                     │ No semantic understanding       │ Requires GPU for large corpora │
  └─────────────────────┴─────────────────────────────────┴────────────────────────────────┘

Hybrid search gets the best of both worlds.

FUSION STRATEGIES implemented here:
  1. Reciprocal Rank Fusion (RRF)       — parameter-free, robust ✅ recommended
  2. Normalised Linear Combination      — weighted α * vector + (1-α) * bm25
"""

from __future__ import annotations

from src.bm25_search import BM25Retriever
from src.vector_search import VectorRetriever


# ---------------------------------------------------------------------------
# Fusion functions
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_field: str = "id",
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF).

    Algorithm (Cormack et al., 2009):
        For each document d in each ranked list l:
            rrf_score(d) += 1 / (k + rank(d, l))
        Final ranking = sort by rrf_score descending.

    Parameters
    ----------
    ranked_lists : list of ranked result lists (each from a different retriever)
    id_field     : key to use as document identifier
    k            : smoothing constant (default 60 — from original paper)

    Returns
    -------
    Merged list of dicts, each with an added 'rrf_score' key.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank_position, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_field]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_position)
            # Store document, preferring the version with more metadata
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

    # Build merged result list
    merged = []
    for rank, (doc_id, rrf_score) in enumerate(
        sorted(scores.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        doc = dict(doc_lookup[doc_id])
        doc["rrf_score"] = round(rrf_score, 6)
        doc["rank"] = rank
        # Remove individual rank fields to avoid confusion
        doc.pop("rank", None)
        merged.append(doc)

    # Re-add final rank
    for i, doc in enumerate(merged, start=1):
        doc["rank"] = i

    return merged


def linear_combination_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    all_doc_ids: list[str],
    alpha: float = 0.5,
    id_field: str = "id",
) -> list[dict]:
    """
    Normalised score combination:
        final_score = alpha * norm_vector_score + (1 - alpha) * norm_bm25_score

    Both score lists are min-max normalised to [0, 1] before combining.

    Parameters
    ----------
    alpha : float in [0, 1]
        Weight for vector score (1 - alpha goes to BM25).
        alpha=0.7 → trust semantic search more
        alpha=0.3 → trust keyword search more
    """
    # Build score dicts
    bm25_scores  = {d[id_field]: d.get("bm25_score",   0.0) for d in bm25_results}
    vector_scores = {d[id_field]: d.get("vector_score", 0.0) for d in vector_results}

    # Min-max normalise BM25 scores
    bm25_vals = list(bm25_scores.values())
    bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
    bm25_range = bm25_max - bm25_min or 1.0

    # Min-max normalise vector scores
    vec_vals = list(vector_scores.values())
    vec_min, vec_max = min(vec_vals), max(vec_vals)
    vec_range = vec_max - vec_min or 1.0

    # Build merged lookup (combine all docs seen in either list)
    all_docs: dict[str, dict] = {}
    for d in bm25_results + vector_results:
        all_docs[d[id_field]] = d

    combined = []
    for doc_id, doc in all_docs.items():
        norm_bm25   = (bm25_scores.get(doc_id,   0.0) - bm25_min) / bm25_range
        norm_vector = (vector_scores.get(doc_id, 0.0) - vec_min)  / vec_range
        final_score = alpha * norm_vector + (1 - alpha) * norm_bm25

        merged_doc = dict(doc)
        merged_doc["linear_score"] = round(final_score, 6)
        combined.append(merged_doc)

    combined.sort(key=lambda x: x["linear_score"], reverse=True)
    for i, doc in enumerate(combined, start=1):
        doc["rank"] = i

    return combined


# ---------------------------------------------------------------------------
# HybridRetriever — the main class that ties everything together
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    End-to-end hybrid retrieval pipeline.

    Usage
    -----
        retriever = HybridRetriever()
        retriever.index(documents)
        results = retriever.search("how does attention work?", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        bm25_k1: float = 1.5,
        bm25_b:  float = 0.75,
        fusion:  str   = "rrf",   # "rrf" | "linear"
        alpha:   float = 0.5,     # only used when fusion="linear"
        rrf_k:   int   = 60,      # only used when fusion="rrf"
    ):
        self.fusion  = fusion
        self.alpha   = alpha
        self.rrf_k   = rrf_k
        self.documents: list[dict] = []

        self.bm25_retriever   = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.vector_retriever = VectorRetriever(model_name=model_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: list[dict]) -> None:
        """Index documents in both BM25 and vector retrievers."""
        self.documents = documents
        self.bm25_retriever.index(documents)
        self.vector_retriever.index(documents)
        print(f"\n[Hybrid] Ready. Fusion strategy: {self.fusion.upper()}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int | None = None,   # how many to fetch from each sub-retriever
    ) -> dict:
        """
        Run hybrid retrieval and return merged results.

        Parameters
        ----------
        fetch_k : int, optional
            How many results to fetch from each sub-retriever before fusion.
            Defaults to max(top_k * 3, 10) to ensure good recall.

        Returns
        -------
        dict with keys:
            'query'          : the original query
            'bm25_results'   : top-k from BM25 alone
            'vector_results' : top-k from vector search alone
            'hybrid_results' : merged top-k results
        """
        if fetch_k is None:
            fetch_k = max(top_k * 3, 10)

        # Run both retrievers
        bm25_results   = self.bm25_retriever.search(query,   top_k=fetch_k)
        vector_results = self.vector_retriever.search(query, top_k=fetch_k)

        # Fuse results
        if self.fusion == "rrf":
            hybrid_results = reciprocal_rank_fusion(
                [bm25_results, vector_results],
                k=self.rrf_k,
            )[:top_k]
        elif self.fusion == "linear":
            all_ids = [d["id"] for d in self.documents]
            hybrid_results = linear_combination_fusion(
                bm25_results,
                vector_results,
                all_doc_ids=all_ids,
                alpha=self.alpha,
            )[:top_k]
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        return {
            "query":          query,
            "bm25_results":   bm25_results[:top_k],
            "vector_results": vector_results[:top_k],
            "hybrid_results": hybrid_results,
        }
