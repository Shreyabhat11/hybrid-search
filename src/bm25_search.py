"""
bm25_search.py
--------------
Keyword search using the BM25 ranking algorithm.

BM25 is the gold standard for lexical retrieval.  It scores each document
for a query by:
  1. Counting how many query terms appear in the document (TF component)
  2. Weighting rare terms higher than common terms        (IDF component)
  3. Normalising for document length                      (length norm)

We use the `rank_bm25` library which is a pure-Python BM25 implementation.

Install:
    pip install rank-bm25
"""

import re
import string
from typing import Optional

from rank_bm25 import BM25Okapi  # BM25 with Okapi variant (standard)


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Lowercase, remove punctuation, and split into tokens.

    A real production system might also do stemming/lemmatisation,
    but this is enough to show the concepts clearly.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    # remove very short tokens (noise)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# BM25Retriever class
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi to make it easy to index and query documents.

    Parameters
    ----------
    k1 : float
        Term frequency saturation.  Higher → term frequency matters more.
        Typical range: 1.2 – 2.0
    b : float
        Length normalisation strength.  b=1 → full normalisation,
        b=0 → no length normalisation.  Typical value: 0.75
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: list[dict]) -> None:
        """
        Build the BM25 index from a list of document dicts.

        Each document must have at least a 'text' key.
        The 'title' is prepended to the text to give it higher weight.
        """
        self.documents = documents
        self._tokenized_corpus = [
            tokenize(doc.get("title", "") + " " + doc["text"])
            for doc in documents
        ]
        self.bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        print(f"[BM25] Indexed {len(documents)} documents.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return the top-k documents ranked by BM25 score.

        Returns
        -------
        List of dicts, each containing:
            - all original document fields
            - 'bm25_score'  : raw BM25 score
            - 'rank'        : rank position (1-indexed)
        """
        if self.bm25 is None:
            raise RuntimeError("Call .index() before .search()")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # pair scores with document indices, sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            doc = dict(self.documents[idx])  # copy so we don't mutate original
            doc["bm25_score"] = float(score)
            doc["rank"] = rank
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def explain(self, query: str, doc_id: str) -> dict:
        """
        Show per-term BM25 contributions for a specific document.
        Useful for debugging why a document ranked where it did.
        """
        if self.bm25 is None:
            raise RuntimeError("Call .index() before .explain()")

        doc_idx = next(
            (i for i, d in enumerate(self.documents) if d["id"] == doc_id),
            None,
        )
        if doc_idx is None:
            return {"error": f"Document {doc_id} not found"}

        query_tokens = tokenize(query)
        doc_tokens = self._tokenized_corpus[doc_idx]

        term_scores = {}
        for term in set(query_tokens):
            # BM25Okapi stores idf and tf internals we can use
            term_scores[term] = float(
                self.bm25.get_scores([term])[doc_idx]
            )

        return {
            "doc_id": doc_id,
            "query_tokens": query_tokens,
            "doc_token_count": len(doc_tokens),
            "term_contributions": term_scores,
            "total_score": sum(term_scores.values()),
        }
