"""
vector_search.py
----------------
Semantic search using dense vector embeddings.

In a production setup you'd use sentence-transformers with GPU.
Here we use TF-IDF + Truncated SVD (Latent Semantic Analysis) as a
lightweight substitute — same API, same concepts.

LSA captures latent semantic relationships: synonyms and related concepts
score higher than in raw keyword search.

NOTE: In production, replace with:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
The rest of the code stays IDENTICAL.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class VectorRetriever:
    """
    Dense vector retrieval via TF-IDF + LSA (same API as sentence-transformers).

    Parameters
    ----------
    model_name    : kept for API compatibility
    n_components  : dimensionality of the dense embedding space
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_components: int = 128):
        self.model_name   = model_name
        self.n_components = n_components
        self.documents:  list[dict]        = []
        self.embeddings: np.ndarray | None = None

        self._tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        print(f"[Vector] Ready — TF-IDF + LSA (dim={n_components})")
        print(f"         Production swap: SentenceTransformer('{model_name}')")

    def index(self, documents: list[dict]) -> None:
        """Encode all documents into dense vectors."""
        self.documents = documents
        texts = [doc.get("title", "") + ". " + doc["text"] for doc in documents]

        tfidf_matrix = self._tfidf.fit_transform(texts)
        dense = self._svd.fit_transform(tfidf_matrix)
        self.embeddings = normalize(dense, norm="l2")

        print(
            f"[Vector] Indexed {len(documents)} docs. "
            f"Shape: {self.embeddings.shape}  "
            f"Variance explained: {self._svd.explained_variance_ratio_.sum():.1%}"
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k documents by cosine similarity to query."""
        if self.embeddings is None:
            raise RuntimeError("Call .index() before .search()")

        q_tfidf = self._tfidf.transform([query])
        q_dense = self._svd.transform(q_tfidf)
        q_norm  = normalize(q_dense, norm="l2")

        scores = (self.embeddings @ q_norm.T).squeeze()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = dict(self.documents[idx])
            doc["vector_score"] = float(scores[idx])
            doc["rank"] = rank
            results.append(doc)
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Return normalised query embedding."""
        q_tfidf = self._tfidf.transform([query])
        q_dense = self._svd.transform(q_tfidf)
        return normalize(q_dense, norm="l2")

    def top_terms(self, query: str, n: int = 10) -> list[tuple[str, float]]:
        """Show which vocab terms are most activated by the query."""
        q_tfidf = self._tfidf.transform([query])
        feature_names = self._tfidf.get_feature_names_out()
        scores = q_tfidf.toarray().squeeze()
        top_idx = np.argsort(scores)[::-1][:n]
        return [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]
