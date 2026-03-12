"""
main.py
-------
Day 3: Hybrid Search Demo — BM25 + Vector Retrieval + Fusion

Run with:
    python main.py

What this script demonstrates:
  1. Build keyword (BM25) and semantic (vector) indexes
  2. Run test queries and compare results side-by-side
  3. Fuse results with Reciprocal Rank Fusion (RRF)
  4. Evaluate all three retrievers using IR metrics
  5. Show where hybrid search beats either individual retriever
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.documents import DOCUMENTS
from src.bm25_search  import BM25Retriever
from src.vector_search import VectorRetriever
from src.hybrid_search import HybridRetriever
from src.evaluation    import compare_retrievers, print_comparison_table


# ---------------------------------------------------------------------------
# Test queries — carefully designed to showcase different retriever strengths
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # ── Keyword-friendly query ──────────────────────────────────────────────
    # BM25 should win here: exact technical terms "BM25" and "k1 parameter"
    {
        "query": "BM25 k1 parameter term frequency",
        "relevant_ids": ["doc_002", "doc_015"],
        "note": "Keyword-friendly: exact technical terms",
    },

    # ── Semantic-friendly query ──────────────────────────────────────────────
    # Vector should win: "how neural nets learn" is a paraphrase of gradient descent
    {
        "query": "how neural networks learn during training",
        "relevant_ids": ["doc_006", "doc_014"],
        "note": "Semantic-friendly: conceptual paraphrase",
    },

    # ── Hybrid should win ────────────────────────────────────────────────────
    # Mix of exact terms ("retrieval", "LLM") and semantic concepts
    {
        "query": "retrieval augmented generation LLM context",
        "relevant_ids": ["doc_004", "doc_003", "doc_008"],
        "note": "Hybrid case: exact + semantic",
    },

    # ── Vocabulary mismatch ──────────────────────────────────────────────────
    # "car" vs "automobile" — BM25 fails, vector handles it
    {
        "query": "sparse index for fast term lookup",
        "relevant_ids": ["doc_007", "doc_002"],
        "note": "Vocabulary mismatch: different terms, same concept",
    },

    # ── Rare exact term ──────────────────────────────────────────────────────
    # BM25 highly rewards rare specific terms
    {
        "query": "reciprocal rank fusion score normalisation",
        "relevant_ids": ["doc_013"],
        "note": "Rare specific term: BM25 strength",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_results(label: str, results: list[dict], query: str) -> None:
    """Pretty-print a list of retrieval results."""
    print(f"\n  ── {label} ──")
    for doc in results:
        score_str = ""
        if "bm25_score"   in doc: score_str = f"BM25={doc['bm25_score']:.3f}"
        if "vector_score" in doc: score_str = f"Vec={doc['vector_score']:.3f}"
        if "rrf_score"    in doc: score_str = f"RRF={doc['rrf_score']:.6f}"
        if "linear_score" in doc: score_str = f"Lin={doc['linear_score']:.3f}"
        print(f"    [{doc['rank']}] {score_str:20s} | {doc['id']} | {doc['title']}")


def print_section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ────────────────────────────────────────────────────────────────────
    # STEP 1 — Index documents
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 1: Indexing documents")

    # Individual retrievers (for comparison)
    bm25_ret   = BM25Retriever()
    vector_ret = VectorRetriever(model_name="all-MiniLM-L6-v2")
    hybrid_ret = HybridRetriever(model_name="all-MiniLM-L6-v2", fusion="rrf")

    bm25_ret.index(DOCUMENTS)
    vector_ret.index(DOCUMENTS)
    hybrid_ret.index(DOCUMENTS)

    print(f"\n  Corpus size: {len(DOCUMENTS)} documents")

    # ────────────────────────────────────────────────────────────────────
    # STEP 2 — Side-by-side comparisons for each test query
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 2: Query Comparisons")

    TOP_K = 3

    for i, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        note  = test_case["note"]

        print(f"\n  Query {i}: \"{query}\"")
        print(f"  [{note}]")
        print(f"  Relevant docs: {test_case['relevant_ids']}")

        bm25_results   = bm25_ret.search(query, top_k=TOP_K)
        vector_results = vector_ret.search(query, top_k=TOP_K)
        hybrid_out     = hybrid_ret.search(query, top_k=TOP_K)

        print_results("BM25 (Keyword)",   bm25_results,                query)
        print_results("Vector (Semantic)", vector_results,             query)
        print_results("Hybrid (RRF)",     hybrid_out["hybrid_results"], query)

    # ────────────────────────────────────────────────────────────────────
    # STEP 3 — BM25 explain (show term-level contributions)
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 3: BM25 Term-Level Explanation")

    explain_query  = "BM25 k1 parameter term frequency"
    explain_doc_id = "doc_002"
    explanation    = bm25_ret.explain(explain_query, explain_doc_id)

    print(f"\n  Query    : \"{explain_query}\"")
    print(f"  Document : {explain_doc_id}")
    print(f"  Tokens   : {explanation['query_tokens']}")
    print(f"  Doc length: {explanation['doc_token_count']} tokens")
    print(f"\n  Per-term BM25 contributions:")
    for term, score in sorted(explanation["term_contributions"].items(), key=lambda x: -x[1]):
        bar_len = int(score * 40 / max(explanation["term_contributions"].values(), default=1))
        bar     = "█" * bar_len
        print(f"    {term:20s} {score:.4f}  {bar}")

    # ────────────────────────────────────────────────────────────────────
    # STEP 4 — Linear combination fusion demo
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 4: Linear Combination Fusion (alpha=0.7)")

    hybrid_linear = HybridRetriever(
        model_name="all-MiniLM-L6-v2",
        fusion="linear",
        alpha=0.7,   # 70% vector, 30% BM25
    )
    hybrid_linear.index(DOCUMENTS)

    linear_query = "neural network learning optimisation"
    linear_out   = hybrid_linear.search(linear_query, top_k=4)

    print(f"\n  Query: \"{linear_query}\"  (alpha=0.7 → 70% semantic, 30% keyword)")
    print_results("Hybrid (Linear)", linear_out["hybrid_results"], linear_query)

    # ────────────────────────────────────────────────────────────────────
    # STEP 5 — Quantitative evaluation
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 5: Evaluation — Precision, Recall, MRR, HitRate@5")

    K = 5
    eval_queries = [
        {"query": q["query"], "relevant_ids": q["relevant_ids"]}
        for q in TEST_QUERIES
    ]

    retriever_fns = {
        "BM25":   lambda q: bm25_ret.search(q,   top_k=K),
        "Vector": lambda q: vector_ret.search(q, top_k=K),
        "Hybrid": lambda q: hybrid_ret.search(q, top_k=K)["hybrid_results"],
    }

    print(f"\n  Running evaluation over {len(eval_queries)} queries (K={K}) ...")
    comparison = compare_retrievers(retriever_fns, eval_queries, k=K)
    print_comparison_table(comparison)

    # ────────────────────────────────────────────────────────────────────
    # STEP 6 — Key takeaways
    # ────────────────────────────────────────────────────────────────────
    print_section("STEP 6: Key Takeaways")
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │ When to use each retriever                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │ BM25 (Keyword):                                                 │
  │   ✓ Exact technical terms, product codes, error messages        │
  │   ✓ Fast, no GPU required                                       │
  │   ✗ Fails on synonyms and paraphrases                           │
  │                                                                 │
  │ Vector (Semantic):                                              │
  │   ✓ Conceptual questions, paraphrases, cross-lingual            │
  │   ✗ Can miss rare/specific technical terms                      │
  │   ✗ Requires model + GPU for large corpora                      │
  │                                                                 │
  │ Hybrid (BM25 + Vector + RRF):                                   │
  │   ✓ Best of both worlds in most real-world scenarios            │
  │   ✓ RRF is robust, parameter-free, easy to implement            │
  │   ✓ The default choice for production RAG systems               │
  └─────────────────────────────────────────────────────────────────┘
    """)

    print("  Day 3 complete! Next up: Re-ranking with cross-encoders.\n")


if __name__ == "__main__":
    main()
