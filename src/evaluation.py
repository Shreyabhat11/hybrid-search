"""
evaluation.py
-------------
Tools for evaluating and comparing retrieval systems.

Metrics implemented:
  - Precision@K   : fraction of top-K results that are relevant
  - Recall@K      : fraction of relevant docs found in top-K
  - MRR           : Mean Reciprocal Rank (rewards finding the best doc early)
  - Hit Rate@K    : did any relevant doc appear in top-K? (binary)

These are the standard metrics for information retrieval evaluation.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:K]| / K
    """
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_doc
    Returns 0 if no relevant doc is found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Hit Rate@K = 1 if any relevant doc is in top-K, else 0.
    Also called "Recall@K" in some papers (for single relevant doc).
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever_fn,           # callable: (query) -> list[dict]
    test_queries: list[dict],   # list of {"query": str, "relevant_ids": list[str]}
    k: int = 5,
) -> dict:
    """
    Run a retriever over test queries and return aggregate metrics.

    Parameters
    ----------
    retriever_fn  : function that takes a query string and returns a ranked list of dicts
    test_queries  : list of dicts with 'query' and 'relevant_ids' keys
    k             : cut-off for Precision/Recall/Hit Rate

    Returns
    -------
    Dict with averaged metrics across all queries.
    """
    p_scores, r_scores, rr_scores, hr_scores = [], [], [], []

    for test_case in test_queries:
        query        = test_case["query"]
        relevant_ids = set(test_case["relevant_ids"])

        results = retriever_fn(query)
        retrieved_ids = [d["id"] for d in results]

        p_scores.append(precision_at_k(retrieved_ids, relevant_ids, k))
        r_scores.append(recall_at_k(retrieved_ids,    relevant_ids, k))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
        hr_scores.append(hit_rate_at_k(retrieved_ids, relevant_ids, k))

    return {
        f"Precision@{k}":  round(sum(p_scores)  / len(p_scores),  4),
        f"Recall@{k}":     round(sum(r_scores)  / len(r_scores),  4),
        "MRR":             round(sum(rr_scores) / len(rr_scores), 4),
        f"HitRate@{k}":    round(sum(hr_scores) / len(hr_scores), 4),
        "num_queries":     len(test_queries),
    }


def compare_retrievers(
    retriever_fns: dict[str, callable],
    test_queries: list[dict],
    k: int = 5,
) -> dict[str, dict]:
    """
    Evaluate multiple retrievers and return a side-by-side comparison.

    Parameters
    ----------
    retriever_fns : {"BM25": fn1, "Vector": fn2, "Hybrid": fn3}
    """
    results = {}
    for name, fn in retriever_fns.items():
        print(f"  Evaluating: {name} ...")
        results[name] = evaluate_retriever(fn, test_queries, k=k)
    return results


def print_comparison_table(comparison: dict[str, dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not comparison:
        return

    metrics = list(next(iter(comparison.values())).keys())
    metrics = [m for m in metrics if m != "num_queries"]

    col_w   = 14
    name_w  = 12
    header  = f"{'Retriever':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name, scores in comparison.items():
        row = f"{name:<{name_w}}" + "".join(
            f"{scores.get(m, 0.0):>{col_w}.4f}" for m in metrics
        )
        print(row)

    print("=" * len(header))
