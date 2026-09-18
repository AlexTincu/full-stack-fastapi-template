"""Per-stage retrieval metrics, reusing eval.py's keyword-in-content proxy for relevance."""

from dataclasses import dataclass, fields

from langchain_core.documents import Document

from app.rag_simple.evaluation.eval import calculate_mrr, calculate_ndcg
from app.rag_simple.evaluation.test import TestQuestion
from app.rag_simple.retrieval.pipeline import RetrievalTrace


def keyword_hit_ranks(keyword: str, docs: list[Document]) -> int | None:
    """1-based rank of the first doc containing keyword (case-insensitive), or None."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return rank
    return None


def recall_at_k(keywords: list[str], docs: list[Document], k: int) -> float:
    """Fraction of keywords found within the first k documents."""
    if not keywords:
        return 0.0
    top_k = docs[:k]
    found = sum(1 for keyword in keywords if keyword_hit_ranks(keyword, top_k) is not None)
    return found / len(keywords)


def precision_at_k(keywords: list[str], docs: list[Document], k: int) -> float:
    """Average, over keywords, of the fraction of the top-k documents containing that keyword."""
    if not keywords:
        return 0.0
    top_k = docs[:k]
    if not top_k:
        return 0.0
    scores = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        hits = sum(1 for doc in top_k if keyword_lower in doc.page_content.lower())
        scores.append(hits / len(top_k))
    return sum(scores) / len(scores)


def hit_rate(keywords: list[str], docs: list[Document], k: int) -> float:
    """1.0 if at least one keyword is found within the first k documents, else 0.0.

    Binary per question, unlike recall_at_k's continuous fraction: this answers
    "did this stage surface anything useful at all", not "how much of it".
    """
    if not keywords:
        return 0.0
    top_k = docs[:k]
    return 1.0 if any(keyword_hit_ranks(keyword, top_k) is not None for keyword in keywords) else 0.0


@dataclass
class StageMetrics:
    stage: str
    mrr: float
    ndcg: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    hit_rate_at_10: float


def evaluate_stage(stage_name: str, docs: list[Document], keywords: list[str]) -> StageMetrics:
    mrr_scores = [calculate_mrr(keyword, docs) for keyword in keywords]
    ndcg_scores = [calculate_ndcg(keyword, docs, 10) for keyword in keywords]
    return StageMetrics(
        stage=stage_name,
        mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        recall_at_5=recall_at_k(keywords, docs, 5),
        recall_at_10=recall_at_k(keywords, docs, 10),
        precision_at_5=precision_at_k(keywords, docs, 5),
        precision_at_10=precision_at_k(keywords, docs, 10),
        hit_rate_at_10=hit_rate(keywords, docs, 10),
    )


def evaluate_trace(trace: RetrievalTrace, test: TestQuestion) -> dict[str, StageMetrics]:
    """Run evaluate_stage on each of the 4 traced stages of a RetrievalTrace."""
    stage_docs = {
        "semantic": trace.semantic_hits,
        "lexical": trace.lexical_hits,
        "fused": trace.fused,
        "reranked": trace.reranked,
    }
    return {
        stage_name: evaluate_stage(stage_name, docs, test.keywords)
        for stage_name, docs in stage_docs.items()
    }


_METRIC_FIELDS = [f.name for f in fields(StageMetrics) if f.name != "stage"]


def average_stage_metrics(per_question: list[dict[str, StageMetrics]]) -> dict[str, dict[str, float]]:
    """Mean of each StageMetrics field across questions, per stage.

    Input: one evaluate_trace() result per question. Output: stage name ->
    {metric_name: mean} — the shape EvalSnapshot.stage_metrics expects.
    """
    if not per_question:
        return {}
    stages = per_question[0].keys()
    return {
        stage: {
            metric: sum(getattr(result[stage], metric) for result in per_question) / len(per_question)
            for metric in _METRIC_FIELDS
        }
        for stage in stages
    }
