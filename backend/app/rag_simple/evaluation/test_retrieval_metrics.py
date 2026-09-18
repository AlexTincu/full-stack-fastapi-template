from langchain_core.documents import Document

from app.rag_simple.evaluation.retrieval_metrics import (
    StageMetrics,
    average_stage_metrics,
    evaluate_stage,
    evaluate_trace,
    hit_rate,
    keyword_hit_ranks,
    precision_at_k,
    recall_at_k,
)
from app.rag_simple.evaluation.test import TestQuestion
from app.rag_simple.retrieval.pipeline import RetrievalTrace


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


DOCS = [
    _doc("apples and oranges"),
    _doc("bananas only"),
    _doc("apples again"),
    _doc("nothing relevant"),
]


def test_keyword_hit_ranks_finds_first_match_case_insensitive():
    assert keyword_hit_ranks("APPLES", DOCS) == 1


def test_keyword_hit_ranks_returns_none_when_absent():
    assert keyword_hit_ranks("mango", DOCS) is None


def test_recall_at_k_counts_fraction_of_keywords_found():
    # "apples" hits rank 1, "bananas" hits rank 2, "mango" never hits.
    assert recall_at_k(["apples", "bananas", "mango"], DOCS, k=4) == 2 / 3


def test_recall_at_k_respects_k_cutoff():
    # "bananas" only appears at rank 2; cutting off at k=1 must miss it.
    assert recall_at_k(["bananas"], DOCS, k=1) == 0.0


def test_precision_at_k_averages_per_keyword_fraction():
    # "apples" appears in 2 of the top 4 docs -> 0.5; "bananas" in 1 of 4 -> 0.25.
    assert precision_at_k(["apples", "bananas"], DOCS, k=4) == (0.5 + 0.25) / 2


def test_precision_at_k_handles_k_larger_than_available_docs():
    # top_k is clamped to the 4 available docs, not padded to k=100.
    assert precision_at_k(["apples"], DOCS, k=100) == 2 / 4


def test_hit_rate_is_binary_per_question_not_per_keyword():
    assert hit_rate(["apples", "mango"], DOCS, k=4) == 1.0
    assert hit_rate(["mango"], DOCS, k=4) == 0.0


def test_evaluate_stage_returns_stage_metrics_with_expected_values():
    metrics = evaluate_stage("semantic", DOCS, ["apples"])
    assert isinstance(metrics, StageMetrics)
    assert metrics.stage == "semantic"
    assert metrics.mrr == 1.0  # "apples" found at rank 1
    assert metrics.recall_at_10 == 1.0
    assert metrics.hit_rate_at_10 == 1.0


def test_evaluate_trace_runs_all_four_stages():
    trace = RetrievalTrace(
        question="q",
        rewritten_question="q",
        semantic_hits=DOCS,
        lexical_hits=DOCS[:1],
        fused=DOCS[:2],
        reranked=DOCS,
    )
    test = TestQuestion(question="q", keywords=["apples"], reference_answer="ref", category="direct_fact")

    result = evaluate_trace(trace, test)

    assert set(result.keys()) == {"semantic", "lexical", "fused", "reranked"}
    assert result["semantic"].recall_at_10 == 1.0
    assert result["lexical"].recall_at_10 == 1.0  # lexical_hits = ["apples and oranges"]


def test_average_stage_metrics_computes_mean_across_questions():
    per_question = [
        {"semantic": StageMetrics("semantic", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)},
        {"semantic": StageMetrics("semantic", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
    ]

    averaged = average_stage_metrics(per_question)

    assert averaged["semantic"]["mrr"] == 0.5
    assert averaged["semantic"]["recall_at_5"] == 0.5


def test_average_stage_metrics_empty_input_returns_empty_dict():
    assert average_stage_metrics([]) == {}
