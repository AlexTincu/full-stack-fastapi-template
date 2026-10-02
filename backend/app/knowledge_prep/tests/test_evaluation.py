import json
from pathlib import Path

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.knowledge_prep.evaluation import (
    METRICS,
    TestCase,
    evaluate,
    fold,
    keyword_rank,
    load_tests,
    question_metrics,
    split_answerable,
)


def docs(*texts: str) -> list[Document]:
    return [Document(page_content=t) for t in texts]


class FixedRetriever(BaseRetriever):
    """Aceleași rezultate pentru orice întrebare; reține întrebările primite."""

    results: list[Document]
    queries: list[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.results


def test_fold_ignores_case_diacritics_and_spacing():
    assert fold("  Două\n SĂPTĂMÂNI, grâu încolțit ") == "doua saptamani, grau incoltit"
    assert fold("grãsimi") == "grasimi"  # ã din textele vechi


def test_keyword_rank_matches_without_diacritics():
    results = docs("nimic aici", "Postul negru: 2-5 săptămâni")
    assert keyword_rank("2-5 saptamani", results) == 2
    assert keyword_rank("absent", results) is None


def test_question_metrics_same_definitions_as_rag_simple():
    results = docs("alfa", "beta gama", "alfa gama")
    m = question_metrics(["alfa", "gama", "delta"], results)

    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert m["recall_at_5"] == pytest.approx(2 / 3)
    assert m["recall_at_10"] == pytest.approx(2 / 3)
    # alfa în 2/3 rezultate, gama în 2/3, delta în 0
    assert m["precision_at_5"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["hit_rate_at_10"] == 1.0
    assert 0 < m["ndcg"] < 1


def test_question_metrics_empty():
    assert question_metrics([], docs("x")) == dict.fromkeys(METRICS, 0.0)
    assert question_metrics(["x"], []) == dict.fromkeys(METRICS, 0.0)


def test_hit_rate_only_counts_first_ten():
    results = docs(*(["nimic"] * 10), "ținta")
    assert question_metrics(["ținta"], results)["hit_rate_at_10"] == 0.0
    assert question_metrics(["ținta"], results)["mrr"] == pytest.approx(1 / 11)


def test_load_tests_reports_bad_line(tmp_path: Path):
    good = {"question": "Q?", "keywords": ["k"], "category": "direct_fact"}
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps(good) + "\n\n" + json.dumps(good) + ",\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"t.jsonl:3"):
        load_tests(path)

    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert load_tests(path) == [
        TestCase(question="Q?", keywords=["k"], category="direct_fact")
    ]


def test_default_tests_file_is_valid():
    assert load_tests()


def test_split_answerable_skips_questions_without_keywords_in_corpus():
    a = TestCase(question="grâu?", keywords=["absent", "graul incoltit"])
    b = TestCase(question="orz?", keywords=["orz"])
    answerable, skipped = split_answerable([a, b], ["Grâul Încolțit dă putere."])
    # potrivire pe subșir, ca în rag_simple: „grau incoltit” n-ar fi găsit „graul încolțit”
    assert answerable == [a]
    assert skipped == [b]


def test_evaluate_runs_every_question_through_every_mode():
    tests = [
        TestCase(question="Q1", keywords=["alfa"], category="direct_fact"),
        TestCase(question="Q2", keywords=["beta"], category="reasoning"),
    ]
    good = FixedRetriever(results=docs("alfa", "beta"), queries=[])
    bad = FixedRetriever(results=docs("nimic"), queries=[])
    skipped = [TestCase(question="Q3", keywords=["x"])]

    report = evaluate({"bun": good, "rău": bad}, tests, skipped)

    assert good.queries == ["Q1", "Q2"] and bad.queries == ["Q1", "Q2"]
    assert report.modes == ["bun", "rău"]
    assert report.skipped == skipped
    avg = report.averages()
    assert avg["bun"]["mrr"] == pytest.approx((1 + 1 / 2) / 2)
    assert avg["rău"]["hit_rate_at_10"] == 0.0
    assert report.categories() == ["direct_fact", "reasoning"]
    assert report.averages("reasoning")["bun"]["mrr"] == pytest.approx(1 / 2)
    assert evaluate({"bun": good}, []).averages() == {}
