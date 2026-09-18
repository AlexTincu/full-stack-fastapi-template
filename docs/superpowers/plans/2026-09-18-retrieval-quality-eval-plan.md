# Retrieval Quality Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `rag_simple` a quantified, repeatable way to measure retrieval quality per pipeline stage (semantic / lexical / fused / reranked), so future retrieval changes can be compared against a saved baseline instead of eyeballed.

**Architecture:** Two new pure-Python modules under `evaluation/` (`retrieval_metrics.py` for per-stage metrics, `results.py` for JSON snapshot persistence), reusing the existing `retrieve_with_trace` pipeline and `calculate_mrr`/`calculate_ndcg` from `eval.py` unchanged. A new notebook (`retrieval_quality.ipynb`) wires them together over the existing 10-question test set and prints an aggregate table + per-question failure breakdown, then saves/compares snapshots.

**Tech Stack:** Python 3.14, `langchain_core.documents.Document`, stdlib `dataclasses`/`json`/`pathlib`, pytest, Jupyter (`nbformat`, already a dev dependency). No new dependencies — no `pandas`.

**Spec:** [`docs/superpowers/specs/2026-09-17-retrieval-quality-eval-design.md`](../specs/2026-09-17-retrieval-quality-eval-design.md)

## Global Constraints

- No new dependencies in `pyproject.toml` — no `pandas`; tables are built with plain `print`/f-strings, matching `eval.py`'s existing style.
- No existing file under `backend/app/rag_simple/` is modified — every deliverable in this plan is a new file. (Spec: "Niciun fișier existent din `rag_simple/` nu se modifică.")
- `pyproject.toml`'s `[tool.ruff] exclude` already contains `"app/rag_simple/evaluation"`, and `*.ipynb` is excluded too — the pre-commit `ruff check` hook (`--force-exclude`) will skip every file this plan creates. `mypy --strict` is nominally configured for this tree but already fails on the pre-existing `eval.py` (8 errors, unrelated to this work) — it is not actually enforced here. **Do not gate any task in this plan on ruff or mypy.** Write clean, fully-typed code (the signatures below already are), verify via pytest and the notebook's own execution, and stop there.
- Config values this plan reads (from `app/rag_simple/config.py`, unchanged): `RETRIEVAL_K = 20`, `FINAL_K = 20`, `CHAT_MODEL = "openai:gpt-4.1-mini"`, `EMBEDDING_MODEL = "openai:text-embedding-3-large"`.
- The corpus is already ingested: Postgres (`pgvector`) table `rag_simple_chunks` currently holds 96 rows from the 5 documents in `KNOWLEDGE_DIR`, and `backend/app/rag_simple/db/chunks.json` (used for the BM25 index) already exists on disk. **No `ingest.py` run is needed.** The `db` Postgres container (service `db` in `full-stack-fastapi-template/compose.yml`) must be running before executing the notebook in Task 3 — start it from the `full-stack-fastapi-template/` repo root with `docker compose up -d db` (it uses a persistent named volume, so the existing 96 rows come back as soon as it's healthy).
- All commands below assume a shell `cd`'d to `full-stack-fastapi-template/backend/` (the `uv` project root) unless stated otherwise.
- Two spec ambiguities were resolved while writing this plan (both load-bearing for Task 1 — carried into its brief verbatim, not left for the implementer to guess):
  1. **`hit_rate` vs `recall_at_k`**: the spec's prose defines them almost identically ("fraction of keywords found in top-k" for both). Implemented literally, `StageMetrics` would show the same number in `recall_at_10` and `hit_rate_at_10`, which defeats the point of having both columns in the aggregate table. Ruling: `hit_rate` follows the standard IR "hit rate / success rate" definition — a **binary** 1.0/0.0 per question ("did this stage surface *anything* useful at all"), as opposed to `recall_at_k`'s **continuous** fraction ("how much of the ground truth did it surface"). This gives the aggregate table two genuinely different signals.
  2. **Where per-question averaging happens**: the spec's `evaluation/results.py::EvalSnapshot.stage_metrics` needs "metrici agregate (medie pe toate întrebările)", but no function in the spec's `retrieval_metrics.py` pseudocode computes that mean — the spec only lists `evaluate_stage`/`evaluate_trace`, which operate on one question at a time. Ruling: add one small function, `average_stage_metrics`, to `retrieval_metrics.py` (not listed in the spec's pseudocode, but same file, same style, unit-tested like its neighbors) rather than writing untested aggregation math inline in the notebook. This keeps the notebook a thin consumer, consistent with the spec's own goal of quantified/testable logic over notebook-only code.

---

### Task 1: `evaluation/retrieval_metrics.py` — per-stage metric functions

**Files:**
- Create: `backend/app/rag_simple/evaluation/retrieval_metrics.py`
- Test: `backend/app/rag_simple/evaluation/test_retrieval_metrics.py`

**Interfaces:**
- Consumes (all pre-existing, unmodified):
  - `calculate_mrr(keyword: str, retrieved_docs: list) -> float` and `calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float` from `app.rag_simple.evaluation.eval`.
  - `TestQuestion` (fields: `question: str`, `keywords: list[str]`, `reference_answer: str`, `category: str`) from `app.rag_simple.evaluation.test`.
  - `RetrievalTrace` (fields: `question`, `rewritten_question`, `semantic_hits: list[Document]`, `lexical_hits: list[Document]`, `fused: list[Document]`, `reranked: list[Document]`) from `app.rag_simple.retrieval.pipeline`.
  - `langchain_core.documents.Document` (`.page_content: str`, `.metadata: dict`).
- Produces (used by Task 3):
  - `keyword_hit_ranks(keyword: str, docs: list[Document]) -> int | None`
  - `recall_at_k(keywords: list[str], docs: list[Document], k: int) -> float`
  - `precision_at_k(keywords: list[str], docs: list[Document], k: int) -> float`
  - `hit_rate(keywords: list[str], docs: list[Document], k: int) -> float`
  - `StageMetrics` dataclass: `stage: str`, `mrr: float`, `ndcg: float`, `recall_at_5: float`, `recall_at_10: float`, `precision_at_5: float`, `precision_at_10: float`, `hit_rate_at_10: float`
  - `evaluate_stage(stage_name: str, docs: list[Document], keywords: list[str]) -> StageMetrics`
  - `evaluate_trace(trace: RetrievalTrace, test: TestQuestion) -> dict[str, StageMetrics]` (keys: `"semantic"`, `"lexical"`, `"fused"`, `"reranked"`)
  - `average_stage_metrics(per_question: list[dict[str, StageMetrics]]) -> dict[str, dict[str, float]]`

- [ ] **Step 1: Write the failing tests**

Create `backend/app/rag_simple/evaluation/test_retrieval_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest app/rag_simple/evaluation/test_retrieval_metrics.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'app.rag_simple.evaluation.retrieval_metrics'`.

- [ ] **Step 3: Write the implementation**

Create `backend/app/rag_simple/evaluation/retrieval_metrics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest app/rag_simple/evaluation/test_retrieval_metrics.py -v`
Expected: all tests PASS. (A `PytestCollectionWarning` about `app/rag_simple/evaluation/test.py`'s `TestQuestion` class is pre-existing and unrelated — ignore it.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag_simple/evaluation/retrieval_metrics.py backend/app/rag_simple/evaluation/test_retrieval_metrics.py
git commit -m "feat(rag_simple/eval): add per-stage retrieval metrics"
```

---

### Task 2: `evaluation/results.py` — JSON snapshot persistence

**Files:**
- Create: `backend/app/rag_simple/evaluation/results.py`
- Test: `backend/app/rag_simple/evaluation/test_results.py`

**Interfaces:**
- Consumes: nothing from Task 1 — this module only deals with plain `dict`/`str`/`Path`, independent of `retrieval_metrics.py`.
- Produces (used by Task 3):
  - `EvalSnapshot` dataclass: `label: str`, `timestamp: str`, `config: dict`, `stage_metrics: dict[str, dict]`, `notes: str = ""`
  - `EvalSnapshot.save(self, results_dir: Path) -> Path`
  - `EvalSnapshot.load(cls, path: Path) -> EvalSnapshot` (classmethod)
  - `list_snapshots(results_dir: Path) -> list[EvalSnapshot]`
  - `diff_snapshots(a: EvalSnapshot, b: EvalSnapshot) -> dict[str, dict]`
- Filename convention: `save()` writes to `<timestamp>_<slugified-label>.json`, e.g. `20260917T120000Z_baseline-run.json`. `list_snapshots` relies on this prefix to sort oldest-first by filename — callers needing "the last two saved" use `list_snapshots(...)[-2:]`.

- [ ] **Step 1: Write the failing tests**

Create `backend/app/rag_simple/evaluation/test_results.py`:

```python
from pathlib import Path

import pytest

from app.rag_simple.evaluation.results import EvalSnapshot, diff_snapshots, list_snapshots


def _snapshot(label: str, timestamp: str, semantic_mrr: float) -> EvalSnapshot:
    return EvalSnapshot(
        label=label,
        timestamp=timestamp,
        config={
            "RETRIEVAL_K": 20,
            "FINAL_K": 20,
            "CHAT_MODEL": "openai:gpt-4.1-mini",
            "EMBEDDING_MODEL": "openai:text-embedding-3-large",
        },
        stage_metrics={"semantic": {"mrr": semantic_mrr, "recall_at_10": 0.5}},
    )


def test_save_writes_json_with_timestamp_and_slugified_label(tmp_path: Path):
    snapshot = _snapshot("baseline run", "20260917T120000Z", 0.5)

    path = snapshot.save(tmp_path)

    assert path.name == "20260917T120000Z_baseline-run.json"
    assert path.exists()


def test_save_creates_results_dir_if_missing(tmp_path: Path):
    results_dir = tmp_path / "results"
    snapshot = _snapshot("baseline", "20260917T120000Z", 0.5)

    snapshot.save(results_dir)

    assert results_dir.exists()


def test_load_round_trips_saved_snapshot(tmp_path: Path):
    snapshot = _snapshot("baseline", "20260917T120000Z", 0.75)
    path = snapshot.save(tmp_path)

    loaded = EvalSnapshot.load(path)

    assert loaded == snapshot


def test_list_snapshots_returns_oldest_first(tmp_path: Path):
    older = _snapshot("baseline", "20260917T090000Z", 0.5)
    newer = _snapshot("contextualize-on", "20260917T150000Z", 0.6)
    newer.save(tmp_path)
    older.save(tmp_path)  # saved out of chronological order

    snapshots = list_snapshots(tmp_path)

    assert [s.label for s in snapshots] == ["baseline", "contextualize-on"]


def test_list_snapshots_missing_dir_returns_empty_list(tmp_path: Path):
    assert list_snapshots(tmp_path / "missing") == []


def test_diff_snapshots_computes_delta_per_stage_and_metric():
    a = _snapshot("baseline", "20260917T090000Z", 0.5)
    b = _snapshot("contextualize-on", "20260917T150000Z", 0.8)

    delta = diff_snapshots(a, b)

    assert delta["semantic"]["mrr"] == pytest.approx(0.3)
    assert delta["semantic"]["recall_at_10"] == pytest.approx(0.0)


def test_diff_snapshots_handles_stage_missing_from_one_side():
    a = EvalSnapshot(label="a", timestamp="t1", config={}, stage_metrics={"semantic": {"mrr": 0.5}})
    b = EvalSnapshot(
        label="b",
        timestamp="t2",
        config={},
        stage_metrics={"semantic": {"mrr": 0.5}, "fused": {"mrr": 0.9}},
    )

    delta = diff_snapshots(a, b)

    assert delta["fused"]["mrr"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest app/rag_simple/evaluation/test_results.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'app.rag_simple.evaluation.results'`.

- [ ] **Step 3: Write the implementation**

Create `backend/app/rag_simple/evaluation/results.py`:

```python
"""JSON snapshots of aggregated per-stage retrieval metrics, for before/after comparison."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "snapshot"


@dataclass
class EvalSnapshot:
    label: str
    timestamp: str
    config: dict
    stage_metrics: dict[str, dict]
    notes: str = ""

    def save(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.timestamp}_{_slugify(self.label)}.json"
        path = results_dir / filename
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalSnapshot":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def list_snapshots(results_dir: Path) -> list[EvalSnapshot]:
    """All snapshots in results_dir, oldest first (filenames start with an ISO-8601 timestamp)."""
    if not results_dir.exists():
        return []
    return [EvalSnapshot.load(path) for path in sorted(results_dir.glob("*.json"))]


def diff_snapshots(a: EvalSnapshot, b: EvalSnapshot) -> dict[str, dict]:
    """Per-stage, per-metric delta (b - a), for before/after comparison."""
    stages = sorted(set(a.stage_metrics) | set(b.stage_metrics))
    diff: dict[str, dict] = {}
    for stage in stages:
        a_metrics = a.stage_metrics.get(stage, {})
        b_metrics = b.stage_metrics.get(stage, {})
        metric_names = sorted(set(a_metrics) | set(b_metrics))
        diff[stage] = {
            metric: b_metrics.get(metric, 0.0) - a_metrics.get(metric, 0.0) for metric in metric_names
        }
    return diff
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest app/rag_simple/evaluation/test_results.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag_simple/evaluation/results.py backend/app/rag_simple/evaluation/test_results.py
git commit -m "feat(rag_simple/eval): add EvalSnapshot JSON persistence"
```

---

### Task 3: `evaluation/retrieval_quality.ipynb` — the eval notebook

**Files:**
- Create: `backend/app/rag_simple/evaluation/retrieval_quality.ipynb`
- Create (as a side effect of running the notebook, see Step 3): `backend/app/rag_simple/evaluation/results/<timestamp>_baseline.json`

**Interfaces:**
- Consumes: everything produced by Task 1 (`evaluate_trace`, `average_stage_metrics`, `keyword_hit_ranks`) and Task 2 (`EvalSnapshot`, `list_snapshots`, `diff_snapshots`), plus pre-existing `load_tests()` (`app.rag_simple.evaluation.test`), `retrieve_with_trace(question: str) -> RetrievalTrace` (`app.rag_simple.retrieval.pipeline`), and `config.ROOT` / `config.RETRIEVAL_K` / `config.FINAL_K` / `config.CHAT_MODEL` / `config.EMBEDDING_MODEL` (`app.rag_simple.config`).
- Produces: nothing consumed by later tasks — this is the last task in the plan.

This task has no unit tests (per spec: "Nu există teste automate clasice pentru un notebook"). Validation is: build the notebook, run it end-to-end for real (live Postgres + OpenAI calls), and confirm the output is plausible and a snapshot file was written and is reloadable.

- [ ] **Step 1: Build the notebook file**

`.ipynb` is JSON; build it with `nbformat` (already a dev dependency, ships with `jupyter`) rather than hand-writing JSON. Create a **temporary, uncommitted** script at `backend/app/rag_simple/evaluation/_build_notebook.py` with exactly this content:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "full-stack-fastapi-template (3.14.4)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.4",
    },
}

CELL_INTRO = """\
# Retrieval quality — evaluare cuantificată

Rulează cele 10 întrebări din `tests.jsonl` prin pipeline-ul de retrieval \
(`retrieve_with_trace`), calculează MRR / nDCG / recall / precision / hit-rate \
pe fiecare din cele 4 etape (semantic, lexical, fused, reranked), și le compară.

Presupune că datele sunt deja ingerate în Postgres (vezi `README.md`). Pentru \
inspecție liberă, nemetrizată, pe întrebări ad-hoc, vezi \
[`retrieval_playground.ipynb`](../retrieval/retrieval_playground.ipynb)."""

CELL_SETUP = """\
from datetime import UTC, datetime

from dotenv import load_dotenv

load_dotenv(override=True)

from app.rag_simple import config
from app.rag_simple.evaluation.retrieval_metrics import (
    average_stage_metrics,
    evaluate_trace,
    keyword_hit_ranks,
)
from app.rag_simple.evaluation.results import EvalSnapshot, diff_snapshots, list_snapshots
from app.rag_simple.evaluation.test import load_tests
from app.rag_simple.retrieval.pipeline import retrieve_with_trace

RESULTS_DIR = config.ROOT / "evaluation" / "results"
STAGES = ("semantic", "lexical", "fused", "reranked")
METRIC_COLUMNS = (
    "mrr",
    "ndcg",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "hit_rate_at_10",
)

print(f"RETRIEVAL_K={config.RETRIEVAL_K}  FINAL_K={config.FINAL_K}  CHAT_MODEL={config.CHAT_MODEL}")"""

CELL_RUN = """\
tests = load_tests()
traces = []
per_question_metrics = []

for test in tests:
    trace = retrieve_with_trace(test.question)
    traces.append(trace)
    per_question_metrics.append(evaluate_trace(trace, test))

print(f"Evaluated {len(tests)} questions across stages {STAGES}")"""

CELL_TABLE = """\
aggregated = average_stage_metrics(per_question_metrics)

header = f"{'stage':<10}" + "".join(f"{col:>16}" for col in METRIC_COLUMNS)
print(header)
print("-" * len(header))
for stage in STAGES:
    row = f"{stage:<10}" + "".join(f"{aggregated[stage][col]:>16.3f}" for col in METRIC_COLUMNS)
    print(row)"""

CELL_BREAKDOWN_INTRO = """\
## Breakdown per întrebare

Pentru fiecare (întrebare, keyword) nefound în etapa finală (reranked), sau al \
cărui rang se schimbă cu mai mult de `RANK_SHIFT_THRESHOLD` poziții între etape, \
afișează sursa + heading-ul chunk-ului găsit (la cea mai avansată etapă unde apare) \
— diagnostic direct al eșecurilor, în stilul `show_chunk` din `retrieval_playground.ipynb`."""

CELL_BREAKDOWN = """\
RANK_SHIFT_THRESHOLD = 5  # out of RETRIEVAL_K=20, a shift bigger than this is a meaningful stage effect


def show_chunk(doc):
    source = doc.metadata.get("source", "?")
    heading = doc.metadata.get("h4") or doc.metadata.get("h3") or ""
    text = doc.metadata.get("original_text", doc.page_content)
    preview = text.strip().replace("\\n", " ")
    if len(preview) > 220:
        preview = preview[:220] + "..."
    return f"{source} | {heading}\\n       {preview}"


for test, trace in zip(tests, traces, strict=True):
    stage_docs = {
        "semantic": trace.semantic_hits,
        "lexical": trace.lexical_hits,
        "fused": trace.fused,
        "reranked": trace.reranked,
    }
    for keyword in test.keywords:
        ranks = {stage: keyword_hit_ranks(keyword, docs) for stage, docs in stage_docs.items()}
        found_ranks = [rank for rank in ranks.values() if rank is not None]
        missed_final = ranks["reranked"] is None
        shifted = bool(found_ranks) and (max(found_ranks) - min(found_ranks) > RANK_SHIFT_THRESHOLD)
        if not (missed_final or shifted):
            continue

        print(f"Q: {test.question}")
        print(f"   keyword: {keyword!r}  ranks: {ranks}")
        best_stage = next(
            (stage for stage in ("reranked", "fused", "lexical", "semantic") if ranks[stage] is not None),
            None,
        )
        if best_stage is not None:
            doc = stage_docs[best_stage][ranks[best_stage] - 1]
            print(f"   found at [{best_stage}]: {show_chunk(doc)}")
        else:
            print("   not found at any stage")
        print()"""

CELL_SAVE_INTRO = """\
## Salvare snapshot

Editează `LABEL` mai jos înainte de a rula celula (ex. `"baseline"`, `"contextualize-on"`)."""

CELL_SAVE = """\
LABEL = "baseline"

snapshot = EvalSnapshot(
    label=LABEL,
    timestamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    config={
        "RETRIEVAL_K": config.RETRIEVAL_K,
        "FINAL_K": config.FINAL_K,
        "CHAT_MODEL": config.CHAT_MODEL,
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
    },
    stage_metrics=aggregated,
)
saved_path = snapshot.save(RESULTS_DIR)
print(f"Saved snapshot: {saved_path}")"""

CELL_COMPARE_INTRO = """\
## Comparație before/after

Încarcă ultimele două snapshot-uri salvate și afișează delta per etapă/metrică."""

CELL_COMPARE = """\
snapshots = list_snapshots(RESULTS_DIR)

if len(snapshots) >= 2:
    older, newer = snapshots[-2], snapshots[-1]
    delta = diff_snapshots(older, newer)
    print(f"Comparing {older.label!r} ({older.timestamp}) -> {newer.label!r} ({newer.timestamp})")
    for stage in STAGES:
        print(f"\\n[{stage}]")
        for metric in METRIC_COLUMNS:
            value = delta[stage][metric]
            sign = "+" if value >= 0 else ""
            print(f"  {metric:<16} {sign}{value:.3f}")
else:
    print("Need at least two saved snapshots to compare — run this notebook again after a change.")"""

cells = [
    ("markdown", CELL_INTRO),
    ("code", CELL_SETUP),
    ("code", CELL_RUN),
    ("code", CELL_TABLE),
    ("markdown", CELL_BREAKDOWN_INTRO),
    ("code", CELL_BREAKDOWN),
    ("markdown", CELL_SAVE_INTRO),
    ("code", CELL_SAVE),
    ("markdown", CELL_COMPARE_INTRO),
    ("code", CELL_COMPARE),
]

nb["cells"] = [
    nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src) for kind, src in cells
]

with open("retrieval_quality.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote retrieval_quality.ipynb")
```

Run it from `backend/app/rag_simple/evaluation/` (`uv run` walks up from cwd to find `backend/pyproject.toml` and uses that venv):

```bash
cd app/rag_simple/evaluation   # relative to backend/; adjust if your shell is elsewhere
uv run python _build_notebook.py
```

Expected: `retrieval_quality.ipynb` is created in `backend/app/rag_simple/evaluation/`, 10 cells (5 markdown, 5 code), matching the `retrieval/retrieval_playground.ipynb` metadata format.

Delete the generator script — it is not a deliverable:

```bash
rm backend/app/rag_simple/evaluation/_build_notebook.py
```

- [ ] **Step 2: Start the database**

From the `full-stack-fastapi-template/` repo root (not `backend/`):

```bash
docker compose up -d db
```

Wait for it to report healthy (`docker compose ps` shows `db` as `healthy`), typically a few seconds — it's reusing the existing persistent volume, no re-ingestion needed.

- [ ] **Step 3: Execute the notebook end-to-end and inspect the result**

From `backend/`:

```bash
cd app/rag_simple/evaluation
uv run jupyter nbconvert --to notebook --execute --inplace retrieval_quality.ipynb
```

This makes real OpenAI API calls (query rewrite + rerank + embeddings for 10 questions) — expect it to take roughly a minute and cost a small, real amount.

Expected: the command exits 0 with no exception cells in the output. Then:

1. Open the executed notebook (or `jupyter nbconvert --to script --stdout retrieval_quality.ipynb` won't show outputs — instead inspect via `python -c "import json; nb=json.load(open('retrieval_quality.ipynb')); [print(''.join(o.get('text',[]))) for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[])]"` or just read the `.ipynb` JSON directly) and confirm the aggregate table cell printed 4 rows (`semantic`, `lexical`, `fused`, `reranked`) with plausible values in `[0, 1]` for every column.
2. Per the spec's own validation bar: `fused` should be `≥` `semantic`-only and `lexical`-only on most metrics — but if it isn't, that's a valid, useful finding to leave as-is (not a bug to chase); the spec explicitly treats "fusion doesn't help" as a legitimate signal, not a failure.
3. Confirm `backend/app/rag_simple/evaluation/results/` now contains exactly one `*_baseline.json` file, and that it's valid JSON with `label`, `timestamp`, `config`, `stage_metrics` keys populated.
4. Sanity-check reload: `uv run python -c "from pathlib import Path; from app.rag_simple.evaluation.results import EvalSnapshot; p = next(Path('app/rag_simple/evaluation/results').glob('*.json')); print(EvalSnapshot.load(p))"` from `backend/` — must print the snapshot without error.

If the notebook errors: this is a live-system integration point (Postgres connectivity, `OPENAI_API_KEY`, BM25 index reading `db/chunks.json`) — debug the actual failure (check `db` container health, check `.env`, check `db/chunks.json` exists) rather than modifying `retrieval_metrics.py`/`results.py` to work around it, since those were already unit-tested in Tasks 1–2 in isolation.

- [ ] **Step 4: Commit**

```bash
git add backend/app/rag_simple/evaluation/retrieval_quality.ipynb backend/app/rag_simple/evaluation/results/
git commit -m "feat(rag_simple/eval): add retrieval_quality notebook + baseline snapshot"
```
