"""Evaluarea retrieval-ului (spec 2026-09-25, „Retrieval” → „Evaluare”).

Aceleași întrebări de test trec prin mai multe moduri ale lui `build_retriever` (baseline /
multi-vector, cu/fără BM25), pe chunk-uri (`expand=False`: părinții mari ar umfla metricile).
Relevanța e aproximată ca în `rag_simple`: un rezultat e relevant pentru un cuvânt-cheie dacă îl
conține. Metricile au aceleași definiții ca `rag_simple/evaluation/retrieval_metrics.py`, dar nu
se importă de acolo: importul acela încarcă modelele de chat și BM25-ul din `rag_simple` (~10 s,
cere cheile API).

Diferență față de `rag_simple`: potrivirea ignoră diacriticele și spațiile multiple. Cuvintele-cheie
din `tests.jsonl` au fost scrise pe textele vechi, adesea fără diacritice („2-5 saptamani”), iar
markdown-ul restructurat le are („2-5 săptămâni”).

Întrebările ale căror cuvinte-cheie nu apar în niciun chunk (documentul lor nu e indexat) se sar
implicit: ar da 0 în toate modurile și ar tasa diferențele.
"""

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel
from sqlmodel import Session, select

from app.knowledge_prep.docstore import Bind
from app.knowledge_prep.models import Chunk

EVAL_K = 10  # metricile merg până la @10
DEFAULT_TESTS = (
    Path(__file__).resolve().parents[1] / "rag_simple" / "evaluation" / "tests.jsonl"
)
RESULTS_DIR = Path(__file__).parent / "eval_results"

# mod → (questions, hybrid) pentru `build_retriever`
MODES: dict[str, tuple[bool, bool]] = {
    "baseline": (False, False),
    "baseline+bm25": (False, True),
    "multi-vector": (True, False),
    "multi-vector+bm25": (True, True),
}
METRICS = (
    "mrr",
    "ndcg",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "hit_rate_at_10",
)

_SPACES = re.compile(r"\s+")


class TestCase(BaseModel):
    """Un rând din `tests.jsonl` (același format ca `rag_simple.evaluation.test.TestQuestion`)."""

    __test__ = False  # nu e o clasă de teste pytest

    question: str
    keywords: list[str]
    reference_answer: str = ""
    category: str = ""


def load_tests(path: Path = DEFAULT_TESTS) -> list[TestCase]:
    """Întrebările din fișierul JSONL; un rând invalid oprește citirea cu numărul lui."""
    tests = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tests.append(TestCase(**json.loads(line)))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path}:{n}: rând invalid: {e}") from e
    return tests


def fold(text: str) -> str:
    """Litere mici, fără diacritice, spații simple: forma în care se compară cuvintele-cheie."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACES.sub(" ", plain).strip()


def _contains(keyword: str, doc: Document) -> bool:
    return fold(keyword) in fold(doc.page_content)


def keyword_rank(keyword: str, docs: Sequence[Document]) -> int | None:
    """Rangul (de la 1) primului rezultat care conține cuvântul-cheie, sau None."""
    return next((r for r, d in enumerate(docs, 1) if _contains(keyword, d)), None)


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg(keyword: str, docs: Sequence[Document], k: int) -> float:
    relevances = [int(_contains(keyword, d)) for d in docs[:k]]
    ideal = _dcg(sorted(relevances, reverse=True))
    return _dcg(relevances) / ideal if ideal else 0.0


def _recall(keywords: Sequence[str], docs: Sequence[Document], k: int) -> float:
    return sum(keyword_rank(kw, docs[:k]) is not None for kw in keywords) / len(
        keywords
    )


def _precision(keywords: Sequence[str], docs: Sequence[Document], k: int) -> float:
    top = docs[:k]
    if not top:
        return 0.0
    return sum(sum(_contains(kw, d) for d in top) / len(top) for kw in keywords) / len(
        keywords
    )


def question_metrics(
    keywords: Sequence[str], docs: Sequence[Document]
) -> dict[str, float]:
    """Metricile unei întrebări, ca `rag_simple.evaluation.retrieval_metrics.evaluate_stage`."""
    if not keywords:
        return dict.fromkeys(METRICS, 0.0)
    ranks = [keyword_rank(kw, docs) for kw in keywords]
    return {
        "mrr": sum(1 / r if r else 0.0 for r in ranks) / len(keywords),
        "ndcg": sum(_ndcg(kw, docs, 10) for kw in keywords) / len(keywords),
        "recall_at_5": _recall(keywords, docs, 5),
        "recall_at_10": _recall(keywords, docs, 10),
        "precision_at_5": _precision(keywords, docs, 5),
        "precision_at_10": _precision(keywords, docs, 10),
        "hit_rate_at_10": float(any(r is not None and r <= 10 for r in ranks)),
    }


def split_answerable(
    tests: Iterable[TestCase], texts: Iterable[str]
) -> tuple[list[TestCase], list[TestCase]]:
    """(întrebări cu cel puțin un cuvânt-cheie într-un text, restul)."""
    corpus = [fold(t) for t in texts]
    answerable, skipped = [], []
    for t in tests:
        hit = any(fold(kw) in text for kw in t.keywords for text in corpus)
        (answerable if hit else skipped).append(t)
    return answerable, skipped


def chunk_texts(bind: Bind) -> list[str]:
    with Session(bind) as s:
        return list(s.exec(select(Chunk.text)))


@dataclass
class QuestionResult:
    test: TestCase
    # mod → metrică → valoare
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class EvalReport:
    results: list[QuestionResult]
    skipped: list[TestCase] = field(default_factory=list)

    @property
    def modes(self) -> list[str]:
        return list(self.results[0].metrics) if self.results else []

    def averages(self, category: str | None = None) -> dict[str, dict[str, float]]:
        """mod → metrică → media pe întrebări (doar din `category`, dacă e dată)."""
        rows = [r for r in self.results if category in (None, r.test.category)]
        if not rows:
            return {}
        return {
            mode: {
                m: sum(r.metrics[mode][m] for r in rows) / len(rows) for m in METRICS
            }
            for mode in self.modes
        }

    def categories(self) -> list[str]:
        return sorted({r.test.category for r in self.results})


def evaluate(
    retrievers: Mapping[str, BaseRetriever],
    tests: Sequence[TestCase],
    skipped: Sequence[TestCase] = (),
) -> EvalReport:
    """Fiecare întrebare prin fiecare retriever (mod); retriever-ele întorc chunk-uri în ordinea rangului."""
    results = []
    for t in tests:
        result = QuestionResult(t)
        for mode, retriever in retrievers.items():
            result.metrics[mode] = question_metrics(
                t.keywords, retriever.invoke(t.question)
            )
        results.append(result)
    return EvalReport(results, list(skipped))
