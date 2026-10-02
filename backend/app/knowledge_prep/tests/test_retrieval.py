import pytest
from langchain_core.documents import Document
from sqlmodel import Session

from app.knowledge_prep.models import Chunk
from app.knowledge_prep.retrieval import (
    CHILDREN_PER_CHUNK,
    build_retriever,
    reciprocal_rank_fusion,
)
from app.knowledge_prep.tests.retrieval_fakes import (
    FakeChildStore,
    add_document,
    child,
)


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


@pytest.fixture
def chunks(session: Session) -> list[Chunk]:
    return add_document(
        session,
        "grau",
        [
            (["Grâul", "Beneficii"], "Grâul încolțit dă putere."),
            (["Grâul", "Preparare"], "Boabele se țin la apă o zi."),
            (["Grâul", "Contraindicații"], "Nu se dă copiilor mici."),
            (["Grâul", "Istoric"], "Metoda zxqvortex a fost descrisă de Popa."),
        ],
    )


def test_rrf_sums_reciprocal_ranks():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert reciprocal_rank_fusion([["x", "y"]]) == ["x", "y"]
    assert reciprocal_rank_fusion([["x"], ["y"]]) == [
        "x",
        "y",
    ]  # egalitate: prima listă
    assert reciprocal_rank_fusion([]) == []


def test_multi_vector_dedupes_children_and_asks_for_more_children_than_k(
    session: Session, chunks: list[Chunk]
):
    c1, c2, c3, _ = chunks
    store = FakeChildStore(
        [
            child(c2, "question"),
            child(c2, "content"),
            child(c1, "question"),
            child(c3, "content"),
        ]
    )
    retriever = build_retriever(
        store, session.get_bind(), k=2, hybrid=False, expand=False
    )

    assert ids(retriever.invoke("orice")) == [str(c2.id), str(c1.id)]
    assert store.calls == [
        {"query": "orice", "k": 2 * CHILDREN_PER_CHUNK, "filter": None}
    ]


def test_baseline_searches_only_content_vectors(session: Session, chunks: list[Chunk]):
    c1, c2, c3, _ = chunks
    store = FakeChildStore(
        [
            child(c2, "question"),
            child(c1, "question"),
            child(c3, "content"),
            child(c2, "content"),
        ]
    )
    retriever = build_retriever(
        store, session.get_bind(), k=5, questions=False, hybrid=False, expand=False
    )

    assert ids(retriever.invoke("orice")) == [str(c3.id), str(c2.id)]
    assert store.calls[0]["filter"] == {"kind": "content"}


def test_many_questions_of_one_chunk_do_not_crowd_out_the_others(
    session: Session, chunks: list[Chunk]
):
    c1, c2, *_ = chunks
    children = [child(c1, "question") for _ in range(5)]
    children += [child(c1, "content"), child(c2, "content")]
    retriever = build_retriever(
        FakeChildStore(children), session.get_bind(), k=2, hybrid=False, expand=False
    )
    assert ids(retriever.invoke("orice")) == [str(c1.id), str(c2.id)]


def test_hybrid_adds_bm25_hits_by_rrf_on_chunk_id(
    session: Session, chunks: list[Chunk]
):
    c1, c2, c3, c4 = chunks
    store = FakeChildStore(
        [child(c2, "content"), child(c1, "content"), child(c3, "content")]
    )
    bind = session.get_bind()

    # semantic [c2, c1, c3] + BM25 [c4] → RRF: c2 (1/61, primul la egalitate), c4 (1/61), ...
    assert ids(build_retriever(store, bind, k=2, expand=False).invoke("zxqvortex")) == [
        str(c2.id),
        str(c4.id),
    ]
    assert ids(
        build_retriever(store, bind, k=2, hybrid=False, expand=False).invoke(
            "zxqvortex"
        )
    ) == [
        str(c2.id),
        str(c1.id),
    ]


def test_chunks_missing_from_db_are_skipped(session: Session, chunks: list[Chunk]):
    c1, c2, c3, c4 = chunks
    store = FakeChildStore(
        [child(c1, "content"), child(c2, "content"), child(c3, "content")]
    )
    retriever = build_retriever(
        store, session.get_bind(), k=2, expand=False
    )  # BM25 îl conține pe c4
    session.delete(c1)  # vector orfan
    session.delete(c4)  # dispărut din BM25 după construire
    session.commit()

    assert ids(retriever.invoke("zxqvortex")) == [str(c2.id), str(c3.id)]


def test_empty_index_returns_nothing(session: Session):
    retriever = build_retriever(FakeChildStore([]), session.get_bind(), k=3)
    assert retriever.invoke("grâu") == []


def test_k_must_be_positive(session: Session):
    with pytest.raises(ValueError, match="k"):
        build_retriever(FakeChildStore([]), session.get_bind(), k=0)
