"""Integrare pe Postgres (spec, „Testare”): ChunkDocStore, BM25 și MultiVectorRetriever pe date reale.

Scrie un document `retrieval-test-<hex>` în tabelele reale `document`/`chunk` (șters la final,
chunk-urile în cascadă) și un tabel propriu `knowledge_vectors_test_<hex>` (șters la final).
Nu atinge `knowledge_vectors`. Se sare fără Postgres sau fără migrarea fazei 1.
"""

import uuid
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import Engine, create_engine
from sqlmodel import Session

from app.knowledge_prep import models
from app.knowledge_prep.chunking import embedding_text
from app.knowledge_prep.docstore import ChunkDocStore
from app.knowledge_prep.lexical import ChunkBM25
from app.knowledge_prep.models import Chunk
from app.knowledge_prep.retrieval import build_retriever
from app.knowledge_prep.tests.pg import PG_URL, drop_table, has_table, requires_pg
from app.knowledge_prep.tests.retrieval_fakes import add_document
from app.knowledge_prep.vectors import PgVectorIndex, VectorEntry

pytestmark = [
    requires_pg,
    pytest.mark.skipif(
        not has_table("chunk"), reason="lipsește tabelul chunk (alembic upgrade head)"
    ),
]
SIZE = 8
QUESTION = "Ce cereală se macină pentru testul de retrieval?"


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert PG_URL is not None
    e = create_engine(PG_URL)
    yield e
    e.dispose()


@pytest.fixture
def chunks(engine: Engine) -> Iterator[list[Chunk]]:
    slug = f"retrieval-test-{uuid.uuid4().hex[:8]}"
    with Session(engine) as s:
        made = add_document(
            s,
            slug,
            [
                (["Test", "Ovăz"], "Ovăzul zxqvortex se fierbe încet."),
                (["Test", "Orz"], "Orzul se macină."),
                (["Test", "Mei"], "Meiul se spală bine."),
            ],
        )
        s.expunge_all()
    try:
        yield made
    finally:
        with Session(engine) as s:
            doc = s.get(models.Document, made[0].document_id)
            if doc is not None:
                s.delete(doc)  # chunk-urile: ON DELETE CASCADE
                s.commit()


@pytest.fixture
def index(chunks: list[Chunk]) -> Iterator[PgVectorIndex]:
    assert PG_URL is not None
    table = f"knowledge_vectors_test_{uuid.uuid4().hex[:8]}"
    idx = PgVectorIndex(
        PG_URL,
        DeterministicFakeEmbedding(size=SIZE),
        table_name=table,
        vector_size=SIZE,
    )
    document_id = chunks[0].document_id
    entries = [
        VectorEntry(
            c.id, c.id, document_id, "content", embedding_text(c.heading_path, c.text)
        )
        for c in chunks
    ]
    entries.append(
        VectorEntry(uuid.uuid4(), chunks[1].id, document_id, "question", QUESTION)
    )
    idx.replace_document(document_id, entries, idx.embed([e.text for e in entries]))
    try:
        yield idx
    finally:
        drop_table(table)


def test_docstore_reads_postgres_rows_and_uuid_keys(
    engine: Engine, chunks: list[Chunk]
):
    a = chunks[0]
    got = ChunkDocStore(engine).mget([a.id, "nu-e-uuid", str(uuid.uuid4())])  # type: ignore
    assert got[1:] == [None, None]
    first = got[0]
    assert first is not None
    assert first.page_content == "Ovăzul zxqvortex se fierbe încet."
    assert first.metadata["heading_path"] == ["Test", "Ovăz"]
    assert first.metadata["slug"].startswith("retrieval-test-")


def test_bm25_from_postgres_finds_rare_word(engine: Engine, chunks: list[Chunk]):
    assert ChunkBM25.from_db(engine).search("zxqvortex", 3) == [str(chunks[0].id)]


def test_multi_vector_finds_chunk_through_its_question(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    retriever = build_retriever(index.store, engine, k=1, hybrid=False)
    assert ids(retriever.invoke(QUESTION)) == [str(chunks[1].id)]


def test_baseline_finds_chunk_by_its_content_vector(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    target = chunks[2]
    retriever = build_retriever(index.store, engine, k=1, questions=False, hybrid=False)
    query = embedding_text(target.heading_path, target.text)
    assert ids(retriever.invoke(query)) == [str(target.id)]


def test_hybrid_brings_in_the_bm25_hit(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    retriever = build_retriever(index.store, engine, k=2)
    assert str(chunks[0].id) in ids(retriever.invoke("zxqvortex"))
