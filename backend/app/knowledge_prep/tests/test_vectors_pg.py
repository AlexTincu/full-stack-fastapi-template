"""Integrare pe Postgres cu pgvector (spec, „Testare”): ștergerea și refacerea vectorilor unui document.

Se sare dacă baza din `.env` nu răspunde (`docker compose up -d db` din rădăcina full-stack).
"""

import uuid
from collections.abc import Iterator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.knowledge_prep.tests.pg import PG_URL, drop_table, requires_pg
from app.knowledge_prep.vectors import PgVectorIndex, VectorEntry

pytestmark = requires_pg
SIZE = 8


def make_index(table: str) -> PgVectorIndex:
    assert PG_URL is not None
    return PgVectorIndex(
        PG_URL,
        DeterministicFakeEmbedding(size=SIZE),
        table_name=table,
        vector_size=SIZE,
    )


@pytest.fixture
def pg() -> Iterator[PgVectorIndex]:
    assert PG_URL is not None
    table = f"knowledge_vectors_test_{uuid.uuid4().hex[:8]}"
    yield make_index(table)
    drop_table(table)


def entries_for(document_id: uuid.UUID, n_chunks: int = 2) -> list[VectorEntry]:
    out = []
    for i in range(n_chunks):
        chunk_id = uuid.uuid4()
        out.append(
            VectorEntry(chunk_id, chunk_id, document_id, "content", f"chunk {i}")
        )
        out.append(
            VectorEntry(
                uuid.uuid4(), chunk_id, document_id, "question", f"întrebarea {i}?"
            )
        )
    return out


def put(pg: PgVectorIndex, document_id: uuid.UUID, entries: list[VectorEntry]) -> None:
    pg.replace_document(document_id, entries, pg.embed([e.text for e in entries]))


def stored(pg: PgVectorIndex, entries: list[VectorEntry]) -> dict[str, dict]:
    return {
        str(d.id): d.metadata for d in pg.store.get_by_ids([str(e.id) for e in entries])
    }


def test_replace_document_swaps_only_that_documents_vectors(pg: PgVectorIndex):
    a, b = uuid.uuid4(), uuid.uuid4()
    old_a, only_b = entries_for(a), entries_for(b)
    put(pg, a, old_a)
    put(pg, b, only_b)

    new_a = entries_for(a, n_chunks=3)
    put(pg, a, new_a)

    assert stored(pg, old_a) == {}
    assert len(stored(pg, new_a)) == 6
    b_rows = stored(pg, only_b)
    assert len(b_rows) == 4
    question = only_b[1]
    meta = b_rows[str(question.id)]
    assert (str(meta["doc_id"]), str(meta["document_id"]), meta["kind"]) == (
        str(question.doc_id),
        str(b),
        "question",
    )


def test_kind_filter_works_and_existing_table_is_reused(pg: PgVectorIndex):
    a = uuid.uuid4()
    put(pg, a, entries_for(a))
    docs = pg.store.similarity_search("chunk", k=10, filter={"kind": "content"})
    assert len(docs) == 2
    assert {d.metadata["kind"] for d in docs} == {"content"}

    again = make_index(pg.table_name)  # tabelul există: nu se recreează, datele rămân
    assert len(again.store.similarity_search("chunk", k=10)) == 4


def test_replace_document_with_no_entries_only_deletes(pg: PgVectorIndex):
    a = uuid.uuid4()
    entries = entries_for(a)
    put(pg, a, entries)
    pg.replace_document(a, [], [])
    assert stored(pg, entries) == {}
