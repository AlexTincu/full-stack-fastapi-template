import uuid

import pytest
from sqlmodel import Session

from app.knowledge_prep.docstore import ChunkDocStore
from app.knowledge_prep.tests.retrieval_fakes import add_document

SECTIONS = [(["Grâul", "A"], "text a"), (["Grâul", "B"], "text b")]


def test_mget_returns_documents_in_key_order_with_metadata(session: Session):
    a, b = add_document(session, "grau", SECTIONS)
    store = ChunkDocStore(session.get_bind())

    got = store.mget([str(b.id), str(a.id)])

    assert [d.page_content for d in got if d] == ["text b", "text a"]
    first = got[0]
    assert first is not None
    assert first.id == str(b.id)
    assert first.metadata == {
        "chunk_id": str(b.id),
        "document_id": str(b.document_id),
        "slug": "grau",
        "title": "Titlu grau",
        "heading_path": ["Grâul", "B"],
    }


def test_mget_missing_and_invalid_keys_are_none(session: Session):
    (a,) = add_document(session, "grau", SECTIONS[:1])
    store = ChunkDocStore(session.get_bind())

    got = store.mget([str(uuid.uuid4()), "nu-e-uuid", str(a.id)])

    assert got[:2] == [None, None]
    third = got[2]
    assert third is not None and third.page_content == "text a"
    assert store.mget([]) == []


def test_mget_accepts_uuid_objects(session: Session):
    # PGVectorStore pune în metadate `doc_id` ca uuid.UUID (coloana e UUID),
    # iar MultiVectorRetriever îl dă neschimbat lui mget.
    (a,) = add_document(session, "grau", SECTIONS[:1])
    (got,) = ChunkDocStore(session.get_bind()).mget([a.id])  # type: ignore
    assert got is not None and got.metadata["chunk_id"] == str(a.id)


def test_store_is_read_only(session: Session):
    store = ChunkDocStore(session.get_bind())
    with pytest.raises(NotImplementedError, match="index"):
        store.mset([])
    with pytest.raises(NotImplementedError, match="index"):
        store.mdelete([])


def test_yield_keys_lists_chunk_ids(session: Session):
    a, b = add_document(session, "grau", SECTIONS)
    store = ChunkDocStore(session.get_bind())
    assert set(store.yield_keys()) == {str(a.id), str(b.id)}
    assert list(store.yield_keys(prefix=str(a.id))) == [str(a.id)]
