import pytest
from sqlalchemy import create_engine

from app.rag_simple.ingestion.load import SourceDocument
from app.rag_simple.ingestion.tracking import (
    adopt_untracked,
    content_hash,
    ensure_documents_table,
    plan_ingest,
    record_ingested,
    tracked_documents,
)


@pytest.fixture
def engine():
    # the tracking table is portable SQL; only chunk_counts() needs Postgres
    engine = create_engine("sqlite://")
    ensure_documents_table(engine)
    yield engine
    engine.dispose()


def test_record_and_read_back(engine):
    docs = [SourceDocument("a.md", "alpha"), SourceDocument("b.md", "beta")]
    record_ingested(engine, docs, {"a.md": 3, "b.md": 1}, "openai:emb")

    tracked = tracked_documents(engine)
    assert list(tracked) == ["a.md", "b.md"]
    assert tracked["a.md"].chunks == 3
    assert tracked["a.md"].content_hash == content_hash("alpha")
    assert tracked["a.md"].embedding_model == "openai:emb"
    assert tracked["a.md"].ingested_at is not None


def test_ensure_table_is_idempotent(engine):
    record_ingested(engine, [SourceDocument("a.md", "alpha")], {"a.md": 1}, "m")
    ensure_documents_table(engine)
    assert list(tracked_documents(engine)) == ["a.md"]


def test_plan_splits_new_unchanged_changed(engine):
    record_ingested(
        engine,
        [SourceDocument("same.md", "x"), SourceDocument("edited.md", "old")],
        {"same.md": 1, "edited.md": 1},
        "m",
    )
    new = SourceDocument("new.md", "n")
    plan = plan_ingest(
        [new, SourceDocument("same.md", "x"), SourceDocument("edited.md", "new")],
        tracked_documents(engine),
    )
    assert plan.to_ingest == [new]
    assert plan.unchanged == ["same.md"]
    assert plan.changed == ["edited.md"]


def test_adopt_records_only_untracked_sources_without_hash(engine):
    record_ingested(engine, [SourceDocument("a.md", "alpha")], {"a.md": 2}, "m")

    assert adopt_untracked(engine, {"a.md": 2, "legacy.md": 7}) == ["legacy.md"]
    assert adopt_untracked(engine, {"a.md": 2, "legacy.md": 7}) == []

    legacy = tracked_documents(engine)["legacy.md"]
    assert (legacy.chunks, legacy.content_hash, legacy.ingested_at) == (7, None, None)


def test_adopted_source_is_not_reingested(engine):
    adopt_untracked(engine, {"legacy.md": 7})
    plan = plan_ingest([SourceDocument("legacy.md", "text")], tracked_documents(engine))
    assert plan.to_ingest == []
    assert plan.changed == ["legacy.md"]
