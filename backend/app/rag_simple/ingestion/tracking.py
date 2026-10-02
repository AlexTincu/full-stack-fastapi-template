"""Record of ingested documents, in a table next to the vectors (`rag_simple_documents`).

One row per source file: its content hash, how many chunks it produced and when. Ingest uses
it to skip files it has already seen, so source files stay where they are (no more moving them
to `ingested/`). The table is created here, outside Alembic, like `rag_simple_chunks`.
"""

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    inspect,
    select,
    text,
)

from app.rag_simple.config import DOCUMENTS_TABLE_NAME, PGVECTOR_TABLE_NAME
from app.rag_simple.ingestion.load import SourceDocument

# Own MetaData, not SQLModel's: Alembic must not manage (or drop) this table.
_metadata = MetaData()
documents_table = Table(
    DOCUMENTS_TABLE_NAME,
    _metadata,
    Column("source", String, primary_key=True),
    # NULL for documents ingested before this table existed (adopted from the chunks)
    Column("content_hash", String(64)),
    Column("chunks", Integer, nullable=False),
    Column("embedding_model", String),
    Column("ingested_at", DateTime(timezone=True)),
)


@dataclass(frozen=True)
class IngestedDocument:
    source: str
    content_hash: str | None
    chunks: int
    embedding_model: str | None
    ingested_at: datetime | None


@dataclass
class IngestPlan:
    to_ingest: list[SourceDocument] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    # already ingested with other content (or with no known hash): re-ingesting would
    # duplicate its chunks, since replacing them isn't supported
    changed: list[str] = field(default_factory=list)


def content_hash(text_: str) -> str:
    return hashlib.sha256(text_.encode("utf-8")).hexdigest()


def plan_ingest(
    documents: Iterable[SourceDocument], tracked: Mapping[str, IngestedDocument]
) -> IngestPlan:
    plan = IngestPlan()
    for doc in documents:
        record = tracked.get(doc.source)
        if record is None:
            plan.to_ingest.append(doc)
        elif record.content_hash == content_hash(doc.text):
            plan.unchanged.append(doc.source)
        else:
            plan.changed.append(doc.source)
    return plan


def ensure_documents_table(engine: Engine) -> None:
    _metadata.create_all(engine, checkfirst=True)


def chunk_counts(engine: Engine) -> dict[str, int]:
    """Chunks per source, as actually stored in the vectorstore table (Postgres only)."""
    if not inspect(engine).has_table(PGVECTOR_TABLE_NAME):
        return {}
    query = text(
        f"SELECT langchain_metadata->>'source', count(*) FROM {PGVECTOR_TABLE_NAME} GROUP BY 1"
    )
    with engine.connect() as conn:
        return dict(conn.execute(query).tuples().all())


def tracked_documents(engine: Engine) -> dict[str, IngestedDocument]:
    with engine.connect() as conn:
        rows = conn.execute(select(documents_table).order_by(documents_table.c.source))
        return {row.source: IngestedDocument(**row._mapping) for row in rows}


def adopt_untracked(engine: Engine, counts: Mapping[str, int]) -> list[str]:
    """Record sources that have chunks but no row (ingested before tracking), without a hash."""
    untracked = sorted(set(counts) - set(tracked_documents(engine)))
    if untracked:
        with engine.begin() as conn:
            conn.execute(
                documents_table.insert(),
                [{"source": s, "chunks": counts[s]} for s in untracked],
            )
    return untracked


def record_ingested(
    engine: Engine,
    documents: Iterable[SourceDocument],
    chunks_per_source: Mapping[str, int],
    embedding_model: str,
) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "source": doc.source,
            "content_hash": content_hash(doc.text),
            "chunks": chunks_per_source.get(doc.source, 0),
            "embedding_model": embedding_model,
            "ingested_at": now,
        }
        for doc in documents
    ]
    if rows:
        with engine.begin() as conn:
            conn.execute(documents_table.insert(), rows)
