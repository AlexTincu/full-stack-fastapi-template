from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.knowledge_prep.models  # noqa: F401  (înregistrează tabelele în metadata)


@pytest.fixture
def session() -> Iterator[Session]:
    """Registrul pe SQLite in-memory: testele nu au nevoie de Postgres."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    tables = [SQLModel.metadata.tables[t] for t in ("document", "chunk", "question")]
    SQLModel.metadata.create_all(engine, tables=tables)
    with Session(engine) as s:
        yield s
