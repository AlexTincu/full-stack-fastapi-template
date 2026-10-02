"""Tabelul `knowledge_vectors`: vectorii-copii ai chunk-urilor (spec 2026-09-25, „Tabelul de vectori”).

Un rând per vector: `content` (id = chunk.id) sau `question` (id = question.id). Coloanele
filtrabile: `doc_id` (chunk-ul părinte, cheia lui MultiVectorRetriever din faza 4),
`document_id` și `kind`.

Tabelul îl creează `PGEngine.init_vectorstore_table` la prima folosire (ca în `rag_simple`), nu
Alembic: schema e a lui langchain_postgres, iar `alembic/env.py` îl exclude din autogenerate.
Fără index ANN: pgvector indexează `vector` doar până la 2000 de dimensiuni (aici 3072), iar la
câteva mii de rânduri scanarea exactă e suficientă.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import openai
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, ColumnDict, PGEngine, PGVectorStore
from pydantic import SecretStr
from sqlalchemy import create_engine, inspect
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.knowledge_prep.converters.gemini import log_retry

TABLE_NAME = "knowledge_vectors"
EMBEDDING_MODEL = "text-embedding-3-large"  # același model ca rag_simple
EMBEDDING_DIMENSIONS = 3072
METADATA_COLUMNS = ["doc_id", "document_id", "kind"]


@dataclass(frozen=True)
class VectorEntry:
    id: uuid.UUID  # chunk.id pentru content, question.id pentru question
    doc_id: uuid.UUID  # chunk-ul părinte
    document_id: uuid.UUID
    kind: Literal["content", "question"]
    text: str  # textul embedded


class VectorIndex(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def delete_chunks(self, chunk_ids: Sequence[uuid.UUID]) -> None: ...

    def add(
        self, entries: Sequence[VectorEntry], embeddings: Sequence[list[float]]
    ) -> None: ...


def _metadata_columns() -> list[Column | ColumnDict]:
    # obiecte noi la fiecare apel: init_vectorstore_table le modifică numele pe loc
    return [
        Column("doc_id", "UUID", nullable=False),
        Column("document_id", "UUID", nullable=False),
        Column("kind", "VARCHAR(20)", nullable=False),
    ]


def _table_exists(url: str, table_name: str) -> bool:
    engine = create_engine(url)
    try:
        return inspect(engine).has_table(table_name)
    finally:
        engine.dispose()


class PgVectorIndex:
    def __init__(
        self,
        url: str,
        embeddings: Embeddings,
        *,
        table_name: str = TABLE_NAME,
        vector_size: int = EMBEDDING_DIMENSIONS,
    ):
        self.table_name = table_name
        self._embeddings = embeddings
        engine = PGEngine.from_connection_string(url=url)
        # init_vectorstore_table face CREATE TABLE simplu: doar dacă nu există deja
        if not _table_exists(url, table_name):
            engine.init_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
                metadata_columns=_metadata_columns(),
            )
        self.store = PGVectorStore.create_sync(
            engine=engine,
            embedding_service=embeddings,
            table_name=table_name,
            metadata_columns=METADATA_COLUMNS,
        )

    @classmethod
    def from_settings(cls) -> PgVectorIndex:
        """Baza și cheia OpenAI din `.env`."""
        from app.core.config import settings

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, api_key=SecretStr(settings.OPENAI_API_KEY)
        )
        return cls(str(settings.DATABASE_URL), embeddings)

    @retry(
        retry=retry_if_exception_type(openai.APIError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(3),
        before_sleep=log_retry,
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts) if texts else []

    def delete_chunks(self, chunk_ids: Sequence[uuid.UUID]) -> None:
        """Șterge vectorii-copii ai chunk-urilor (după `doc_id`, deci și resturi de la o rulare întreruptă)."""
        if chunk_ids:
            self.store.delete(filter={"doc_id": {"$in": [str(c) for c in chunk_ids]}})

    def add(
        self, entries: Sequence[VectorEntry], embeddings: Sequence[list[float]]
    ) -> None:
        if not entries:
            return
        self.store.add_embeddings(
            texts=[e.text for e in entries],
            embeddings=list(embeddings),
            metadatas=[
                {
                    "doc_id": str(e.doc_id),
                    "document_id": str(e.document_id),
                    "kind": e.kind,
                }
                for e in entries
            ],
            ids=[str(e.id) for e in entries],
        )
