"""Părinții lui MultiVectorRetriever: chunk-urile din tabelul `chunk` (spec 2026-09-25, „Retrieval”).

Doar citire: chunk-urile se scriu numai prin `index` (faza 3), deci `mset`/`mdelete` ridică
`NotImplementedError`. Fiecare apel deschide o sesiune scurtă pe `bind`, ca retriever-ul să poată
trăi mai mult decât o sesiune (ex. într-un server).
"""

import uuid
from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from sqlalchemy import Connection, Engine
from sqlmodel import Session, col, select

from app.knowledge_prep import models
from app.knowledge_prep.models import Chunk

Bind = Engine | Connection

_READ_ONLY = "ChunkDocStore e doar pentru citire: chunk-urile se scriu prin `index`."


def _as_uuid(key: object) -> uuid.UUID | None:
    # PGVectorStore întoarce `doc_id` ca uuid.UUID, restul lumii ca str
    if isinstance(key, uuid.UUID):
        return key
    try:
        return uuid.UUID(str(key))
    except ValueError:
        return None


def chunk_document(chunk: Chunk, doc: models.Document) -> Document:
    return Document(
        id=str(chunk.id),
        page_content=chunk.text,
        metadata={
            "chunk_id": str(chunk.id),
            "document_id": str(doc.id),
            "slug": doc.slug,
            "title": doc.title,
            "heading_path": list(chunk.heading_path),
        },
    )


class ChunkDocStore(BaseStore[str, Document]):
    def __init__(self, bind: Bind):
        self.bind = bind

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        """Chunk-urile cerute, în ordinea cheilor; `None` pentru o cheie inexistentă sau invalidă."""
        wanted = [_as_uuid(k) for k in keys]
        ids = {u for u in wanted if u is not None}
        if not ids:
            return [None] * len(keys)
        with Session(self.bind) as s:
            rows = s.exec(
                select(Chunk, models.Document)
                .join(
                    models.Document, col(Chunk.document_id) == col(models.Document.id)
                )
                .where(col(Chunk.id).in_(ids))
            ).all()
            found = {chunk.id: chunk_document(chunk, doc) for chunk, doc in rows}
        return [found.get(u) if u is not None else None for u in wanted]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        raise NotImplementedError(_READ_ONLY)

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError(_READ_ONLY)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        with Session(self.bind) as s:
            ids = [str(i) for i in s.exec(select(Chunk.id).order_by(col(Chunk.id)))]
        return (i for i in ids if prefix is None or i.startswith(prefix))
