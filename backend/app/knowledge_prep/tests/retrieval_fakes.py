"""Fake-uri și helper-e pentru testele de retrieval (fără pgvector, fără OpenAI)."""

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlmodel import Session

from app.knowledge_prep import models
from app.knowledge_prep.models import Chunk, DocumentType, Stage
from app.knowledge_prep.vectors import VectorEntry


def add_document(
    session: Session, slug: str, sections: list[tuple[list[str], str]]
) -> list[Chunk]:
    """Un document `indexed` cu chunk-urile date ca (heading_path, text), în ordine."""
    doc = models.Document(
        slug=slug,
        title=f"Titlu {slug}",
        type=DocumentType.carte,
        split=True,
        stage=Stage.indexed,
        markdown_path=f"/nu-exista/{slug}.md",
    )
    session.add(doc)
    session.flush()
    chunks = [
        Chunk(document_id=doc.id, position=i, heading_path=path, text=text)
        for i, (path, text) in enumerate(sections)
    ]
    session.add_all(chunks)
    session.commit()
    for c in chunks:
        session.refresh(c)
    return chunks


def child(chunk: Chunk, kind: str, text: str = "copil") -> Document:
    """Un vector-copil cum îl întoarce PGVectorStore: `doc_id`/`document_id` ca uuid.UUID."""
    return Document(
        page_content=text,
        metadata={"doc_id": chunk.id, "document_id": chunk.document_id, "kind": kind},
    )


class FakeChildStore(VectorStore):
    """Copiii în ordinea dată (primul = cel mai apropiat), indiferent de query.

    Filtrul se aplică la fel ca în PGVectorStore pentru egalități simple (`{"kind": "content"}`).
    """

    def __init__(self, children: list[Document]):
        self.children = children
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_entries(cls, entries: Iterable[VectorEntry]) -> FakeChildStore:
        return cls(
            [
                Document(
                    page_content=e.text,
                    metadata={
                        "doc_id": e.doc_id,
                        "document_id": e.document_id,
                        "kind": e.kind,
                    },
                )
                for e in entries
            ]
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        flt = kwargs.get("filter")
        self.calls.append({"query": query, "k": k, "filter": flt})
        hits = [
            d
            for d in self.children
            if all(d.metadata.get(key) == v for key, v in (flt or {}).items())
        ]
        return hits[:k]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FakeChildStore:
        raise NotImplementedError
