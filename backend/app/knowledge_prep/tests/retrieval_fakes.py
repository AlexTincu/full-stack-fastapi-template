"""Fake-uri și helper-e pentru testele de retrieval (fără pgvector, fără OpenAI)."""

from sqlmodel import Session

from app.knowledge_prep import models
from app.knowledge_prep.models import Chunk, DocumentType, Stage


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
