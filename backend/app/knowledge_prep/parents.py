"""Părinții trimiși LLM-ului: mai mari decât chunk-ul găsit (decizia 2026-09-29).

Căutarea rămâne pe chunk-uri mici (potrivire precisă), dar răspunsul are nevoie de context
întreg: la „colesterol” LLM-ul trebuie să vadă și dozele, și contraindicațiile curei. Fiecare
chunk găsit, în ordinea rangului, se extinde la cea mai largă unitate care încape în
PARENT_MAX_TOKENS:
1. documentul întreg;
2. altfel, secțiunea strămoșă cea mai largă (prefix din `heading_path`, de la rădăcină în jos);
3. altfel, chunk-ul cu vecinii lui (± NEIGHBORS poziții) din aceeași secțiune.

Părinții se deduplică: al doilea chunk găsit în același document/aceeași secțiune nu mai ocupă
un loc din `k`, doar se adaugă la `matched_chunk_ids`. Chunk-urile tăiate din aceeași secțiune
mare (> MAX_CHUNK_TOKENS) au CHUNK_OVERLAP_TOKENS în comun, iar textul lipit le repetă.
"""

import uuid
from collections.abc import Sequence

from langchain_core.documents import Document
from sqlmodel import Session, col, select

from app.knowledge_prep import models
from app.knowledge_prep.chunking import count_tokens
from app.knowledge_prep.docstore import Bind, as_uuid
from app.knowledge_prep.models import Chunk

PARENT_MAX_TOKENS = 4000
NEIGHBORS = 1


def _join(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(c.text for c in chunks)


def _fits(chunks: Sequence[Chunk], max_tokens: int) -> bool:
    return count_tokens(_join(chunks)) <= max_tokens


def parent_of(
    chunks: Sequence[Chunk], i: int, title: str, max_tokens: int = PARENT_MAX_TOKENS
) -> tuple[str, list[str], list[Chunk]]:
    """(scope, heading_path, chunk-urile părintelui) pentru `chunks[i]`; `chunks` = tot documentul, în ordine."""
    if _fits(chunks, max_tokens):
        return "document", [title], list(chunks)
    path = chunks[i].heading_path
    for depth in range(1, len(path) + 1):
        prefix = path[:depth]
        section = [c for c in chunks if c.heading_path[:depth] == prefix]
        if _fits(section, max_tokens):
            return "section", prefix, section
    window = [
        c
        for c in chunks[max(0, i - NEIGHBORS) : i + NEIGHBORS + 1]
        if c.heading_path == path
    ]
    return "chunk", path, window


def expand_to_parents(
    bind: Bind,
    chunk_ids: Sequence[object],
    k: int,
    max_tokens: int = PARENT_MAX_TOKENS,
) -> list[Document]:
    """Primii `k` părinți distincți pentru chunk-urile date în ordinea rangului.

    Chunk-urile inexistente (șterse după indexare) se sar. `metadata["chunk_id"]` e chunk-ul
    cu rangul cel mai bun din părinte, ca la rezultatele fără extindere.
    """
    wanted = [u for u in (as_uuid(c) for c in chunk_ids) if u is not None]
    parents: dict[tuple, Document] = {}
    with Session(bind) as s:
        home = dict(
            s.exec(
                select(Chunk.id, Chunk.document_id).where(col(Chunk.id).in_(wanted))
            ).all()
        )
        loaded: dict[uuid.UUID, tuple[models.Document, list[Chunk]]] = {}
        for cid in wanted:
            doc_id = home.get(cid)
            # după `k` părinți, un chunk mai contează doar dacă e dintr-un document deja încărcat
            if doc_id is None or (len(parents) == k and doc_id not in loaded):
                continue
            if doc_id not in loaded:
                doc = s.get(models.Document, doc_id)
                assert doc is not None  # FK: chunk-ul are document
                chunks = s.exec(
                    select(Chunk)
                    .where(Chunk.document_id == doc_id)
                    .order_by(col(Chunk.position))
                ).all()
                loaded[doc_id] = (doc, list(chunks))
            doc, chunks = loaded[doc_id]
            i = next(n for n, c in enumerate(chunks) if c.id == cid)
            scope, path, members = parent_of(chunks, i, doc.title, max_tokens)
            key = (doc_id, scope, *(c.id for c in members))
            if key in parents:
                parents[key].metadata["matched_chunk_ids"].append(str(cid))
                continue
            if len(parents) == k:
                continue
            parents[key] = Document(
                id=str(cid),
                page_content=_join(members),
                metadata={
                    "chunk_id": str(cid),
                    "matched_chunk_ids": [str(cid)],
                    "chunk_ids": [str(c.id) for c in members],
                    "document_id": str(doc.id),
                    "slug": doc.slug,
                    "title": doc.title,
                    "heading_path": path,
                    "scope": scope,
                },
            )
    return list(parents.values())
