"""Indexarea unui document revizuit: chunk-uri, întrebări, vectori (spec 2026-09-25, „Indexarea”).

Ordinea e aleasă ca un eșec să lase cât mai puțin de curățat:
1. tot ce cere API-uri (întrebări Gemini, embeddings OpenAI) se face în memorie, înainte de scriere;
2. `chunk` și `question` se înlocuiesc într-o singură tranzacție;
3. vectorii documentului se șterg (după `document_id`) și se adaugă cei noi;
4. abia apoi `stage=indexed`, `indexed_hash=reviewed_hash`.
Un eșec la pasul 3 lasă chunk-urile noi fără vectori, stadiul vechi și `last_error`; o nouă
rulare `index` reface totul. Regulile încălcate ridică `RegistryError` fără să atingă registrul.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlmodel import Session, col, delete, select

from app.knowledge_prep.chunking import chunk_markdown, embedding_text
from app.knowledge_prep.frontmatter import strip_frontmatter
from app.knowledge_prep.models import Chunk, Document, Question, Stage
from app.knowledge_prep.questions import QuestionGenerator
from app.knowledge_prep.registry import RegistryError, body_hash, record_step
from app.knowledge_prep.vectors import VectorEntry, VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    chunks: int = 0
    questions: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        logger.warning("atenție: %s", message)
        self.warnings.append(message)


def check_indexable(doc: Document, *, force: bool) -> None:
    if doc.stage.rank < Stage.reviewed.rank:
        raise RegistryError(
            f"{doc.slug} e '{doc.stage}'; index cere 'reviewed' (rulează întâi review)."
        )
    md = Path(doc.markdown_path)
    if not md.is_file():
        raise RegistryError(f"lipsește markdown-ul: {md}")
    if body_hash(md) != doc.reviewed_hash:
        raise RegistryError("markdown modificat după review; rulează din nou `review`")
    if doc.stage == Stage.validated and not force:
        raise RegistryError(
            f"{doc.slug} e 'validated': reindexarea înlocuiește toate întrebările, "
            "inclusiv cele editate manual. Folosește --force."
        )


def _chunk_ids(doc: Document):
    return select(Chunk.id).where(Chunk.document_id == doc.id)


def index_document(
    session: Session,
    doc: Document,
    *,
    vectors: VectorIndex,
    questioner: QuestionGenerator | None,
    force: bool = False,
) -> IndexReport:
    """`reviewed`+ cu markdown nemodificat → chunk-uri, întrebări, vectori → `indexed`.

    `questioner=None` (`--no-questions`) indexează doar vectorii `content`: baseline-ul.
    """
    check_indexable(doc, force=force)
    report = IndexReport()
    replaced = len(
        session.exec(
            select(Question.id).where(col(Question.chunk_id).in_(_chunk_ids(doc)))
        ).all()
    )
    if replaced:
        report.warn(
            f"se înlocuiesc {replaced} întrebări existente (editările manuale se pierd)"
        )

    with record_step(session, doc):
        body = strip_frontmatter(Path(doc.markdown_path).read_text(encoding="utf-8"))
        drafts, warnings = chunk_markdown(body, doc.title, doc.split)
        for w in warnings:
            report.warn(w)
        chunks = [
            Chunk(
                document_id=doc.id, position=i, heading_path=d.heading_path, text=d.text
            )
            for i, d in enumerate(drafts)
        ]

        questions: list[Question] = []
        if questioner is not None:
            for i, chunk in enumerate(chunks, 1):
                logger.info(
                    "întrebări pentru „%s”: chunk %d/%d ...", doc.title, i, len(chunks)
                )
                texts = questioner(
                    doc.title, chunk.heading_path, chunk.text, not doc.split
                )
                if not texts:
                    report.warn(f"chunk {i}: nicio întrebare generată")
                questions += [Question(chunk_id=chunk.id, text=t) for t in texts]

        entries = [
            VectorEntry(
                c.id, c.id, doc.id, "content", embedding_text(c.heading_path, c.text)
            )
            for c in chunks
        ] + [
            VectorEntry(q.id, q.chunk_id, doc.id, "question", q.text) for q in questions
        ]
        logger.info("embeddings pentru %d vectori ...", len(entries))
        embeddings = vectors.embed([e.text for e in entries])

        # relațional, o singură tranzacție; delete explicit pe question (SQLite nu face cascade)
        session.exec(
            delete(Question).where(col(Question.chunk_id).in_(_chunk_ids(doc)))
        )
        session.exec(delete(Chunk).where(col(Chunk.document_id) == doc.id))
        session.add_all(chunks)
        session.flush()
        session.add_all(questions)
        session.commit()

        vectors.replace_document(doc.id, entries, embeddings)
        doc.stage = Stage.indexed
        doc.indexed_hash = doc.reviewed_hash

    report.chunks, report.questions = len(chunks), len(questions)
    return report
