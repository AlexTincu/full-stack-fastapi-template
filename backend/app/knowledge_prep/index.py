"""Indexarea unui document revizuit: chunk-uri, întrebări, vectori (spec 2026-09-25, „Indexarea”).

Indexarea e incrementală și reluabilă:
1. chunking-ul se compară cu chunk-urile din DB după `content_hash`. Un chunk identic, deja
   indexat (`indexed_at` pus) și făcut în același mod (cu/fără întrebări) se păstrează, cu
   întrebările lui, inclusiv cele editate manual. `force` reface tot;
2. restul chunk-urilor vechi (dispărute, schimbate sau neterminate) se șterg, cu vectorii lor;
3. chunk-urile noi se procesează în loturi de `BATCH_SIZE`: întrebări Gemini, embeddings
   OpenAI, `chunk` + `question` într-o tranzacție, vectorii, apoi `indexed_at`;
4. abia când toate chunk-urile au `indexed_at`: `stage=indexed`, `indexed_hash=reviewed_hash`.
Un eșec (sau o oprire manuală) păstrează loturile terminate, stadiul vechi și `last_error`; o
nouă rulare `index` reia de unde a rămas. Regulile încălcate ridică `RegistryError` fără să
atingă registrul.
"""

import logging
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path

from sqlmodel import Session, col, delete, func, select

from app.knowledge_prep.chunking import chunk_hash, chunk_markdown, embedding_text
from app.knowledge_prep.frontmatter import strip_frontmatter
from app.knowledge_prep.models import Chunk, Document, Question, Stage
from app.knowledge_prep.questions import QuestionGenerator
from app.knowledge_prep.registry import RegistryError, body_hash, record_step
from app.knowledge_prep.vectors import VectorEntry, VectorIndex
from app.models import get_datetime_utc

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # chunk-uri per lot: ce se pierde la o întrerupere


@dataclass
class IndexReport:
    chunks: int = 0
    reused: int = 0  # chunk-uri păstrate dintr-o rulare anterioară
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


def _question_counts(session: Session, chunk_ids: list) -> dict:
    rows = session.exec(
        select(Question.chunk_id, func.count())
        .where(col(Question.chunk_id).in_(chunk_ids))
        .group_by(col(Question.chunk_id))
    ).all()
    return dict(rows)


def index_document(
    session: Session,
    doc: Document,
    *,
    vectors: VectorIndex,
    questioner: QuestionGenerator | None,
    force: bool = False,
    batch_size: int = BATCH_SIZE,
) -> IndexReport:
    """`reviewed`+ cu markdown nemodificat → chunk-uri, întrebări, vectori → `indexed`.

    `questioner=None` (`--no-questions`) indexează doar vectorii `content`: baseline-ul.
    """
    check_indexable(doc, force=force)
    report = IndexReport()

    with record_step(session, doc):
        body = strip_frontmatter(Path(doc.markdown_path).read_text(encoding="utf-8"))
        drafts, warnings = chunk_markdown(body, doc.title, doc.split)
        for w in warnings:
            report.warn(w)

        existing = list(
            session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
        )
        n_questions = _question_counts(session, [c.id for c in existing])
        reusable: dict[str, list[Chunk]] = {}
        if not force:
            for c in existing:
                # același mod: un chunk fără întrebări nu servește unei rulări cu întrebări
                if (
                    c.indexed_at is not None
                    and c.content_hash is not None
                    and ((n_questions.get(c.id, 0) > 0) == (questioner is not None))
                ):
                    reusable.setdefault(c.content_hash, []).append(c)

        kept: list[Chunk] = []
        pending: list[Chunk] = []
        for i, d in enumerate(drafts):
            h = chunk_hash(d.heading_path, d.text)
            if reusable.get(h):
                c = reusable[h].pop(0)
                c.position = i
                kept.append(c)
            else:
                pending.append(
                    Chunk(
                        document_id=doc.id,
                        position=i,
                        heading_path=d.heading_path,
                        text=d.text,
                        content_hash=h,
                    )
                )

        kept_ids = {c.id for c in kept}
        stale = [c.id for c in existing if c.id not in kept_ids]
        replaced = sum(n_questions.get(cid, 0) for cid in stale)
        if replaced:
            report.warn(
                f"se înlocuiesc {replaced} întrebări existente (editările manuale se pierd)"
            )
        # vectorii întâi: un eșec aici lasă rândurile, pe care rularea următoare le găsește
        vectors.delete_chunks(stale)
        # delete explicit pe question (SQLite nu face cascade)
        session.exec(delete(Question).where(col(Question.chunk_id).in_(stale)))
        session.exec(delete(Chunk).where(col(Chunk.id).in_(stale)))
        session.add_all(kept)
        session.commit()
        logger.info(
            "„%s”: %d chunk-uri, %d păstrate, %d de indexat",
            doc.title,
            len(drafts),
            len(kept),
            len(pending),
        )

        batches = list(batched(pending, batch_size, strict=False))
        for b, batch in enumerate(batches, 1):
            questions: list[Question] = []
            if questioner is not None:
                for chunk in batch:
                    logger.info(
                        "întrebări pentru „%s”: chunk %d/%d ...",
                        doc.title,
                        chunk.position + 1,
                        len(drafts),
                    )
                    texts = questioner(
                        doc.title, chunk.heading_path, chunk.text, not doc.split
                    )
                    if not texts:
                        report.warn(
                            f"chunk {chunk.position + 1}: nicio întrebare generată"
                        )
                    questions += [Question(chunk_id=chunk.id, text=t) for t in texts]

            entries = [
                VectorEntry(
                    c.id,
                    c.id,
                    doc.id,
                    "content",
                    embedding_text(c.heading_path, c.text),
                )
                for c in batch
            ] + [
                VectorEntry(q.id, q.chunk_id, doc.id, "question", q.text)
                for q in questions
            ]
            logger.info(
                "lot %d/%d: embeddings și scriere pentru %d vectori ...",
                b,
                len(batches),
                len(entries),
            )
            embeddings = vectors.embed([e.text for e in entries])

            session.add_all(batch)
            session.flush()
            session.add_all(questions)
            session.commit()  # fără indexed_at: neterminate până la vectori

            vectors.add(entries, embeddings)
            now = get_datetime_utc()
            for c in batch:
                c.indexed_at = now
            session.add_all(batch)
            session.commit()

        doc.stage = Stage.indexed
        doc.indexed_hash = doc.reviewed_hash

    report.chunks, report.reused = len(drafts), len(kept)
    report.questions = sum(
        _question_counts(session, [c.id for c in kept + pending]).values()
    )
    return report
