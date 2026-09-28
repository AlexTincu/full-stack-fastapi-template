from pathlib import Path

import pytest
from sqlmodel import Session, col, select

from app.knowledge_prep import registry
from app.knowledge_prep.frontmatter import FrontMatter, render, slugify
from app.knowledge_prep.index import index_document
from app.knowledge_prep.models import Chunk, Document, Question, Stage
from app.knowledge_prep.registry import RegistryError
from app.knowledge_prep.tests.index_fakes import FakeQuestions, FakeVectors

MID = " ".join(["Grâul încolțit dă putere organismului slăbit."] * 10)
BOOK = f"# Grâul încolțit\n\n## Beneficii\n\n{MID}\n\n## Contraindicații\n\n{MID}\n"


def make_doc(
    session: Session,
    tmp_path: Path,
    body: str = BOOK,
    *,
    title: str = "Grâul încolțit",
    doc_type: str = "carte",
    review: bool = True,
) -> Document:
    md = tmp_path / f"{slugify(title)}.md"
    fm = FrontMatter(title=title, slug=slugify(title), type=doc_type, author=None)
    md.write_text(render(fm, body), encoding="utf-8")
    doc = registry.register_structured(session, md)
    return registry.review(session, doc) if review else doc


def chunks_of(session: Session, doc: Document) -> list[Chunk]:
    return list(
        session.exec(
            select(Chunk)
            .where(Chunk.document_id == doc.id)
            .order_by(col(Chunk.position))
        )
    )


def questions_of(session: Session, doc: Document) -> list[Question]:
    return list(
        session.exec(
            select(Question)
            .join(Chunk, col(Question.chunk_id) == col(Chunk.id))
            .where(Chunk.document_id == doc.id)
        )
    )


def edit_and_review(session: Session, doc: Document, extra: str) -> None:
    md = Path(doc.markdown_path)
    md.write_text(md.read_text(encoding="utf-8") + extra, encoding="utf-8")
    registry.review(session, doc)


def test_index_writes_chunks_questions_vectors_and_marks_indexed(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors, questioner = FakeVectors(), FakeQuestions()

    report = index_document(session, doc, vectors=vectors, questioner=questioner)

    chunks = chunks_of(session, doc)
    assert [c.heading_path for c in chunks] == [
        ["Grâul încolțit", "Beneficii"],
        ["Grâul încolțit", "Contraindicații"],
    ]
    assert [c.position for c in chunks] == [0, 1]
    assert chunks[0].text.startswith(
        "# Grâul încolțit\n\n## Beneficii"
    )  # fără front-matter
    questions = questions_of(session, doc)
    assert (report.chunks, report.questions, len(questions)) == (2, 4, 4)
    assert questioner.calls[0][:2] == (
        "Grâul încolțit",
        ["Grâul încolțit", "Beneficii"],
    )
    assert questioner.calls[0][3] is False

    content = {e.id: e for e in vectors.of(doc.id, "content")}
    assert set(content) == {c.id for c in chunks}
    assert content[chunks[0].id].doc_id == chunks[0].id
    assert (
        content[chunks[0].id].text == f"Grâul încolțit > Beneficii\n\n{chunks[0].text}"
    )
    by_id = {e.id: e for e in vectors.of(doc.id, "question")}
    assert set(by_id) == {q.id for q in questions}
    assert all(
        by_id[q.id].doc_id == q.chunk_id and by_id[q.id].text == q.text
        for q in questions
    )

    assert (doc.stage, doc.indexed_hash, doc.last_error) == (
        Stage.indexed,
        doc.reviewed_hash,
        None,
    )
    assert registry.hash_warnings(doc) == []


def test_no_questions_indexes_only_content(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    report = index_document(session, doc, vectors=vectors, questioner=None)
    assert report.questions == 0 and questions_of(session, doc) == []
    assert {e.kind for e in vectors.rows.values()} == {"content"}
    assert doc.stage == Stage.indexed


def test_unsplit_document_is_one_chunk_with_whole_document_questions(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path, title="Regim DRI - Anemie", doc_type="regim")
    assert doc.split is False
    questioner = FakeQuestions()
    index_document(session, doc, vectors=FakeVectors(), questioner=questioner)
    [chunk] = chunks_of(session, doc)
    assert chunk.heading_path == ["Regim DRI - Anemie"]
    assert chunk.text == BOOK.strip()
    assert questioner.calls[0][3] is True


def test_reindex_replaces_old_rows_and_vectors_only_for_that_document(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    other = make_doc(session, tmp_path, title="Regim DRI - Anemie", doc_type="regim")
    vectors = FakeVectors()
    index_document(session, other, vectors=vectors, questioner=FakeQuestions())
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    old_ids = {e.id for e in vectors.of(doc.id)}
    other_ids = {e.id for e in vectors.of(other.id)}

    edit_and_review(session, doc, f"\n## Mod de folosire\n\n{MID}\n")
    report = index_document(session, doc, vectors=vectors, questioner=FakeQuestions())

    assert len(chunks_of(session, doc)) == 3
    assert len(questions_of(session, doc)) == 6
    assert len(session.exec(select(Question)).all()) == 6 + 2
    assert len(session.exec(select(Chunk)).all()) == 3 + 1
    new_ids = {e.id for e in vectors.of(doc.id)}
    assert not (old_ids & new_ids) and len(new_ids) == 3 + 6
    assert {e.id for e in vectors.of(other.id)} == other_ids
    assert any("se înlocuiesc 4 întrebări" in w for w in report.warnings)


def test_index_requires_reviewed_and_unchanged_markdown(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path, review=False)
    with pytest.raises(RegistryError, match="index cere 'reviewed'"):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())

    registry.review(session, doc)
    md = Path(doc.markdown_path)
    md.write_text(md.read_text(encoding="utf-8") + "\nadăugat\n", encoding="utf-8")
    with pytest.raises(
        RegistryError, match="markdown modificat după review; rulează din nou `review`"
    ):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())
    assert chunks_of(session, doc) == []
    assert (doc.stage, doc.last_error) == (Stage.reviewed, None)


def test_validated_document_needs_force(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    registry.validate(session, doc)

    with pytest.raises(RegistryError, match="--force"):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert doc.stage == Stage.validated

    index_document(
        session, doc, vectors=vectors, questioner=FakeQuestions(), force=True
    )
    assert doc.stage == Stage.indexed


def test_embedding_failure_keeps_previous_index_and_records_error(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    old_chunks = {c.id for c in chunks_of(session, doc)}
    edit_and_review(session, doc, f"\n## Nou\n\n{MID}\n")

    vectors.fail = "embed"
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())

    assert (doc.stage, doc.last_error) == (Stage.reviewed, "RuntimeError: OpenAI căzut")
    assert {c.id for c in chunks_of(session, doc)} == old_chunks  # nimic scris
    assert {e.doc_id for e in vectors.of(doc.id, "content")} == old_chunks


def test_vector_write_failure_keeps_stage_and_rerun_recovers(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors(fail="write")
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert (doc.stage, doc.last_error) == (
        Stage.reviewed,
        "RuntimeError: pgvector căzut",
    )
    assert len(chunks_of(session, doc)) == 2  # commit-ul relațional s-a făcut

    vectors.fail = None
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert (doc.stage, doc.last_error) == (Stage.indexed, None)
    assert len(chunks_of(session, doc)) == 2
    assert len(vectors.of(doc.id)) == 2 + 4


def test_question_failure_writes_nothing(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)

    def broken(_t: str, _h: list[str], _x: str, _w: bool) -> list[str]:
        raise RuntimeError("Gemini căzut")

    vectors = FakeVectors()
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=broken)
    assert (doc.stage, doc.last_error) == (Stage.reviewed, "RuntimeError: Gemini căzut")
    assert chunks_of(session, doc) == [] and vectors.rows == {}


def test_empty_markdown_records_error(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path, body="<!-- VERIFICĂ: fereastra 1 -->\n")
    with pytest.raises(ValueError):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())
    assert doc.stage == Stage.reviewed
    assert doc.last_error is not None and doc.last_error.startswith(
        "ValueError: markdown gol"
    )


def test_chunk_without_questions_is_reported(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    report = index_document(
        session, doc, vectors=FakeVectors(), questioner=FakeQuestions(per_chunk=0)
    )
    assert doc.stage == Stage.indexed
    assert "chunk 1: nicio întrebare generată" in report.warnings
