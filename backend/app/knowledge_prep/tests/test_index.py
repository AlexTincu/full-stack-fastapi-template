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


def test_reindex_after_edit_redoes_only_changed_chunks(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    other = make_doc(session, tmp_path, title="Regim DRI - Anemie", doc_type="regim")
    vectors = FakeVectors()
    index_document(session, other, vectors=vectors, questioner=FakeQuestions())
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    before = chunks_of(session, doc)
    old_ids = {e.id for e in vectors.of(doc.id)}
    other_ids = {e.id for e in vectors.of(other.id)}
    # o întrebare editată manual pe un chunk neschimbat rămâne
    q = questions_of(session, doc)[0]
    q.text = "Întrebare editată?"
    session.add(q)
    session.commit()

    md = Path(doc.markdown_path)
    md.write_text(
        md.read_text(encoding="utf-8").replace(
            "## Contraindicații", "## Contraindicații la copii"
        )
        + f"\n## Mod de folosire\n\n{MID}\n",
        encoding="utf-8",
    )
    registry.review(session, doc)
    questioner = FakeQuestions()
    report = index_document(session, doc, vectors=vectors, questioner=questioner)

    after = chunks_of(session, doc)
    assert after[0].id == before[0].id  # Beneficii: neschimbat, păstrat
    assert after[1].id != before[1].id  # Contraindicații: titlu schimbat, refăcut
    assert [c.heading_path[-1] for c in after] == [
        "Beneficii",
        "Contraindicații la copii",
        "Mod de folosire",
    ]
    assert [c[1][-1] for c in questioner.calls] == [
        "Contraindicații la copii",
        "Mod de folosire",
    ]
    assert "Întrebare editată?" in {q.text for q in questions_of(session, doc)}
    assert len(questions_of(session, doc)) == 6
    assert len(session.exec(select(Question)).all()) == 6 + 2
    assert len(session.exec(select(Chunk)).all()) == 3 + 1
    new_ids = {e.id for e in vectors.of(doc.id)}
    assert len(new_ids) == 3 + 6 and len(old_ids & new_ids) == 1 + 2
    assert {e.id for e in vectors.of(other.id)} == other_ids
    assert (report.chunks, report.reused, report.questions) == (3, 1, 6)
    assert any("se înlocuiesc 2 întrebări" in w for w in report.warnings)
    assert all(c.indexed_at is not None for c in after)
    assert doc.stage == Stage.indexed


def test_rerun_on_complete_index_only_marks_indexed(session: Session, tmp_path: Path):
    """Cazul unei rulări întrerupte după scrierea ultimului lot: nimic de refăcut."""
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    ids = {c.id for c in chunks_of(session, doc)}
    doc.stage, doc.indexed_hash = Stage.reviewed, None
    session.add(doc)
    session.commit()

    questioner = FakeQuestions()
    vectors.embedded.clear()
    report = index_document(session, doc, vectors=vectors, questioner=questioner)

    assert questioner.calls == [] and vectors.embedded == []
    assert {c.id for c in chunks_of(session, doc)} == ids
    assert (report.chunks, report.reused, report.questions) == (2, 2, 4)
    assert (doc.stage, doc.indexed_hash) == (Stage.indexed, doc.reviewed_hash)


def test_force_redoes_every_chunk(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    old = {c.id for c in chunks_of(session, doc)}

    questioner = FakeQuestions()
    report = index_document(
        session, doc, vectors=vectors, questioner=questioner, force=True
    )
    assert len(questioner.calls) == 2 and report.reused == 0
    assert not old & {c.id for c in chunks_of(session, doc)}
    assert len(vectors.of(doc.id)) == 2 + 4


def test_switching_questions_mode_redoes_chunks(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())

    report = index_document(session, doc, vectors=vectors, questioner=None)
    assert report.reused == 0 and questions_of(session, doc) == []
    assert {e.kind for e in vectors.of(doc.id)} == {"content"}

    questioner = FakeQuestions()
    report = index_document(session, doc, vectors=vectors, questioner=questioner)
    assert report.reused == 0 and len(questioner.calls) == 2
    assert len(vectors.of(doc.id, "question")) == 4


def test_failure_in_second_batch_keeps_first_and_rerun_resumes(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors(fail="write", writes_ok=1)
    with pytest.raises(RuntimeError):
        index_document(
            session, doc, vectors=vectors, questioner=FakeQuestions(), batch_size=1
        )
    assert (doc.stage, doc.last_error) == (
        Stage.reviewed,
        "RuntimeError: pgvector căzut",
    )
    [done, pending] = chunks_of(session, doc)
    assert done.indexed_at is not None and pending.indexed_at is None
    assert {e.doc_id for e in vectors.of(doc.id)} == {done.id}

    vectors.fail = None
    questioner = FakeQuestions()
    report = index_document(
        session, doc, vectors=vectors, questioner=questioner, batch_size=1
    )
    assert [c[1][-1] for c in questioner.calls] == ["Contraindicații"]
    after = chunks_of(session, doc)
    assert after[0].id == done.id and after[1].id != pending.id
    assert (report.chunks, report.reused, report.questions) == (2, 1, 4)
    assert len(vectors.of(doc.id)) == 2 + 4
    assert (doc.stage, doc.last_error) == (Stage.indexed, None)


def test_interrupt_is_recorded_and_keeps_finished_batches(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    calls = FakeQuestions()

    def interrupted(title: str, path: list[str], text: str, whole: bool) -> list[str]:
        if calls.calls:
            raise KeyboardInterrupt
        return calls(title, path, text, whole)

    with pytest.raises(KeyboardInterrupt):
        index_document(
            session, doc, vectors=FakeVectors(), questioner=interrupted, batch_size=1
        )
    assert (doc.stage, doc.last_error) == (
        Stage.reviewed,
        "KeyboardInterrupt: întrerupt",
    )
    assert [c.indexed_at is not None for c in chunks_of(session, doc)] == [True]


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
