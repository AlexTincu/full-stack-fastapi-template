from sqlmodel import Session

from app.knowledge_prep.models import Chunk, Document, DocumentType, Question, Stage


def test_stage_rank_follows_pipeline_order():
    assert [s.value for s in Stage] == [
        "uploaded",
        "structured",
        "reviewed",
        "indexed",
        "validated",
    ]
    assert Stage.uploaded.rank < Stage.reviewed.rank < Stage.validated.rank


def test_document_chunk_question_roundtrip(session: Session):
    doc = Document(
        slug="graul-incoltit",
        title="Grâul încolțit",
        type=DocumentType.carte,
        split=True,
        stage=Stage.structured,
        markdown_path="/k/converted/graul-incoltit.md",
    )
    session.add(doc)
    session.commit()
    chunk = Chunk(
        document_id=doc.id,
        position=0,
        heading_path=["Grâul încolțit", "Contraindicații"],
        text="t",
    )
    session.add(chunk)
    session.commit()
    session.add(Question(chunk_id=chunk.id, text="Cine nu are voie grâu încolțit?"))
    session.commit()
    session.expire_all()

    loaded = session.get(Document, doc.id)
    assert loaded is not None
    assert (loaded.stage, loaded.type) == (Stage.structured, DocumentType.carte)
    assert loaded.last_error is None and loaded.created_at is not None
    loaded_chunk = session.get(Chunk, chunk.id)
    assert loaded_chunk is not None
    assert loaded_chunk.heading_path == ["Grâul încolțit", "Contraindicații"]
