from langchain_core.documents import Document
from sqlmodel import Session

from app.knowledge_prep.explorer import (
    ONLY_HERE,
    SAME_CHUNK,
    SAME_DOCUMENT,
    Hit,
    PanelResult,
    coverage_rows,
    document_files,
    knowledge_coverage,
    knowledge_hits,
    overlap_marks,
    rag_simple_hits,
    render,
)
from app.knowledge_prep.tests.retrieval_fakes import add_document


def test_rag_simple_hits_use_source_and_headings():
    doc = Document(
        page_content="text",
        metadata={"source": "carte.md", "h3": "Capitol", "h4": "Sub"},
    )
    assert rag_simple_hits([doc]) == [Hit("carte.md", "Capitol > Sub", "text")]


def test_knowledge_hits_for_chunks_and_parents():
    chunk = Document(
        page_content="c",
        metadata={
            "chunk_id": "c1",
            "document_id": "d1",
            "slug": "carte",
            "heading_path": ["A", "B"],
        },
    )
    parent = Document(
        page_content="p",
        metadata={
            "chunk_id": "c2",
            "document_id": "d2",
            "slug": "alta",
            "heading_path": ["X"],
            "scope": "secțiune",
            "matched_chunk_ids": ["c2", "c3"],
            "chunk_ids": ["c2", "c3", "c4"],
        },
    )
    a, b = knowledge_hits([chunk, parent], {"d1": "carte.md"})
    assert (a.source, a.location, a.chunk_ids) == ("carte.md", "A > B", {"c1"})
    # documentul lipsește din registru → slug-ul
    assert (b.source, b.chunk_ids) == ("alta", {"c2", "c3"})
    assert b.detail == "secțiune, 2/3 chunk-uri găsite"


def test_overlap_marks_prefer_chunk_over_document():
    hits = [
        Hit("a.md", "", "", frozenset({"1"})),
        Hit("a.md", "", "", frozenset({"9"})),
        Hit("b.md", "", ""),
    ]
    other = [Hit("a.md", "", "", frozenset({"1", "2"}))]
    assert overlap_marks(hits, other) == [SAME_CHUNK, SAME_DOCUMENT, ONLY_HERE]
    assert overlap_marks(hits, None) == ["", "", ""]


def test_render_shows_error_and_ignores_failed_other_panel():
    failed = PanelResult("rag_simple", error="boom")
    assert "EROARE" in render(failed, None)
    ok = PanelResult("baseline", hits=[Hit("a.md", "Cap", "text")])
    out = render(ok, failed)
    assert "**1. a.md**" in out
    assert ONLY_HERE not in out  # fără comparație cu un panou eșuat


def test_coverage_rows_put_common_documents_first():
    rows = coverage_rows(
        {"doar-vechi.md": 5, "comun.md": 3},
        {"comun.md": ("indexed", 2), "doar-nou.md": ("reviewed", 0)},
    )
    assert rows[0] == ["comun.md", 3, "indexed", 2]
    assert sorted(r[0] for r in rows[1:]) == ["doar-nou.md", "doar-vechi.md"]
    assert ["doar-vechi.md", 5, "—", 0] in rows


def test_registry_queries(session: Session):
    chunks = add_document(session, "carte", [(["A"], "x"), (["B"], "y")])
    bind = session.get_bind()
    assert document_files(bind) == {str(chunks[0].document_id): "carte.md"}
    assert knowledge_coverage(bind) == {"carte.md": ("indexed", 2)}
