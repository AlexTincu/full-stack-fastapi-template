import uuid

from sqlmodel import Session

from app.knowledge_prep.chunking import count_tokens
from app.knowledge_prep.models import Chunk
from app.knowledge_prep.parents import expand_to_parents, parent_of
from app.knowledge_prep.tests.retrieval_fakes import add_document

WORDS = "grâul încolțit dă putere " * 10  # câteva zeci de tokeni


def make_chunks(paths: list[list[str]]) -> list[Chunk]:
    doc_id = uuid.uuid4()
    return [
        Chunk(document_id=doc_id, position=i, heading_path=p, text=f"{i} {WORDS}")
        for i, p in enumerate(paths)
    ]


PATHS = [["A", "x"], ["A", "y"], ["A", "z"], ["B", "w"]]
ONE = count_tokens(f"0 {WORDS}")  # tokenii unui chunk (toate au aceeași mărime)


def texts(chunks: list[Chunk]) -> list[int]:
    return [c.position for c in chunks]


def test_small_document_is_the_parent():
    chunks = make_chunks(PATHS)
    scope, path, members = parent_of(chunks, 1, "Titlu", max_tokens=10 * ONE)
    assert (scope, path, texts(members)) == ("document", ["Titlu"], [0, 1, 2, 3])


def test_widest_section_that_fits():
    chunks = make_chunks(PATHS)
    # documentul (4 chunk-uri) nu încape, secțiunea A (3 chunk-uri) da
    scope, path, members = parent_of(chunks, 1, "Titlu", max_tokens=3 * ONE + 10)
    assert (scope, path, texts(members)) == ("section", ["A"], [0, 1, 2])


def test_narrower_section_when_the_top_one_is_too_big():
    chunks = make_chunks([["A", "x"], ["A", "x"], ["A", "y"], ["B"]])
    scope, path, members = parent_of(chunks, 0, "Titlu", max_tokens=2 * ONE + 10)
    assert (scope, path, texts(members)) == ("section", ["A", "x"], [0, 1])


def test_neighbors_of_the_same_section_when_nothing_fits():
    # o secțiune mare tăiată în 4 bucăți, plus o secțiune vecină
    chunks = make_chunks([["A"], ["A"], ["A"], ["A"], ["B"]])
    scope, path, members = parent_of(chunks, 2, "Titlu", max_tokens=ONE + 5)
    assert (scope, path, texts(members)) == ("chunk", ["A"], [1, 2, 3])
    # la margine: vecinul din altă secțiune nu intră
    scope, _, members = parent_of(chunks, 3, "Titlu", max_tokens=ONE + 5)
    assert (scope, texts(members)) == ("chunk", [2, 3])


def test_expand_dedupes_parents_and_keeps_rank_order(session: Session):
    small = add_document(session, "mic", [(["M", "1"], "unu"), (["M", "2"], "doi")])
    other = add_document(session, "altul", [(["O"], "trei")])

    got = expand_to_parents(
        session.get_bind(), [small[1].id, other[0].id, str(small[0].id)], k=5
    )

    assert [d.metadata["slug"] for d in got] == ["mic", "altul"]
    first = got[0]
    assert first.page_content == "unu\n\ndoi"
    assert first.metadata["chunk_id"] == str(small[1].id)
    assert first.metadata["matched_chunk_ids"] == [str(small[1].id), str(small[0].id)]
    assert first.metadata["chunk_ids"] == [str(c.id) for c in small]
    assert first.metadata["scope"] == "document"
    assert first.metadata["heading_path"] == ["Titlu mic"]


def test_expand_stops_at_k_distinct_parents_and_skips_missing(session: Session):
    a = add_document(session, "a", [(["A"], "a")])
    b = add_document(session, "b", [(["B"], "b")])
    c = add_document(session, "c", [(["C"], "c")])

    got = expand_to_parents(
        session.get_bind(),
        [str(uuid.uuid4()), "nu-e-uuid", b[0].id, a[0].id, c[0].id],
        k=2,
    )

    assert [d.metadata["slug"] for d in got] == ["b", "a"]
    assert expand_to_parents(session.get_bind(), [], k=3) == []
