import pytest

from app.knowledge_prep import chunking
from app.knowledge_prep.chunking import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    ChunkDraft,
    chunk_markdown,
    count_tokens,
    embedding_text,
)

MID = " ".join(["Grâul încolțit dă putere organismului slăbit."] * 10)
SHORT = "Doar o frază."


def paths(chunks: list[ChunkDraft]) -> list[list[str]]:
    return [c.heading_path for c in chunks]


def test_fixture_sizes():
    # ancorează presupunerile de mai jos: MID e o secțiune „normală”, SHORT una mică
    assert MIN_CHUNK_TOKENS < count_tokens(MID) < MAX_CHUNK_TOKENS // 3
    assert count_tokens(SHORT) < MIN_CHUNK_TOKENS


def test_sections_follow_headers_and_keep_text_verbatim():
    md = (
        f"# Carte\n\n{MID}\n\n## Capitol A\n\n{MID}\n\n{MID}\n\n"
        f"### Sub A1\n\n{MID}\n\n## Capitol B\n\n{MID}\n"
    )
    chunks, warnings = chunk_markdown(md, "Titlu doc", split=True)
    assert paths(chunks) == [
        ["Carte"],
        ["Carte", "Capitol A"],
        ["Carte", "Capitol A", "Sub A1"],
        ["Carte", "Capitol B"],
    ]
    assert chunks[1].text == f"## Capitol A\n\n{MID}\n\n{MID}"  # rândurile goale rămân
    assert warnings == []


def test_small_sections_merge_forward_within_same_parent():
    md = f"# Carte\n\n## A\n\n{SHORT}\n\n## B\n\n{MID}\n"
    chunks, _ = chunk_markdown(md, "Titlu doc", split=True)
    assert paths(chunks) == [["Carte"]]
    assert chunks[0].text == f"# Carte\n\n## A\n\n{SHORT}\n\n## B\n\n{MID}"


def test_header_only_section_takes_path_of_next_and_small_does_not_cross_parent():
    md = f"# Carte\n\n## A\n\n{MID}\n\n### A1\n\n{SHORT}\n\n## B\n\n{MID}\n"
    chunks, _ = chunk_markdown(md, "Titlu doc", split=True)
    assert paths(chunks) == [["Carte", "A"], ["Carte", "A", "A1"], ["Carte", "B"]]
    assert chunks[0].text.startswith("# Carte\n\n## A\n\n")
    assert chunks[1].text == f"### A1\n\n{SHORT}"  # B nu e în același părinte


def test_large_section_is_split_keeping_heading_path():
    big = "\n\n".join(f"Paragraful {i}. {MID}" for i in range(12))
    chunks, _ = chunk_markdown(
        f"# Carte\n\n## Mare\n\n{big}\n", "Titlu doc", split=True
    )
    assert len(chunks) >= 2
    assert all(c.heading_path == ["Carte", "Mare"] for c in chunks)
    assert all(count_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    assert chunks[0].text.startswith("# Carte\n\n## Mare\n\nParagraful 0.")
    for i in range(12):
        assert any(f"Paragraful {i}." in c.text for c in chunks)


def test_verify_markers_are_removed():
    md = (
        "# Carte\n\n<!-- VERIFICĂ: fereastra 1, lipsă 2.0%, adăugat 0.0% -->\n\n"
        f"{MID} <!-- VERIFICĂ: text ilizibil -->\n"
    )
    [chunk], _ = chunk_markdown(md, "Titlu doc", split=True)
    assert "VERIFICĂ" not in chunk.text
    assert "\n\n\n" not in chunk.text


def test_strip_verify_markers_preserves_unrelated_multiple_blank_lines():
    # 3+ newlines with no marker nearby (ex. într-un bloc de cod sau un split=false cu
    # spațiere intenționată) trebuie să rămână neatinse, byte-identic.
    body = "Paragraf unu.\n\n\n\nParagraf doi."
    assert chunking._strip_verify_markers(body) == body


def test_strip_verify_markers_removes_marker_without_leaving_extra_blank_lines():
    body = "Paragraf unu.\n\n<!-- VERIFICĂ: motiv -->\n\nParagraf doi."
    assert chunking._strip_verify_markers(body) == "Paragraf unu.\n\nParagraf doi."


def test_code_fence_and_hashtag_are_not_headers():
    md = f"# Carte\n\n{MID}\n\n```\n# nu e titlu\n```\n\n#nici_asta nu e\n"
    [chunk], _ = chunk_markdown(md, "Titlu doc", split=True)
    assert chunk.heading_path == ["Carte"]
    assert "# nu e titlu" in chunk.text and "#nici_asta" in chunk.text


def test_text_before_first_header_gets_document_title():
    chunks, _ = chunk_markdown(f"{MID}\n\n## A\n\n{MID}\n", "Titlu doc", split=True)
    assert paths(chunks) == [["Titlu doc"], ["A"]]


def test_unsplit_document_is_one_chunk_named_after_title(monkeypatch):
    md = f"# Regim\n\n## Dimineața\n\n{SHORT}\n\n## Seara\n\n{MID}\n"
    chunks, warnings = chunk_markdown(md, "Regim DRI - Anemie", split=False)
    assert chunks == [ChunkDraft(["Regim DRI - Anemie"], md.strip())]
    assert warnings == []

    monkeypatch.setattr(chunking, "UNSPLIT_WARN_TOKENS", 10)
    _, warnings = chunk_markdown(md, "Regim DRI - Anemie", split=False)
    assert len(warnings) == 1 and "tokeni" in warnings[0]


def test_empty_body_raises():
    with pytest.raises(ValueError, match="markdown gol"):
        chunk_markdown("<!-- VERIFICĂ: fereastra 1 -->\n\n", "Titlu doc", split=True)


def test_embedding_text_prefixes_heading_path():
    assert embedding_text(["Carte", "A"], "text") == "Carte > A\n\ntext"
