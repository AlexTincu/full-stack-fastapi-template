from pathlib import Path

import pytest
from sqlmodel import Session

from app.knowledge_prep import registry
from app.knowledge_prep.frontmatter import FrontMatter, render
from app.knowledge_prep.models import Document, DocumentType, Stage
from app.knowledge_prep.registry import AlreadyRegistered, RegistryError


def fake_converter(path: Path) -> str:
    return f"conținut din {path.name}\n"


CONVERTERS = {".txt": fake_converter, ".ogg": lambda _p: "transcriere\n"}


def make_file(tmp_path: Path, name: str, content: str = "brut") -> Path:
    f = tmp_path / "baza" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


def reg(session: Session, path: Path, **kwargs) -> Document:
    return registry.register(session, path, converters=CONVERTERS, **kwargs)


def structured_doc(
    session: Session, tmp_path: Path, body: str = "# Titlu\n\ncorp\n"
) -> tuple[Document, Path]:
    fm = FrontMatter(title="Carte", slug="carte", type="carte", author=None)
    md = make_file(tmp_path, "converted/carte.md", render(fm, body))
    return registry.register_structured(session, md), md


def test_register_original_is_uploaded_with_planned_markdown_path(
    session: Session, tmp_path: Path
):
    src = make_file(tmp_path, "regimuri/Regim DRI - Anemie.txt")

    doc = reg(session, src, doc_type=DocumentType.regim)

    assert (doc.slug, doc.title, doc.stage, doc.split) == (
        "regim-dri-anemie",
        "Regim DRI - Anemie",
        Stage.uploaded,
        False,
    )
    assert doc.original_path == str(src.resolve())
    assert doc.markdown_path == str(
        src.parent.resolve() / "converted" / "regim-dri-anemie.md"
    )


def test_register_defaults_type_and_split(session: Session, tmp_path: Path):
    audio = reg(session, make_file(tmp_path, "digestia.ogg"))
    book = reg(
        session, make_file(tmp_path, "Alimente.txt"), title="Alimente vii", author="X"
    )
    note = reg(
        session, make_file(tmp_path, "n.txt"), doc_type=DocumentType.nota, split=True
    )

    assert (audio.type, audio.split) == (DocumentType.transcriere, True)
    assert (book.type, book.split, book.slug, book.author) == (
        DocumentType.carte,
        True,
        "alimente-vii",
        "X",
    )
    assert note.split is True


def test_register_rejects_unsupported_extension(session: Session, tmp_path: Path):
    with pytest.raises(RegistryError, match=".xyz"):
        reg(session, make_file(tmp_path, "a.xyz"))


def test_register_same_file_twice_is_already_registered(
    session: Session, tmp_path: Path
):
    src = make_file(tmp_path, "a.txt")
    reg(session, src)
    with pytest.raises(AlreadyRegistered):
        reg(session, src)


def test_register_rejects_slug_used_by_another_file(session: Session, tmp_path: Path):
    reg(session, make_file(tmp_path, "unu.txt"), title="Același titlu")
    with pytest.raises(RegistryError, match="slug"):
        reg(session, make_file(tmp_path, "doi.txt"), title="Același titlu")


def test_register_structured_reads_legacy_frontmatter_and_links_original(
    session: Session, tmp_path: Path
):
    original = make_file(tmp_path, "audio/Rugăciunea este intimitate.ogg")
    md = make_file(
        tmp_path,
        "audio/converted/rugaciunea-este-intimitate.md",
        "---\ntitle: Rugăciunea este intimitate\nslug: rugaciunea-este-intimitate\n"
        "type: transcriere\nauthor: null\nsource_file: Rugăciunea este intimitate.ogg\n"
        "status: reviewed\n---\n\ncorp\n",
    )

    doc = registry.register_structured(session, md, doc_type=DocumentType.nota)

    assert (doc.title, doc.type, doc.stage) == (
        "Rugăciunea este intimitate",
        DocumentType.transcriere,
        Stage.structured,
    )
    assert doc.original_path == str(original.resolve())
    assert doc.markdown_path == str(md.resolve())
    assert doc.reviewed_hash is None  # vechiul `status: reviewed` nu contează


def test_register_structured_without_frontmatter_uses_options_or_filename(
    session: Session, tmp_path: Path
):
    md = make_file(
        tmp_path,
        "whatsapp/extracted/2024-1/bucata_020.md",
        "<!-- bucata 020 -->\n\n## Nuci\n",
    )

    doc = registry.register_structured(session, md, doc_type=DocumentType.conversatie)

    assert (doc.title, doc.slug, doc.type, doc.original_path) == (
        "bucata 020",
        "bucata-020",
        DocumentType.conversatie,
        None,
    )


def test_register_structured_rejects_invalid_type_in_frontmatter(
    session: Session, tmp_path: Path
):
    md = make_file(tmp_path, "x.md", "---\ntitle: X\ntype: roman\n---\n\ncorp\n")
    with pytest.raises(RegistryError, match="roman"):
        registry.register_structured(session, md)


def test_resolve_by_id_slug_and_paths_in_any_form(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = make_file(tmp_path, "audio DRI/Rugăciunea este intimitate.txt")
    doc = reg(session, src)
    monkeypatch.chdir(tmp_path / "baza")

    for target in (
        str(doc.id),
        doc.slug,
        str(src),
        "audio DRI/Rugăciunea este intimitate.txt",
        "audio DRI/converted/rugaciunea-este-intimitate.md",
    ):
        assert registry.resolve(session, target).id == doc.id
    with pytest.raises(RegistryError, match="negăsit"):
        registry.resolve(session, "nu-exista")


def test_review_saves_hash_of_body_only(session: Session, tmp_path: Path):
    doc, md = structured_doc(session, tmp_path)

    doc = registry.review(session, doc)

    assert doc.stage == Stage.reviewed
    assert doc.reviewed_hash == registry.body_hash(md)
    md.write_text(md.read_text().replace("title: Carte", "title: Carte nouă"))
    assert registry.hash_warnings(doc) == []  # front-matter-ul nu intră în hash


def test_body_edit_after_review_warns_until_reviewed_again(
    session: Session, tmp_path: Path
):
    doc, md = structured_doc(session, tmp_path)
    registry.review(session, doc)
    md.write_text(md.read_text() + "adăugat\n")

    assert registry.hash_warnings(doc) == ["modificat după review"]
    doc = registry.review(session, doc)
    assert registry.hash_warnings(doc) == []


def test_hash_warnings_after_indexing_and_missing_file(
    session: Session, tmp_path: Path
):
    doc, md = structured_doc(session, tmp_path)
    doc = registry.review(session, doc)
    doc.indexed_hash = doc.reviewed_hash
    md.write_text(md.read_text() + "x\n")
    assert registry.hash_warnings(doc) == [
        "modificat după review",
        "modificat după indexare",
    ]
    md.unlink()
    assert registry.hash_warnings(doc) == ["markdown lipsă"]


def test_review_requires_markdown(session: Session, tmp_path: Path):
    doc = reg(session, make_file(tmp_path, "a.txt"))
    with pytest.raises(RegistryError, match="prepare"):
        registry.review(session, doc)


def test_review_unchanged_indexed_document_keeps_stage(
    session: Session, tmp_path: Path
):
    doc, _ = structured_doc(session, tmp_path)
    doc = registry.review(session, doc)
    doc.stage = Stage.indexed
    assert registry.review(session, doc).stage == Stage.indexed


def test_validate_requires_indexed(session: Session, tmp_path: Path):
    doc, _ = structured_doc(session, tmp_path)
    with pytest.raises(RegistryError, match="indexed"):
        registry.validate(session, doc)
    doc.stage = Stage.indexed
    assert registry.validate(session, doc).stage == Stage.validated
    assert registry.validate(session, doc).stage == Stage.validated


def test_record_step_saves_error_without_changing_stage_and_success_clears_it(
    session: Session, tmp_path: Path
):
    doc, _ = structured_doc(session, tmp_path)
    with pytest.raises(RuntimeError), registry.record_step(session, doc):
        doc.stage = Stage.reviewed
        raise RuntimeError("API căzut")

    assert (doc.stage, doc.last_error) == (Stage.structured, "RuntimeError: API căzut")
    with registry.record_step(session, doc):
        pass
    assert doc.last_error is None
