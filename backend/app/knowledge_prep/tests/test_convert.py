import logging
from pathlib import Path

import pytest

from app.knowledge_prep.convert import ConversionError, convert, main
from app.knowledge_prep.frontmatter import parse, render


def fake_converter(path: Path) -> str:
    return f"conținut din {path.name}\n"


CONVERTERS = {".txt": fake_converter, ".ogg": fake_converter}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "knowledge"


def make_source(tmp_path: Path, name: str, content: str = "brut") -> Path:
    src = tmp_path / "baza" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(content)
    return src


def written(src: Path, root: Path, **kwargs) -> Path:
    """convert(...) care trebuie să fi scris un fișier."""
    out = convert(src, root=root, converters=CONVERTERS, **kwargs)
    assert out is not None
    return out


def test_convert_copies_source_and_writes_draft_markdown(tmp_path: Path, root: Path):
    src = make_source(tmp_path, "Regim DRI - Anemie.txt")

    out = written(src, root, doc_type="regim")

    assert out == root / "converted" / "regim-dri-anemie.md"
    assert (root / "files" / "Regim DRI - Anemie.txt").read_text() == "brut"
    fm, body = parse(out.read_text())
    assert (fm.title, fm.slug, fm.type, fm.status) == (
        "Regim DRI - Anemie",
        "regim-dri-anemie",
        "regim",
        "draft",
    )
    assert fm.source_file == "files/Regim DRI - Anemie.txt"
    assert body == "conținut din Regim DRI - Anemie.txt\n"


def test_convert_uses_explicit_title_and_author(tmp_path: Path, root: Path):
    src = make_source(tmp_path, "Maria_Treben.txt")

    out = written(
        src, root, title="Sănătate din farmacia Domnului", author="Maria Treben"
    )

    fm, _ = parse(out.read_text())
    assert (fm.slug, fm.author) == ("sanatate-din-farmacia-domnului", "Maria Treben")


def test_convert_infers_type_transcriere_for_audio_and_carte_otherwise(
    tmp_path: Path, root: Path
):
    audio = written(make_source(tmp_path, "digestia.ogg"), root)
    text = written(make_source(tmp_path, "Alimente.txt"), root)

    assert parse(audio.read_text())[0].type == "transcriere"
    assert parse(text.read_text())[0].type == "carte"


def test_convert_skips_existing_draft_without_force(tmp_path: Path, root: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src, root)
    out.write_text(out.read_text() + "editat de om\n")

    assert convert(src, root=root, converters=CONVERTERS) is None
    assert out.read_text().endswith("editat de om\n")


def test_convert_overwrites_draft_with_force(tmp_path: Path, root: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src, root)
    out.write_text(out.read_text() + "editat de om\n")

    assert written(src, root, force=True) == out
    assert not out.read_text().endswith("editat de om\n")


def test_convert_never_overwrites_reviewed_even_with_force(tmp_path: Path, root: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src, root)
    fm, body = parse(out.read_text())
    fm.status = "reviewed"
    out.write_text(render(fm, body + "review\n"))

    with pytest.raises(ConversionError, match="reviewed"):
        convert(src, root=root, converters=CONVERTERS, force=True)
    assert out.read_text().endswith("review\n")


def test_convert_rejects_slug_already_used_by_another_source(
    tmp_path: Path, root: Path
):
    written(make_source(tmp_path, "unu.txt"), root, title="Același titlu")

    with pytest.raises(ConversionError, match="slug"):
        convert(
            make_source(tmp_path, "doi.txt"),
            root=root,
            converters=CONVERTERS,
            title="Același titlu",
            force=True,
        )


def test_convert_rejects_different_file_with_same_name_in_files(
    tmp_path: Path, root: Path
):
    written(make_source(tmp_path, "a.txt", "v1"), root)
    other = tmp_path / "alt_folder" / "a.txt"
    other.parent.mkdir()
    other.write_text("v2")

    with pytest.raises(ConversionError, match="files/"):
        convert(other, root=root, converters=CONVERTERS, force=True)


def test_convert_accepts_source_already_inside_files(root: Path):
    src = root / "files" / "deja.txt"
    src.parent.mkdir(parents=True)
    src.write_text("brut")

    out = written(src, root)

    assert parse(out.read_text())[0].source_file == "files/deja.txt"


def test_convert_rejects_unsupported_extension(tmp_path: Path, root: Path):
    with pytest.raises(ConversionError, match=".xyz"):
        convert(make_source(tmp_path, "a.xyz"), root=root, converters=CONVERTERS)
    assert not (root / "files").exists()


def test_main_converts_folder_and_skips_unsupported_files(
    tmp_path: Path, root: Path, caplog: pytest.LogCaptureFixture
):
    make_source(tmp_path, "a.txt")
    make_source(tmp_path, "sub/b.ogg")
    make_source(tmp_path, "c.xyz")

    with caplog.at_level(logging.INFO):
        code = main(
            [str(tmp_path / "baza"), "--root", str(root)], converters=CONVERTERS
        )

    assert code == 0
    assert sorted(p.name for p in (root / "converted").iterdir()) == ["a.md", "b.md"]
    assert "c.xyz" in caplog.text


def test_main_rejects_title_for_folder(tmp_path: Path, root: Path):
    make_source(tmp_path, "a.txt")

    with pytest.raises(SystemExit):
        main(
            [str(tmp_path / "baza"), "--root", str(root), "--title", "X"],
            converters=CONVERTERS,
        )


def test_main_returns_error_code_when_a_file_fails(tmp_path: Path, root: Path):
    make_source(tmp_path, "a.txt")

    def broken(_path: Path) -> str:
        raise RuntimeError("API căzut")

    code = main(
        [str(tmp_path / "baza"), "--root", str(root)], converters={".txt": broken}
    )

    assert code == 1
