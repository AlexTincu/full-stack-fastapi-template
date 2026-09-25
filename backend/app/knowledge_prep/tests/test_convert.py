import logging
from pathlib import Path

import pytest

from app.knowledge_prep.convert import ConversionError, convert, main
from app.knowledge_prep.frontmatter import parse, render


def fake_converter(path: Path) -> str:
    return f"conținut din {path.name}\n"


CONVERTERS = {".txt": fake_converter, ".ogg": fake_converter}


def make_source(tmp_path: Path, name: str, content: str = "brut") -> Path:
    src = tmp_path / "baza" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(content)
    return src


def written(src: Path, **kwargs) -> Path:
    """convert(...) care trebuie să fi scris un fișier."""
    out = convert(src, converters=CONVERTERS, **kwargs)
    assert out is not None
    return out


def test_convert_writes_draft_markdown_next_to_source_in_converted(tmp_path: Path):
    src = make_source(tmp_path, "regimuri/Regim DRI - Anemie.txt")

    out = written(src, doc_type="regim")

    assert out == tmp_path / "baza" / "regimuri" / "converted" / "regim-dri-anemie.md"
    assert src.read_text() == "brut"  # sursa rămâne neatinsă, pe loc
    fm, body = parse(out.read_text())
    assert (fm.title, fm.slug, fm.type, fm.status) == (
        "Regim DRI - Anemie",
        "regim-dri-anemie",
        "regim",
        "draft",
    )
    assert fm.source_file == "Regim DRI - Anemie.txt"
    assert body == "conținut din Regim DRI - Anemie.txt\n"


def test_convert_uses_explicit_title_and_author(tmp_path: Path):
    src = make_source(tmp_path, "Maria_Treben.txt")

    out = written(src, title="Sănătate din farmacia Domnului", author="Maria Treben")

    fm, _ = parse(out.read_text())
    assert (fm.slug, fm.author) == ("sanatate-din-farmacia-domnului", "Maria Treben")


def test_convert_infers_type_transcriere_for_audio_and_carte_otherwise(tmp_path: Path):
    audio = written(make_source(tmp_path, "digestia.ogg"))
    text = written(make_source(tmp_path, "Alimente.txt"))

    assert parse(audio.read_text())[0].type == "transcriere"
    assert parse(text.read_text())[0].type == "carte"


def test_convert_skips_existing_draft_without_force(tmp_path: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src)
    out.write_text(out.read_text() + "editat de om\n")

    assert convert(src, converters=CONVERTERS) is None
    assert out.read_text().endswith("editat de om\n")


def test_convert_overwrites_draft_with_force(tmp_path: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src)
    out.write_text(out.read_text() + "editat de om\n")

    assert written(src, force=True) == out
    assert not out.read_text().endswith("editat de om\n")


def test_convert_never_overwrites_reviewed_even_with_force(tmp_path: Path):
    src = make_source(tmp_path, "a.txt")
    out = written(src)
    fm, body = parse(out.read_text())
    fm.status = "reviewed"
    out.write_text(render(fm, body + "review\n"))

    with pytest.raises(ConversionError, match="reviewed"):
        convert(src, converters=CONVERTERS, force=True)
    assert out.read_text().endswith("review\n")


def test_convert_rejects_slug_already_used_by_another_source(tmp_path: Path):
    written(make_source(tmp_path, "unu.txt"), title="Același titlu")

    with pytest.raises(ConversionError, match="slug"):
        convert(
            make_source(tmp_path, "doi.txt"),
            converters=CONVERTERS,
            title="Același titlu",
            force=True,
        )


def test_convert_rejects_unsupported_extension(tmp_path: Path):
    with pytest.raises(ConversionError, match=".xyz"):
        convert(make_source(tmp_path, "a.xyz"), converters=CONVERTERS)
    assert not (tmp_path / "baza" / "converted").exists()


def test_main_converts_folder_into_each_subfolder_converted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    make_source(tmp_path, "a.txt")
    make_source(tmp_path, "sub/b.ogg")
    make_source(tmp_path, "c.xyz")

    with caplog.at_level(logging.INFO):
        code = main([str(tmp_path / "baza")], converters=CONVERTERS)

    assert code == 0
    assert [p.name for p in (tmp_path / "baza" / "converted").iterdir()] == ["a.md"]
    assert [p.name for p in (tmp_path / "baza" / "sub" / "converted").iterdir()] == [
        "b.md"
    ]
    assert "c.xyz" in caplog.text


def test_main_does_not_reconvert_files_inside_converted_folders(tmp_path: Path):
    make_source(tmp_path, "a.txt")
    main([str(tmp_path / "baza")], converters=CONVERTERS)

    (tmp_path / "baza" / "converted" / "note.txt").write_text("de mână")
    code = main([str(tmp_path / "baza")], converters=CONVERTERS)

    assert code == 0
    assert sorted(p.name for p in (tmp_path / "baza" / "converted").iterdir()) == [
        "a.md",
        "note.txt",
    ]


def test_main_rejects_title_for_folder(tmp_path: Path):
    make_source(tmp_path, "a.txt")

    with pytest.raises(SystemExit):
        main([str(tmp_path / "baza"), "--title", "X"], converters=CONVERTERS)


def test_main_returns_error_code_when_a_file_fails(tmp_path: Path):
    make_source(tmp_path, "a.txt")

    def broken(_path: Path) -> str:
        raise RuntimeError("API căzut")

    code = main([str(tmp_path / "baza")], converters={".txt": broken})

    assert code == 1
