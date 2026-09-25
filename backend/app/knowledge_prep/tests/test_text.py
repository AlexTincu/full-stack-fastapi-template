from pathlib import Path

import pytest

from app.knowledge_prep.converters import CONVERTERS
from app.knowledge_prep.converters.diacritics import normalize
from app.knowledge_prep.converters.text import TEXT_EXTENSIONS, convert_text


def write(tmp_path: Path, content: bytes, name: str = "Cerealele.md") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_normalize_uses_comma_below_and_fixes_broken_t_glyph():
    assert normalize("şi ţară sănătăĠii ŞŢ") == "și țară sănătății ȘȚ"


def test_convert_text_cleans_line_endings_spaces_and_blank_lines(tmp_path: Path):
    raw = "\n\nRegimul pentru anemie. \r\n\r\n\r\n\r\nCartofii  cu frunze.\t\r\n\n\n"
    path = write(tmp_path, raw.encode())

    assert convert_text(path) == "Regimul pentru anemie.\n\nCartofii  cu frunze.\n"


def test_convert_text_normalizes_diacritics_bom_and_unicode_composition(
    tmp_path: Path,
):
    decomposed = "t\u0326a\u0306"  # ț și ă scrise ca literă + semn separat
    path = write(tmp_path, "\ufeffşi ţie ".encode() + decomposed.encode())

    assert convert_text(path) == "și ție ță\n"


def test_convert_text_keeps_markdown_and_content_untouched(tmp_path: Path):
    md = "### Titlu\n\n* listă\n  * sub-listă\nrând rupt\nîn paragraf ...fără diacritice\n"
    path = write(tmp_path, md.encode())

    assert convert_text(path) == md


def test_convert_text_rejects_already_converted_markdown(tmp_path: Path):
    path = write(tmp_path, b"---\ntitle: x\nslug: x\n---\n\ncorp\n")

    with pytest.raises(ValueError, match="front-matter"):
        convert_text(path)


def test_convert_text_rejects_non_utf8(tmp_path: Path):
    path = write(tmp_path, "ţară".encode("cp1250"), "a.txt")

    with pytest.raises(ValueError, match="UTF-8"):
        convert_text(path)


def test_convert_text_rejects_empty_file(tmp_path: Path):
    with pytest.raises(ValueError, match="gol"):
        convert_text(write(tmp_path, b" \r\n\n", "a.txt"))


def test_text_extensions_are_registered():
    assert TEXT_EXTENSIONS == {".txt", ".md"}
    assert all(CONVERTERS[ext] is convert_text for ext in TEXT_EXTENSIONS)
