from pathlib import Path

import pytest

from app.knowledge_prep.convert import ConversionError, convert


def fake_converter(path: Path) -> str:
    return f"conținut din {path.name}\n"


def test_convert_returns_body_from_matching_converter(tmp_path: Path):
    src = tmp_path / "Regim DRI - Anemie.txt"
    src.write_text("brut")

    assert convert(src, converters={".txt": fake_converter}) == (
        "conținut din Regim DRI - Anemie.txt\n"
    )
    assert src.read_text() == "brut"


def test_convert_matches_extension_case_insensitively(tmp_path: Path):
    src = tmp_path / "a.TXT"
    src.write_text("brut")
    assert convert(src, converters={".txt": fake_converter}) == "conținut din a.TXT\n"


def test_convert_rejects_unsupported_extension(tmp_path: Path):
    with pytest.raises(ConversionError, match=".xyz"):
        convert(tmp_path / "a.xyz", converters={".txt": fake_converter})
