import io
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from google import genai
from pypdf import PdfReader

from app.knowledge_prep.converters import CONVERTERS
from app.knowledge_prep.converters.pdf import (
    ScannedPdfError,
    convert_pdf,
    hyphen_breaks,
    join_hyphenated,
    missing_fraction,
    normalize,
)
from app.knowledge_prep.tests.pdf_factory import make_pdf

LINE = "Planta aceasta ajuta la digestie si curata sangele bine"


class FakeClient:
    """Întoarce pe rând textele din `replies`; `None` = răspuns gol."""

    def __init__(self, replies: list[str | None]):
        self.requests: list[dict] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self._replies.pop(0))

    def sent_page_counts(self) -> list[int]:
        return [
            len(PdfReader(io.BytesIO(r["contents"][1].inline_data.data)).pages)
            for r in self.requests
        ]


def write_pdf(tmp_path: Path, pages: list[list[str]], name: str = "carte.pdf") -> Path:
    path = tmp_path / "files" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf(pages))
    return path


def run(path: Path, client: FakeClient) -> str:
    return convert_pdf(
        path, client=cast(genai.Client, client), model="gemini-2.5-flash"
    )


# --- funcții pure -----------------------------------------------------------


def test_normalize_uses_comma_below_and_fixes_broken_t_glyph():
    assert normalize("şi ţară sănătăĠii ŞŢ") == "și țară sănătății ȘȚ"


def test_missing_fraction_ignores_spurious_spaces_and_layout():
    reference = "Dac ă dăm calului ov ăzul\n62\n"
    assert missing_fraction(reference, "# Titlu\n\nDacă **dăm** calului ovăzul.") < 0.2


def test_missing_fraction_detects_omitted_sentence():
    reference = " ".join(
        f"Propozitia numarul {i} are continut propriu." for i in range(20)
    )
    omitted = reference.replace("Propozitia numarul 7 are continut propriu.", "")

    assert missing_fraction(reference, reference) == 0
    assert 0.03 < missing_fraction(reference, omitted) < 0.08


def test_hyphen_breaks_joins_syllable_splits_and_keeps_real_hyphens():
    text = (
        "un gust ex-\ncelent, pen-\ntru ca\n"
        "s-\na dus într-\no zi la floarea-\nmalaiului si\n"
        "ne-\nvoie mare\n"
        "Floarea-malaiului e buna. Avem nevoie de ea."
    )

    breaks = hyphen_breaks(text)

    assert {("ex", "celent"), ("pen", "tru"), ("ne", "voie")} <= breaks
    assert ("s", "a") not in breaks  # clitic
    assert ("într", "o") not in breaks  # clitic
    assert (
        "floarea",
        "malaiului",
    ) not in breaks  # apare cu cratimă și în interiorul rândului


def test_hyphen_breaks_prefers_evidence_from_whole_document():
    # "ma-i" arată ca un clitic ("dă-i"), dar "mai" apare întreg în restul cărții
    assert ("ma", "i") in hyphen_breaks("ma-\ni", context="mai mult")
    assert ("ma", "i") not in hyphen_breaks("ma-\ni")


def test_join_hyphenated_only_touches_known_breaks():
    md = "Un gust ex-celent, s-a dus pen-tru floarea-mălaiului. Ex-Celent!"
    assert (
        join_hyphenated(md, {("ex", "celent"), ("pen", "tru")})
        == "Un gust excelent, s-a dus pentru floarea-mălaiului. ExCelent!"
    )


def test_hyphen_breaks_match_across_diacritic_variants():
    breaks = hyphen_breaks("pro-\nprietăĠi")
    assert join_hyphenated("pro-prietăți", breaks) == "proprietăți"


# --- convert_pdf ---------------------------------------------------------------


def test_convert_pdf_rejects_scanned_pdf(tmp_path: Path):
    path = write_pdf(tmp_path, [[], [], [LINE]])

    with pytest.raises(ScannedPdfError, match="OCR"):
        run(path, FakeClient([]))


def test_convert_pdf_sends_chunks_of_8_pages_and_joins_them(tmp_path: Path):
    path = write_pdf(tmp_path, [[f"{LINE} {i}"] for i in range(10)])
    first = "\n".join(f"{LINE} {i}" for i in range(8))
    client = FakeClient([f"# Capitol\n\n{first}\n", f"{LINE} 8\n{LINE} 9"])

    md = run(path, client)

    assert client.sent_page_counts() == [8, 2]
    assert md == f"# Capitol\n\n{first}\n\n{LINE} 8\n{LINE} 9\n"
    req = client.requests[0]
    assert "diacriticele" in req["contents"][0]
    assert req["contents"][1].inline_data.mime_type == "application/pdf"
    assert req["config"].thinking_config.thinking_budget == 0


def test_convert_pdf_joins_line_end_hyphenation_and_normalizes_diacritics(
    tmp_path: Path,
):
    path = write_pdf(tmp_path, [[f"{LINE} un gust ex-", "celent si s-a dus"]])
    client = FakeClient([f"{LINE} un gust ex-celent şi s-a dus"])

    assert run(path, client) == f"{LINE} un gust excelent și s-a dus\n"


def test_convert_pdf_drops_page_numbers_left_on_their_own_line(tmp_path: Path):
    path = write_pdf(tmp_path, [[LINE], [LINE]])
    client = FakeClient([f"{LINE}\n\n39\n\n{LINE}\n\n- 5\n"])

    assert run(path, client) == f"{LINE}\n\n{LINE}\n\n- 5\n"


def test_convert_pdf_retries_chunk_with_missing_text_then_accepts(tmp_path: Path):
    lines = [f"{LINE} {i}" for i in range(4)]
    path = write_pdf(tmp_path, [lines])
    client = FakeClient([lines[0], "\n".join(lines)])

    assert run(path, client) == "\n".join(lines) + "\n"
    assert len(client.requests) == 2


def test_convert_pdf_marks_chunk_still_missing_text_after_retry(tmp_path: Path):
    lines = [f"{LINE} {i}" for i in range(4)]
    path = write_pdf(tmp_path, [lines])
    client = FakeClient([lines[0], "\n".join(lines[:2])])

    md = run(path, client)

    assert md.startswith("<!-- VERIFICĂ: paginile 1-1, lipsesc ~")
    assert md.endswith("\n".join(lines[:2]) + "\n")  # se păstrează încercarea mai bună


def test_convert_pdf_falls_back_to_pdf_text_when_gemini_returns_nothing(
    tmp_path: Path,
):
    path = write_pdf(tmp_path, [[LINE]])

    md = run(path, FakeClient([None, ""]))

    assert "VERIFICĂ: paginile 1-1" in md
    assert "text brut din PDF" in md
    assert LINE in md


def test_convert_pdf_reuses_cached_chunks_and_cleans_cache(tmp_path: Path):
    path = write_pdf(tmp_path, [[f"{LINE} {i}"] for i in range(10)])
    cache = tmp_path / ".cache" / "pdf" / "carte"
    cache.mkdir(parents=True)
    (cache / "p0001-0008.md").write_text("din cache\n")
    client = FakeClient([f"{LINE} 8\n{LINE} 9"])

    md = run(path, client)

    assert md.startswith("din cache\n\n")
    assert client.sent_page_counts() == [2]
    assert not (tmp_path / ".cache").exists()


def test_convert_pdf_keeps_finished_chunks_in_cache_when_interrupted(tmp_path: Path):
    path = write_pdf(tmp_path, [[f"{LINE} {i}"] for i in range(10)])
    first = "\n".join(f"{LINE} {i}" for i in range(8))
    client = FakeClient([first])  # a doua bucată: IndexError, ca o întrerupere

    with pytest.raises(IndexError):
        run(path, client)

    assert (tmp_path / ".cache" / "pdf" / "carte" / "p0001-0008.md").exists()


def test_pdf_is_registered():
    assert CONVERTERS[".pdf"] is convert_pdf
