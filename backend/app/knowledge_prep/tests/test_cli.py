import logging
from collections.abc import Mapping
from pathlib import Path

import pytest
from langchain_core.vectorstores import VectorStore
from sqlmodel import Session, select

from app.knowledge_prep.__main__ import main
from app.knowledge_prep.converters import Converter
from app.knowledge_prep.models import Document, Stage
from app.knowledge_prep.tests.index_fakes import FakeQuestions, FakeVectors
from app.knowledge_prep.tests.retrieval_fakes import FakeChildStore
from app.knowledge_prep.vectors import PgVectorIndex

MID = " ".join(["Grâul încolțit dă putere organismului slăbit."] * 10)


def fake_converter(path: Path) -> str:
    return f"conținut din {path.name}\n"


def fake_audio(_path: Path) -> str:
    return "transcriere\n"


CONVERTERS = {".txt": fake_converter, ".ogg": fake_audio}


def make_file(tmp_path: Path, name: str, content: str = "brut") -> Path:
    f = tmp_path / "baza" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


def fake_restructure(body: str, title: str) -> str:
    return f"# {title}\n\n{body}"


def run(
    session: Session,
    *argv: str,
    converters: Mapping[str, Converter] = CONVERTERS,
    vectors: FakeVectors | None = None,
    vectorstore: VectorStore | None = None,
) -> int:
    return main(
        list(argv),
        session=session,
        converters=converters,
        restructurer=fake_restructure,
        vectors=vectors or FakeVectors(),
        questioner=FakeQuestions(),
        vectorstore=vectorstore if vectorstore is not None else FakeChildStore([]),
    )


def stages(session: Session) -> dict[str, Stage]:
    session.expire_all()
    return {d.slug: d.stage for d in session.exec(select(Document))}


def test_register_folder_skips_converted_and_unsupported_and_is_idempotent(
    session: Session, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "sub/b.ogg")
    make_file(tmp_path, "c.xyz")
    make_file(tmp_path, "converted/vechi.txt")
    folder = str(tmp_path / "baza")

    with caplog.at_level(logging.INFO):
        assert run(session, "register", folder) == 0
        assert run(session, "register", folder) == 0

    assert stages(session) == {"a": Stage.uploaded, "b": Stage.uploaded}
    assert "c.xyz" in caplog.text and "deja înregistrat" in caplog.text


def test_register_structured_folder(session: Session, tmp_path: Path):
    make_file(tmp_path, "extracted/bucata_1.md", "## Nuci\n")
    make_file(tmp_path, "extracted/bucata_2.md", "## Gât\n")

    code = run(
        session,
        "register",
        "--structured",
        "--type",
        "conversatie",
        str(tmp_path / "baza"),
    )

    assert code == 0
    assert stages(session) == {
        "bucata-1": Stage.structured,
        "bucata-2": Stage.structured,
    }


def test_register_rejects_title_for_folder(session: Session, tmp_path: Path):
    make_file(tmp_path, "a.txt")
    with pytest.raises(SystemExit):
        run(session, "register", str(tmp_path / "baza"), "--title", "X")


def test_register_single_file_error_returns_1(
    session: Session, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    assert run(session, "register", str(make_file(tmp_path, "a.xyz"))) == 1
    assert ".xyz" in caplog.text


def test_full_flow_prepare_review_status(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    src = make_file(tmp_path, "a.txt")
    run(session, "register", str(src), "--no-split")

    assert run(session, "prepare", "a") == 0
    assert run(session, "review", str(src)) == 0
    md = tmp_path / "baza" / "converted" / "a.md"
    md.write_text(md.read_text() + "editat\n")
    capsys.readouterr()

    assert run(session, "status") == 0
    out = capsys.readouterr().out
    assert "reviewed" in out and "modificat după review" in out


def test_status_shows_last_error(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run(session, "register", str(make_file(tmp_path, "a.txt")))

    def broken(_p: Path) -> str:
        raise RuntimeError("API căzut")

    assert run(session, "prepare", "a", converters={".txt": broken}) == 1
    capsys.readouterr()
    assert run(session, "status", "a") == 0
    out = capsys.readouterr().out
    assert "uploaded" in out and "RuntimeError: API căzut" in out


def test_prepare_folder_only_uploaded_without_force(
    session: Session, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "sub/b.txt")
    folder = str(tmp_path / "baza")
    run(session, "register", folder)
    run(session, "prepare", str(tmp_path / "baza" / "a.txt"))

    with caplog.at_level(logging.INFO):
        assert run(session, "prepare", folder) == 0

    assert stages(session) == {"a": Stage.structured, "b": Stage.structured}
    assert "sărit" in caplog.text


def test_prepare_folder_continues_after_a_failure(session: Session, tmp_path: Path):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "b.ogg")
    run(session, "register", str(tmp_path / "baza"))

    def broken(_p: Path) -> str:
        raise RuntimeError("API căzut")

    code = run(
        session,
        "prepare",
        str(tmp_path / "baza"),
        converters={".txt": broken, ".ogg": fake_audio},
    )

    assert code == 1
    assert stages(session) == {"a": Stage.uploaded, "b": Stage.structured}


def test_single_target_rule_violation_returns_1(
    session: Session, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    run(session, "register", str(make_file(tmp_path, "a.txt")))
    assert run(session, "validate", "a") == 1
    assert "indexed" in caplog.text
    assert run(session, "review", "nu-exista") == 1


def test_broken_frontmatter_is_reported_not_crashing(
    session: Session,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    md = make_file(tmp_path, "converted/a.md", "---\ntitle: A\n---\n\ncorp\n")
    run(session, "register", "--structured", str(md))
    md.write_text("---\ntitle: A\ncorp stricat\n")  # al doilea '---' șters în IDE
    capsys.readouterr()

    assert run(session, "status") == 0
    assert "neînchis" in capsys.readouterr().out
    assert run(session, "review", "a") == 1
    assert "neînchis" in caplog.text


def test_prepare_restructures_unless_no_restructure(session: Session, tmp_path: Path):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "b.txt")
    run(session, "register", str(tmp_path / "baza"))

    assert run(session, "prepare", "a") == 0
    assert run(session, "prepare", "b", "--no-restructure") == 0

    converted = tmp_path / "baza" / "converted"
    assert (converted / "a.md").read_text().endswith("\n# a\n\nconținut din a.txt\n")
    assert (converted / "b.md").read_text().endswith("---\n\nconținut din b.txt\n")


def test_index_flow_no_questions_then_force_after_validate(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    md = make_file(tmp_path, "carte.md", f"# Carte\n\n## A\n\n{MID}\n")
    assert run(session, "register", str(md), "--structured") == 0
    assert run(session, "index", "carte") == 1  # încă 'structured'
    assert run(session, "review", "carte") == 0

    vectors = FakeVectors()
    assert run(session, "index", "carte", "--no-questions", vectors=vectors) == 0
    assert {e.kind for e in vectors.rows.values()} == {"content"}
    assert stages(session) == {"carte": Stage.indexed}

    assert run(session, "validate", "carte") == 0
    assert (
        run(session, "index", "carte", vectors=vectors) == 1
    )  # validated, fără --force
    assert stages(session) == {"carte": Stage.validated}
    assert run(session, "index", "carte", "--force", vectors=vectors) == 0
    assert {e.kind for e in vectors.rows.values()} == {"content", "question"}

    capsys.readouterr()
    assert run(session, "status", "carte") == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.split() == [
        "carte",
        "carte",
        "indexed",
    ]  # fără avertismente, fără eroare


def test_index_refuses_before_building_real_vector_index(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """`check_indexable` trebuie să respingă documentul înainte de a construi
    `PgVectorIndex.from_settings()` (care ar face I/O real spre Postgres/OpenAI)."""
    calls: list[None] = []

    def fake_from_settings() -> FakeVectors:
        calls.append(None)
        raise AssertionError("PgVectorIndex.from_settings nu trebuie construit")

    monkeypatch.setattr(PgVectorIndex, "from_settings", fake_from_settings)

    md = make_file(tmp_path, "carte.md", f"# Carte\n\n{MID}\n")
    run(
        session, "register", str(md), "--structured"
    )  # stadiu 'structured', nu 'reviewed'

    with caplog.at_level(logging.INFO):
        code = main(
            ["index", "carte"],
            session=session,
            converters=CONVERTERS,
            restructurer=fake_restructure,
            vectors=None,
            questioner=FakeQuestions(),
        )

    assert code == 1
    assert not calls  # from_settings nu a fost apelat
    assert "cere 'reviewed'" in caplog.text
    assert stages(session) == {"carte": Stage.structured}


def test_index_failure_returns_1_and_status_shows_error(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    md = make_file(tmp_path, "carte.md", f"# Carte\n\n{MID}\n")
    run(session, "register", str(md), "--structured")
    run(session, "review", "carte")

    assert run(session, "index", "carte", vectors=FakeVectors(fail="embed")) == 1
    capsys.readouterr()
    run(session, "status")
    assert "RuntimeError: OpenAI căzut" in capsys.readouterr().out
    assert stages(session) == {"carte": Stage.reviewed}


def section(word: str) -> str:
    return " ".join([f"{word} se pregătește cu grijă în fiecare dimineață."] * 15)


def indexed_book(session: Session, tmp_path: Path) -> FakeVectors:
    body = (
        f"# Carte\n\n## Grâul\n\n{section('Grâul')}\n\n"
        f"## Orzul\n\n{section('Orzul zxqvortex')}\n\n## Meiul\n\n{section('Meiul')}\n"
    )
    md = make_file(tmp_path, "carte.md", body)
    assert run(session, "register", str(md), "--structured") == 0
    assert run(session, "review", "carte") == 0
    vectors = FakeVectors()
    assert run(session, "index", "carte", vectors=vectors) == 0
    return vectors


def test_search_prints_ranked_chunks(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = FakeChildStore.from_entries(indexed_book(session, tmp_path).rows.values())
    capsys.readouterr()

    assert run(session, "search", "zxqvortex", "-k", "2", vectorstore=store) == 0

    out = capsys.readouterr().out
    # Orzul e și în lista semantică, și singurul hit BM25: primul după RRF
    assert "1. carte — Carte > Orzul\n" in out
    assert "2. carte — Carte > " in out
    assert "3. " not in out
    assert store.calls[-1]["filter"] is None


def test_search_baseline_without_bm25(
    session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = FakeChildStore.from_entries(indexed_book(session, tmp_path).rows.values())
    capsys.readouterr()

    code = run(
        session,
        "search",
        "zxqvortex",
        "--no-questions",
        "--no-hybrid",
        vectorstore=store,
    )

    assert code == 0
    assert store.calls[-1]["filter"] == {"kind": "content"}
    assert capsys.readouterr().out.count("carte — ") == 3


def test_search_without_results(session: Session, capsys: pytest.CaptureFixture[str]):
    assert run(session, "search", "grâu", "--no-hybrid") == 0
    assert capsys.readouterr().out == "niciun rezultat\n"


def test_search_rejects_non_positive_k(session: Session):
    with pytest.raises(SystemExit):
        run(session, "search", "grâu", "-k", "0")


def test_search_rejects_blank_query(session: Session, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert run(session, "search", "   ") == 1
    assert "întrebarea e goală" in caplog.text
