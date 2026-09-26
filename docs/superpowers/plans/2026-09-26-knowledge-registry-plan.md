# Registrul de cunoștințe (faza 1) — plan de implementare

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Postgres ține registrul documentelor (stadiu, căi, hash-uri), iar CLI-ul `python -m app.knowledge_prep` le mută prin `register → prepare → review → validate`, cu `status` ca vedere de ansamblu.

**Architecture:** Modele SQLModel (`document`, `chunk`, `question`) în `app/knowledge_prep/models.py` + migrare Alembic. Regulile de stadiu stau într-un singur modul, `registry.py`, cu funcții care primesc un `Session` (le va refolosi și REST-ul din faza 5). `__main__.py` e doar CLI-ul subțire peste `registry.py`. `convert.py` rămâne biblioteca de conversie (sursă → corp markdown); protecțiile lui trec în registru.

**Tech Stack:** Python 3.14, SQLModel 0.0.42 / SQLAlchemy 2.0, Alembic, argparse, pytest (SQLite in-memory pentru teste).

**Spec:** `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` (secțiunile „Modelul de date”, „Stadii”, „Comenzi”, faza 1).

## Global Constraints

- Toate comenzile se rulează din `full-stack-fastapi-template/backend/` (`uv run ...`).
- Registrul decide stadiul; front-matter-ul **nu mai are `status`**; se scrie la `prepare` doar pentru lizibilitate.
- Hash-ul = sha256 pe corpul markdown-ului, **fără front-matter**.
- Stadii, în ordine: `uploaded → structured → reviewed → indexed → validated`.
- Tipuri: `carte | regim | nota | insight | transcriere | conversatie`. `--type` implicit: `transcriere` pentru audio, altfel `carte`. `split` implicit `false` pentru `regim`, `nota`, `insight`; `--split/--no-split` suprascrie.
- `prepare` scrie în `<folderul originalului>/converted/<slug>.md`.
- `prepare` pe un document fără original eșuează cu mesaj clar; pe unul `reviewed` sau mai departe cere `--force` și îl readuce în `structured`.
- Un pas eșuat scrie `last_error` și nu schimbă stadiul; un pas reușit golește `last_error`.
- Țintă = id, slug sau cale (a originalului ori a markdown-ului). `register` și `prepare` acceptă și folder, parcurs recursiv; un fișier eșuat nu oprește restul.
- `rag_simple` rămâne neatins. Fără restructurare în faza asta (`--restructure` vine în faza 2).
- Mesajele pentru user sunt în română, ca în codul existent.

## Review Focus

1. **Fișierele deja convertite, cu `status: draft|reviewed` în front-matter** (`audio DRI/converted/*.md`) intră cu `register --structured`: `status` e ignorat, iar originalul din `source_file` se leagă automat dacă există lângă `converted/`.
2. **Markdown fără front-matter** (`whatsapp/extracted/**/*.md`, încep cu `<!-- bucata ... -->`): `register --structured` ia titlul din numele fișierului, nu crapă.
3. **Căi cu spații și diacritice, relative sau absolute** (`"audio DRI/Rugăciunea este intimitate.ogg"`): ținta se rezolvă la același document indiferent de forma căii.
4. **`register` rulat de două ori pe același folder** e idempotent: documentele existente sunt „sărite”, nu erori, iar codul de ieșire e 0.
5. **Editarea doar a front-matter-ului după `review`** nu declanșează „modificat după review”; editarea corpului da, iar un nou `review` o stinge.

---

## Structura fișierelor

| Fișier | Rol |
|---|---|
| Create `backend/app/knowledge_prep/models.py` | `DocumentType`, `Stage`, `Document`, `Chunk`, `Question` |
| Create `backend/app/alembic/versions/7b3f1c2a9d10_add_knowledge_registry.py` | tabelele `document`, `chunk`, `question` |
| Modify `backend/app/alembic/env.py` | importă `app.knowledge_prep.models` |
| Modify `backend/app/knowledge_prep/frontmatter.py` | scoate `status`; `split_frontmatter` tolerant (lipsă, chei necunoscute) |
| Modify `backend/app/knowledge_prep/convert.py` | doar `convert(source) -> str` + `ConversionError`; fără CLI |
| Create `backend/app/knowledge_prep/registry.py` | toate regulile: register, resolve, prepare, review, validate, status |
| Create `backend/app/knowledge_prep/__main__.py` | CLI cu subcomenzi |
| Create `backend/app/knowledge_prep/tests/conftest.py` | fixture `session` pe SQLite in-memory |
| Create `backend/app/knowledge_prep/tests/test_registry.py`, `test_cli.py` | teste |
| Modify `backend/app/knowledge_prep/tests/test_convert.py`, `test_frontmatter.py` | adaptate |
| Modify `RAG/AGENTS.md` (în afara repo-ului) | comenzile noi, fluxul fără `status` |

---

### Task 1: Front-matter fără `status`, `convert` ca bibliotecă

**Files:**
- Modify: `backend/app/knowledge_prep/frontmatter.py`
- Modify: `backend/app/knowledge_prep/convert.py`
- Test: `backend/app/knowledge_prep/tests/test_frontmatter.py`, `backend/app/knowledge_prep/tests/test_convert.py`

**Interfaces:**
- Produces:
  - `FrontMatter(title: str, slug: str, type: str, author: str | None, source_file: str | None = None)`
  - `split_frontmatter(text: str) -> tuple[dict[str, Any], str]` — `({}, text)` dacă nu există front-matter; `ValueError` dacă e neînchis sau nu e mapping.
  - `strip_frontmatter(text: str) -> str`
  - `render(fm: FrontMatter, body: str) -> str` (neschimbat)
  - `slugify`, `title_from_filename` (neschimbate)
  - `convert.CONVERTED_DIR = "converted"`, `convert.ConversionError`, `convert.convert(source: Path, *, converters: Mapping[str, Converter] = CONVERTERS) -> str`

- [ ] **Step 1: Rescrie testele de front-matter**

În `test_frontmatter.py`: importă `FrontMatter, render, slugify, split_frontmatter, strip_frontmatter, title_from_filename`; păstrează testele `slugify`/`title_from_filename`; înlocuiește roundtrip-ul și adaugă:

```python
def test_render_then_split_roundtrips_metadata_and_body():
    fm = FrontMatter(
        title="Rostul suferinței",
        slug="rostul-suferintei",
        type="transcriere",
        author=None,
        source_file="rostul suferintei.ogg",
    )
    body = "# Titlu\n\nText cu ș și ț.\n"

    meta, parsed_body = split_frontmatter(render(fm, body))

    assert meta == asdict(fm)
    assert "status" not in meta
    assert parsed_body == body


def test_split_frontmatter_without_frontmatter_returns_whole_text():
    text = "<!-- bucata 020 -->\n\n## Nuci\n"
    assert split_frontmatter(text) == ({}, text)


def test_split_frontmatter_keeps_unknown_keys_like_legacy_status():
    meta, body = split_frontmatter("---\ntitle: X\nstatus: reviewed\n---\n\ncorp\n")
    assert meta == {"title": "X", "status": "reviewed"}
    assert body == "corp\n"


def test_split_frontmatter_rejects_unclosed_block():
    with pytest.raises(ValueError, match="neînchis"):
        split_frontmatter("---\ntitle: X\ncorp\n")


def test_strip_frontmatter_ignores_metadata():
    fm = FrontMatter(title="A", slug="a", type="carte", author=None)
    assert strip_frontmatter(render(fm, "corp\n")) == "corp\n"
    assert strip_frontmatter("fără front-matter\n") == "fără front-matter\n"
```

Păstrează `test_render_puts_frontmatter_block_first_and_keeps_diacritics_readable` (fără `source_file` ca argument obligatoriu).

- [ ] **Step 2: Rescrie testele de conversie**

`test_convert.py` devine:

```python
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
```

- [ ] **Step 3: Rulează testele, verifică că pică**

Run: `uv run pytest app/knowledge_prep/tests/test_frontmatter.py app/knowledge_prep/tests/test_convert.py -q`
Expected: FAIL (ImportError `split_frontmatter`; `convert` cu semnătura veche).

- [ ] **Step 4: Implementează `frontmatter.py`**

```python
"""Front-matter-ul documentelor din `converted/`: doar pentru lizibilitate; stadiul stă în registru (spec 2026-09-25, decizia 2)."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import yaml

_DELIMITER = "---\n"


@dataclass
class FrontMatter:
    title: str
    slug: str
    type: str
    author: str | None
    source_file: str | None = None  # numele originalului, din folderul care conține `converted/`


def slugify(text: str) -> str: ...  # neschimbat


def title_from_filename(stem: str) -> str: ...  # neschimbat


def render(fm: FrontMatter, body: str) -> str: ...  # neschimbat


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """(metadate, corp). Fără front-matter: ({}, text). Cheile necunoscute (ex. vechiul `status`) rămân în dict."""
    if not text.startswith(_DELIMITER):
        return {}, text
    meta, sep, body = text[len(_DELIMITER) :].partition(_DELIMITER)
    if not sep:
        raise ValueError("Front-matter neînchis (lipsește al doilea '---').")
    data = yaml.safe_load(meta) or {}
    if not isinstance(data, dict):
        raise ValueError("Front-matter-ul nu e un mapping YAML.")
    return data, body.removeprefix("\n")


def strip_frontmatter(text: str) -> str:
    return split_frontmatter(text)[1]
```

- [ ] **Step 5: Implementează `convert.py`**

```python
"""Conversia unui original în corp markdown (fără front-matter), prin convertorul extensiei lui.

Stadiile, căile și protecțiile (ce se poate suprascrie) stau în `registry.py`; CLI-ul e `python -m app.knowledge_prep`.
"""

from collections.abc import Mapping
from pathlib import Path

from app.knowledge_prep.converters import CONVERTERS, Converter

CONVERTED_DIR = "converted"


class ConversionError(Exception):
    pass


def convert(source: Path, *, converters: Mapping[str, Converter] = CONVERTERS) -> str:
    converter = converters.get(source.suffix.lower())
    if converter is None:
        raise ConversionError(
            f"Niciun convertor pentru '{source.suffix}' ({source.name})."
        )
    return converter(source)
```

- [ ] **Step 6: Rulează testele modulului**

Run: `uv run pytest app/knowledge_prep -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/knowledge_prep
git commit -m "refactor(knowledge_prep): drop status from front-matter, convert() returns body"
```

---

### Task 2: Modele + migrare

**Files:**
- Create: `backend/app/knowledge_prep/models.py`
- Create: `backend/app/alembic/versions/7b3f1c2a9d10_add_knowledge_registry.py`
- Modify: `backend/app/alembic/env.py`
- Create: `backend/app/knowledge_prep/tests/conftest.py`
- Test: `backend/app/knowledge_prep/tests/test_models.py`

**Interfaces:**
- Produces:
  - `DocumentType(StrEnum)`: `carte, regim, nota, insight, transcriere, conversatie`
  - `Stage(StrEnum)`: `uploaded, structured, reviewed, indexed, validated`; proprietatea `rank: int` (poziția în ordine)
  - `Document`, `Chunk`, `Question` (tabelele `document`, `chunk`, `question`, coloane ca în spec)
  - fixture pytest `session: Session` (SQLite in-memory, tabelele create)

- [ ] **Step 1: Fixture + test**

`tests/conftest.py`:

```python
from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.knowledge_prep.models import Chunk, Document, Question


@pytest.fixture
def session() -> Iterator[Session]:
    """Registrul pe SQLite in-memory: testele nu au nevoie de Postgres."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    tables = [m.__table__ for m in (Document, Chunk, Question)]  # type: ignore[attr-defined]
    SQLModel.metadata.create_all(engine, tables=tables)
    with Session(engine) as s:
        yield s
```

`tests/test_models.py`:

```python
from sqlmodel import Session

from app.knowledge_prep.models import Chunk, Document, DocumentType, Question, Stage


def test_stage_rank_follows_pipeline_order():
    assert [s.value for s in Stage] == [
        "uploaded", "structured", "reviewed", "indexed", "validated",
    ]
    assert Stage.uploaded.rank < Stage.reviewed.rank < Stage.validated.rank


def test_document_chunk_question_roundtrip(session: Session):
    doc = Document(
        slug="graul-incoltit",
        title="Grâul încolțit",
        type=DocumentType.carte,
        split=True,
        stage=Stage.structured,
        markdown_path="/k/converted/graul-incoltit.md",
    )
    session.add(doc)
    session.commit()
    chunk = Chunk(
        document_id=doc.id, position=0, heading_path=["Grâul încolțit", "Contraindicații"], text="t"
    )
    session.add(chunk)
    session.commit()
    session.add(Question(chunk_id=chunk.id, text="Cine nu are voie grâu încolțit?"))
    session.commit()
    session.expire_all()

    loaded = session.get(Document, doc.id)
    assert loaded is not None
    assert (loaded.stage, loaded.type) == (Stage.structured, DocumentType.carte)
    assert loaded.last_error is None and loaded.created_at is not None
    assert session.get(Chunk, chunk.id).heading_path == ["Grâul încolțit", "Contraindicații"]
```

- [ ] **Step 2: Verifică că pică**

Run: `uv run pytest app/knowledge_prep/tests/test_models.py -q`
Expected: FAIL (ModuleNotFoundError `app.knowledge_prep.models`).

- [ ] **Step 3: `models.py`**

```python
"""Registrul pipeline-ului de cunoștințe: documente, chunk-uri, întrebări (spec 2026-09-25, „Modelul de date”)."""

import uuid
from datetime import datetime
from enum import StrEnum, auto

import sqlalchemy as sa
from sqlmodel import Field, SQLModel

from app.models import get_datetime_utc


class DocumentType(StrEnum):
    carte = auto()
    regim = auto()
    nota = auto()
    insight = auto()
    transcriere = auto()
    conversatie = auto()


class Stage(StrEnum):
    uploaded = auto()
    structured = auto()
    reviewed = auto()
    indexed = auto()
    validated = auto()

    @property
    def rank(self) -> int:
        return list(Stage).index(self)


def _enum_column(enum: type[StrEnum]) -> sa.Column:  # VARCHAR, nu tip enum Postgres
    return sa.Column(sa.Enum(enum, native_enum=False, length=20), nullable=False)


class Document(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    slug: str = Field(unique=True, index=True, max_length=255)
    title: str = Field(max_length=500)
    author: str | None = Field(default=None, max_length=255)
    type: DocumentType = Field(sa_column=_enum_column(DocumentType))
    split: bool
    stage: Stage = Field(sa_column=_enum_column(Stage))
    original_path: str | None = Field(default=None, unique=True)  # absolut
    markdown_path: str = Field(unique=True)  # absolut; fișierul editat în IDE
    reviewed_hash: str | None = Field(default=None, max_length=64)
    indexed_hash: str | None = Field(default=None, max_length=64)
    last_error: str | None = Field(default=None, sa_type=sa.Text)
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
    )
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
    )


class Chunk(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)  # = doc_id al vectorilor-copii
    document_id: uuid.UUID = Field(foreign_key="document.id", ondelete="CASCADE", index=True)
    position: int
    heading_path: list[str] = Field(sa_type=sa.JSON)
    text: str = Field(sa_type=sa.Text)


class Question(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)  # = id-ul vectorului-copil
    chunk_id: uuid.UUID = Field(foreign_key="chunk.id", ondelete="CASCADE", index=True)
    text: str = Field(sa_type=sa.Text)
```

- [ ] **Step 4: Rulează testul**

Run: `uv run pytest app/knowledge_prep/tests/test_models.py -q`
Expected: PASS.

- [ ] **Step 5: `env.py` + migrarea**

În `app/alembic/env.py`, după `from app.models import SQLModel  # noqa`:

```python
import app.knowledge_prep.models  # noqa: F401  (tabelele registrului în SQLModel.metadata)
```

Migrarea `7b3f1c2a9d10_add_knowledge_registry.py` (`down_revision = "fe56fa70289e"`): `op.create_table` pentru `document` (coloanele din `models.py`, enum-urile ca `sa.Enum(..., native_enum=False, length=20)`, `UniqueConstraint` pe `original_path` și `markdown_path`, index unic `ix_document_slug`), `chunk` (FK `document.id` `ondelete="CASCADE"`, index `ix_chunk_document_id`), `question` (FK `chunk.id` `ondelete="CASCADE"`, index `ix_question_chunk_id`); `downgrade` le șterge în ordine inversă.

- [ ] **Step 6: Verifică migrarea offline**

Run: `uv run alembic upgrade fe56fa70289e:7b3f1c2a9d10 --sql`
Expected: `CREATE TABLE document (...)`, `chunk`, `question`, fără erori.
Cu Postgres pornit (`docker compose up -d db`): `uv run alembic upgrade head`, apoi `uv run alembic check` → „No new upgrade operations detected”.

- [ ] **Step 7: Commit**

```bash
git add backend/app/knowledge_prep/models.py backend/app/knowledge_prep/tests backend/app/alembic
git commit -m "feat(knowledge_prep): registry models and migration"
```

---

### Task 3: `registry.py` — register, resolve, review, validate, status

**Files:**
- Create: `backend/app/knowledge_prep/registry.py`
- Test: `backend/app/knowledge_prep/tests/test_registry.py`

**Interfaces:**
- Consumes: Task 1 (`split_frontmatter`, `strip_frontmatter`, `slugify`, `title_from_filename`, `CONVERTED_DIR`), Task 2 (modele, fixture `session`).
- Produces:
  - `class RegistryError(Exception)`, `class AlreadyRegistered(RegistryError)`
  - `body_hash(path: Path) -> str`
  - `default_type(path: Path) -> DocumentType`, `default_split(doc_type: DocumentType) -> bool`
  - `register(session, original: Path, *, doc_type: DocumentType | None = None, title: str | None = None, author: str | None = None, split: bool | None = None, converters: Mapping[str, Converter] = CONVERTERS) -> Document`
  - `register_structured(session, markdown: Path, *, doc_type=None, title=None, author=None, split=None) -> Document`
  - `resolve(session, target: str) -> Document`
  - `review(session, doc: Document) -> Document`
  - `validate(session, doc: Document) -> Document`
  - `hash_warnings(doc: Document) -> list[str]`
  - `record_step(session, doc) -> ContextManager[None]` — la excepție: rollback, `last_error`, commit, re-raise; la succes: `last_error=None`, `updated_at`, commit.

- [ ] **Step 1: Teste**

```python
from pathlib import Path

import pytest
from sqlmodel import Session

from app.knowledge_prep import registry
from app.knowledge_prep.frontmatter import FrontMatter, render
from app.knowledge_prep.models import DocumentType, Stage
from app.knowledge_prep.registry import AlreadyRegistered, RegistryError

CONVERTERS = {".txt": lambda p: f"conținut din {p.name}\n", ".ogg": lambda p: "transcriere\n"}


def make_file(tmp_path: Path, name: str, content: str = "brut") -> Path:
    f = tmp_path / "baza" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


def reg(session: Session, path: Path, **kw):
    return registry.register(session, path, converters=CONVERTERS, **kw)


def test_register_original_is_uploaded_with_planned_markdown_path(session, tmp_path):
    src = make_file(tmp_path, "regimuri/Regim DRI - Anemie.txt")

    doc = reg(session, src, doc_type=DocumentType.regim)

    assert (doc.slug, doc.title, doc.stage, doc.split) == (
        "regim-dri-anemie", "Regim DRI - Anemie", Stage.uploaded, False,
    )
    assert doc.original_path == str(src.resolve())
    assert doc.markdown_path == str(src.parent.resolve() / "converted" / "regim-dri-anemie.md")


def test_register_defaults_type_and_split(session, tmp_path):
    audio = reg(session, make_file(tmp_path, "digestia.ogg"))
    book = reg(session, make_file(tmp_path, "Alimente.txt"), title="Alimente vii", author="X")
    note = reg(session, make_file(tmp_path, "n.txt"), doc_type=DocumentType.nota, split=True)

    assert (audio.type, audio.split) == (DocumentType.transcriere, True)
    assert (book.type, book.split, book.slug, book.author) == (DocumentType.carte, True, "alimente-vii", "X")
    assert note.split is True


def test_register_rejects_unsupported_extension(session, tmp_path):
    with pytest.raises(RegistryError, match=".xyz"):
        reg(session, make_file(tmp_path, "a.xyz"))


def test_register_same_file_twice_is_already_registered(session, tmp_path):
    src = make_file(tmp_path, "a.txt")
    reg(session, src)
    with pytest.raises(AlreadyRegistered):
        reg(session, src)


def test_register_rejects_slug_used_by_another_file(session, tmp_path):
    reg(session, make_file(tmp_path, "unu.txt"), title="Același titlu")
    with pytest.raises(RegistryError, match="slug"):
        reg(session, make_file(tmp_path, "doi.txt"), title="Același titlu")


def test_register_structured_reads_legacy_frontmatter_and_links_original(session, tmp_path):
    original = make_file(tmp_path, "audio/Rugăciunea este intimitate.ogg")
    md = make_file(
        tmp_path,
        "audio/converted/rugaciunea-este-intimitate.md",
        "---\ntitle: Rugăciunea este intimitate\nslug: rugaciunea-este-intimitate\n"
        "type: transcriere\nauthor: null\nsource_file: Rugăciunea este intimitate.ogg\n"
        "status: reviewed\n---\n\ncorp\n",
    )

    doc = registry.register_structured(session, md, doc_type=DocumentType.nota)

    assert (doc.title, doc.type, doc.stage) == ("Rugăciunea este intimitate", DocumentType.transcriere, Stage.structured)
    assert doc.original_path == str(original.resolve())
    assert doc.markdown_path == str(md.resolve())
    assert doc.reviewed_hash is None  # vechiul `status: reviewed` nu contează


def test_register_structured_without_frontmatter_uses_options_or_filename(session, tmp_path):
    md = make_file(tmp_path, "whatsapp/extracted/2024-1/bucata_020.md", "<!-- bucata 020 -->\n\n## Nuci\n")

    doc = registry.register_structured(session, md, doc_type=DocumentType.conversatie)

    assert (doc.title, doc.slug, doc.type, doc.original_path) == (
        "bucata 020", "bucata-020", DocumentType.conversatie, None,
    )


def test_register_structured_rejects_invalid_type_in_frontmatter(session, tmp_path):
    md = make_file(tmp_path, "x.md", "---\ntitle: X\ntype: roman\n---\n\ncorp\n")
    with pytest.raises(RegistryError, match="roman"):
        registry.register_structured(session, md)


def test_resolve_by_id_slug_and_paths_in_any_form(session, tmp_path, monkeypatch):
    src = make_file(tmp_path, "audio DRI/Rugăciunea este intimitate.txt")
    doc = reg(session, src)
    monkeypatch.chdir(tmp_path / "baza")

    for target in (str(doc.id), doc.slug, str(src), "audio DRI/Rugăciunea este intimitate.txt",
                   "audio DRI/converted/rugaciunea-este-intimitate.md"):
        assert registry.resolve(session, target).id == doc.id
    with pytest.raises(RegistryError, match="negăsit"):
        registry.resolve(session, "nu-exista")


def structured_doc(session, tmp_path, body="# Titlu\n\ncorp\n"):
    fm = FrontMatter(title="Carte", slug="carte", type="carte", author=None)
    md = make_file(tmp_path, "converted/carte.md", render(fm, body))
    return registry.register_structured(session, md), md


def test_review_saves_hash_of_body_only(session, tmp_path):
    doc, md = structured_doc(session, tmp_path)

    doc = registry.review(session, doc)

    assert doc.stage == Stage.reviewed
    assert doc.reviewed_hash == registry.body_hash(md)
    md.write_text(md.read_text().replace("title: Carte", "title: Carte nouă"))
    assert registry.hash_warnings(doc) == []  # front-matter-ul nu intră în hash


def test_body_edit_after_review_warns_until_reviewed_again(session, tmp_path):
    doc, md = structured_doc(session, tmp_path)
    registry.review(session, doc)
    md.write_text(md.read_text() + "adăugat\n")

    assert registry.hash_warnings(doc) == ["modificat după review"]
    doc = registry.review(session, doc)
    assert registry.hash_warnings(doc) == []


def test_hash_warnings_after_indexing_and_missing_file(session, tmp_path):
    doc, md = structured_doc(session, tmp_path)
    doc = registry.review(session, doc)
    doc.indexed_hash = doc.reviewed_hash
    md.write_text(md.read_text() + "x\n")
    assert registry.hash_warnings(doc) == ["modificat după review", "modificat după indexare"]
    md.unlink()
    assert registry.hash_warnings(doc) == ["markdown lipsă"]


def test_review_requires_markdown(session, tmp_path):
    doc = reg(session, make_file(tmp_path, "a.txt"))
    with pytest.raises(RegistryError, match="prepare"):
        registry.review(session, doc)


def test_review_unchanged_indexed_document_keeps_stage(session, tmp_path):
    doc, _ = structured_doc(session, tmp_path)
    doc = registry.review(session, doc)
    doc.stage = Stage.indexed
    assert registry.review(session, doc).stage == Stage.indexed


def test_validate_requires_indexed(session, tmp_path):
    doc, _ = structured_doc(session, tmp_path)
    with pytest.raises(RegistryError, match="indexed"):
        registry.validate(session, doc)
    doc.stage = Stage.indexed
    assert registry.validate(session, doc).stage == Stage.validated
    assert registry.validate(session, doc).stage == Stage.validated


def test_record_step_saves_error_without_changing_stage_and_success_clears_it(session, tmp_path):
    doc, _ = structured_doc(session, tmp_path)
    with pytest.raises(RuntimeError), registry.record_step(session, doc):
        doc.stage = Stage.reviewed
        raise RuntimeError("API căzut")

    assert (doc.stage, doc.last_error) == (Stage.structured, "RuntimeError: API căzut")
    with registry.record_step(session, doc):
        pass
    assert doc.last_error is None
```

- [ ] **Step 2: Verifică că pică**

Run: `uv run pytest app/knowledge_prep/tests/test_registry.py -q`
Expected: FAIL (ModuleNotFoundError `registry`).

- [ ] **Step 3: Implementează `registry.py`**

```python
"""Registrul documentelor: tranzițiile de stadiu și regulile lor (spec 2026-09-25, „Stadii”).

Funcțiile primesc un `Session`; CLI-ul (`__main__.py`) și, mai târziu, REST-ul le cheamă la fel.
"""

import hashlib
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from sqlmodel import Session, or_, select

from app.knowledge_prep.convert import CONVERTED_DIR
from app.knowledge_prep.converters import CONVERTERS, Converter
from app.knowledge_prep.converters.audio import AUDIO_EXTENSIONS
from app.knowledge_prep.frontmatter import (
    slugify, split_frontmatter, strip_frontmatter, title_from_filename,
)
from app.knowledge_prep.models import Document, DocumentType, Stage
from app.models import get_datetime_utc

UNSPLIT_TYPES = {DocumentType.regim, DocumentType.nota, DocumentType.insight}


class RegistryError(Exception):
    pass


class AlreadyRegistered(RegistryError):
    pass


def body_hash(path: Path) -> str:
    return hashlib.sha256(strip_frontmatter(path.read_text(encoding="utf-8")).encode()).hexdigest()


def default_type(path: Path) -> DocumentType:
    return DocumentType.transcriere if path.suffix.lower() in AUDIO_EXTENSIONS else DocumentType.carte


def default_split(doc_type: DocumentType) -> bool:
    return doc_type not in UNSPLIT_TYPES


def _add(session: Session, doc: Document) -> Document:
    same_file = session.exec(select(Document).where(or_(
        Document.markdown_path == doc.markdown_path,
        Document.original_path == doc.original_path if doc.original_path else False,
    ))).first()
    if same_file:
        raise AlreadyRegistered(f"deja înregistrat ca '{same_file.slug}' ({same_file.stage})")
    taken = session.exec(select(Document).where(Document.slug == doc.slug)).first()
    if taken:
        raise RegistryError(
            f"slug '{doc.slug}' e deja folosit de {taken.original_path or taken.markdown_path}; alege alt --title."
        )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def register(session, original, *, doc_type=None, title=None, author=None, split=None, converters=CONVERTERS) -> Document:
    original = original.resolve()
    if original.suffix.lower() not in converters:
        raise RegistryError(f"Niciun convertor pentru '{original.suffix}' ({original.name}).")
    title = title or title_from_filename(original.stem)
    doc_type = doc_type or default_type(original)
    slug = slugify(title)
    return _add(session, Document(
        slug=slug, title=title, author=author, type=doc_type,
        split=default_split(doc_type) if split is None else split,
        stage=Stage.uploaded, original_path=str(original),
        markdown_path=str(original.parent / CONVERTED_DIR / f"{slug}.md"),
    ))


def register_structured(session, markdown, *, doc_type=None, title=None, author=None, split=None) -> Document:
    """Front-matter-ul (dacă există) are prioritate față de opțiuni; `status`-ul vechi e ignorat."""
    markdown = markdown.resolve()
    meta, _ = split_frontmatter(markdown.read_text(encoding="utf-8"))
    title = meta.get("title") or title or title_from_filename(markdown.stem)
    try:
        doc_type = DocumentType(meta.get("type") or doc_type or DocumentType.carte)
    except ValueError:
        raise RegistryError(f"tip necunoscut '{meta.get('type')}' în {markdown.name}")
    original = None
    if (source_file := meta.get("source_file")) and markdown.parent.name == CONVERTED_DIR:
        candidate = markdown.parent.parent / source_file
        original = str(candidate) if candidate.is_file() else None
    return _add(session, Document(
        slug=slugify(title), title=title, author=meta.get("author") or author, type=doc_type,
        split=default_split(doc_type) if split is None else split,
        stage=Stage.structured, original_path=original, markdown_path=str(markdown),
    ))


def resolve(session, target: str) -> Document:
    try:
        doc = session.get(Document, uuid.UUID(target))
    except ValueError:
        doc = None
    if doc is None:
        path = str(Path(target).resolve())
        doc = session.exec(select(Document).where(
            or_(Document.original_path == path, Document.markdown_path == path)
        )).first()
    if doc is None:
        doc = session.exec(select(Document).where(Document.slug == target)).first()
    if doc is None:
        raise RegistryError(f"document negăsit: {target}")
    return doc


@contextmanager
def record_step(session, doc) -> Iterator[None]:
    """Un pas eșuat scrie `last_error` și nu schimbă stadiul; unul reușit golește `last_error`."""
    try:
        yield
    except Exception as e:
        session.rollback()  # expiră obiectele: modificările din pas se pierd
        doc.last_error = f"{type(e).__name__}: {e}"
        doc.updated_at = get_datetime_utc()
        session.add(doc)
        session.commit()
        session.refresh(doc)
        raise
    doc.last_error = None
    doc.updated_at = get_datetime_utc()
    session.add(doc)
    session.commit()
    session.refresh(doc)


def review(session, doc) -> Document:
    if doc.stage == Stage.uploaded:
        raise RegistryError(f"{doc.slug} nu are încă markdown; rulează întâi prepare.")
    md = Path(doc.markdown_path)
    if not md.is_file():
        raise RegistryError(f"lipsește markdown-ul: {md}")
    current = body_hash(md)
    if doc.stage.rank >= Stage.reviewed.rank and current == doc.reviewed_hash:
        return doc  # nimic nou de revizuit; un document indexat rămâne indexat
    with record_step(session, doc):
        doc.reviewed_hash = current
        doc.stage = Stage.reviewed
    return doc


def validate(session, doc) -> Document:
    if doc.stage == Stage.validated:
        return doc
    if doc.stage != Stage.indexed:
        raise RegistryError(f"{doc.slug} e '{doc.stage}'; validate cere 'indexed'.")
    with record_step(session, doc):
        doc.stage = Stage.validated
    return doc


def hash_warnings(doc) -> list[str]:
    if doc.stage == Stage.uploaded:
        return []
    md = Path(doc.markdown_path)
    if not md.is_file():
        return ["markdown lipsă"]
    current = body_hash(md)
    warnings = []
    if doc.reviewed_hash and current != doc.reviewed_hash:
        warnings.append("modificat după review")
    if doc.indexed_hash and current != doc.indexed_hash:
        warnings.append("modificat după indexare")
    return warnings
```

(Adnotările complete de tip se scriu în cod; aici sunt prescurtate doar la parametrii evidenți din „Interfaces”.)

- [ ] **Step 4: Rulează testele**

Run: `uv run pytest app/knowledge_prep/tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge_prep/registry.py backend/app/knowledge_prep/tests/test_registry.py
git commit -m "feat(knowledge_prep): registry with register/review/validate/status rules"
```

---

### Task 4: `prepare` în registru

**Files:**
- Modify: `backend/app/knowledge_prep/registry.py`
- Test: `backend/app/knowledge_prep/tests/test_registry.py`

**Interfaces:**
- Consumes: `convert(source, *, converters) -> str`, `FrontMatter`, `render`, `record_step`.
- Produces: `prepare(session, doc: Document, *, force: bool = False, converters: Mapping[str, Converter] = CONVERTERS) -> Path` (calea markdown-ului scris).

- [ ] **Step 1: Teste**

```python
from app.knowledge_prep.frontmatter import split_frontmatter


def test_prepare_writes_markdown_with_frontmatter_and_moves_to_structured(session, tmp_path):
    src = make_file(tmp_path, "regimuri/Regim DRI - Anemie.txt")
    doc = reg(session, src, doc_type=DocumentType.regim, author="DRI")

    out = registry.prepare(session, doc, converters=CONVERTERS)

    assert out == src.parent.resolve() / "converted" / "regim-dri-anemie.md"
    meta, body = split_frontmatter(out.read_text())
    assert meta == {"title": "Regim DRI - Anemie", "slug": "regim-dri-anemie", "type": "regim",
                    "author": "DRI", "source_file": "Regim DRI - Anemie.txt"}
    assert body == "conținut din Regim DRI - Anemie.txt\n"
    assert (doc.stage, doc.last_error) == (Stage.structured, None)
    assert src.read_text() == "brut"


def test_prepare_without_original_fails_clearly(session, tmp_path):
    doc, _ = structured_doc(session, tmp_path)
    doc.original_path = None
    with pytest.raises(RegistryError, match="original"):
        registry.prepare(session, doc, converters=CONVERTERS, force=True)


def test_prepare_does_not_overwrite_existing_markdown_without_force(session, tmp_path):
    doc = reg(session, make_file(tmp_path, "a.txt"))
    out = registry.prepare(session, doc, converters=CONVERTERS)
    out.write_text(out.read_text() + "editat de om\n")

    with pytest.raises(RegistryError, match="--force"):
        registry.prepare(session, doc, converters=CONVERTERS)
    assert out.read_text().endswith("editat de om\n")


def test_prepare_force_on_reviewed_returns_to_structured(session, tmp_path):
    doc = reg(session, make_file(tmp_path, "a.txt"))
    registry.prepare(session, doc, converters=CONVERTERS)
    doc = registry.review(session, doc)
    doc.indexed_hash = doc.reviewed_hash

    with pytest.raises(RegistryError, match="--force"):
        registry.prepare(session, doc, converters=CONVERTERS)
    registry.prepare(session, doc, converters=CONVERTERS, force=True)

    assert (doc.stage, doc.reviewed_hash) == (Stage.structured, None)
    assert doc.indexed_hash is not None  # chunk-urile rămân până la următorul index


def test_prepare_failure_records_error_and_keeps_stage(session, tmp_path):
    doc = reg(session, make_file(tmp_path, "a.txt"))

    def broken(_p):
        raise RuntimeError("API căzut")

    with pytest.raises(RuntimeError):
        registry.prepare(session, doc, converters={".txt": broken})
    assert (doc.stage, doc.last_error) == (Stage.uploaded, "RuntimeError: API căzut")
    assert not Path(doc.markdown_path).exists()
```

- [ ] **Step 2: Verifică că pică**

Run: `uv run pytest app/knowledge_prep/tests/test_registry.py -q -k prepare`
Expected: FAIL (`registry.prepare` nu există).

- [ ] **Step 3: Implementează**

```python
def prepare(session, doc, *, force=False, converters=CONVERTERS) -> Path:
    """Original → `<folderul originalului>/converted/<slug>.md` → `structured`."""
    if doc.original_path is None:
        raise RegistryError(
            f"{doc.slug} nu are original (a intrat direct ca markdown); nu are ce pregăti."
        )
    md = Path(doc.markdown_path)
    if not force and (doc.stage.rank >= Stage.reviewed.rank or md.exists()):
        raise RegistryError(
            f"{doc.slug} e '{doc.stage}' și are deja {md.name}; folosește --force ca să-l suprascrii."
        )
    original = Path(doc.original_path)
    with record_step(session, doc):
        body = convert(original, converters=converters)
        fm = FrontMatter(title=doc.title, slug=doc.slug, type=doc.type.value,
                         author=doc.author, source_file=original.name)
        md.parent.mkdir(exist_ok=True)
        md.write_text(render(fm, body), encoding="utf-8")
        doc.stage = Stage.structured
        doc.reviewed_hash = None
    return md
```

- [ ] **Step 4: Rulează testele**

Run: `uv run pytest app/knowledge_prep -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge_prep
git commit -m "feat(knowledge_prep): prepare converts originals through the registry"
```

---

### Task 5: CLI `python -m app.knowledge_prep`

**Files:**
- Create: `backend/app/knowledge_prep/__main__.py`
- Test: `backend/app/knowledge_prep/tests/test_cli.py`

**Interfaces:**
- Consumes: tot `registry.py`.
- Produces: `main(argv: Sequence[str] | None = None, *, session: Session | None = None, converters: Mapping[str, Converter] = CONVERTERS) -> int` (0 = totul ok, 1 = cel puțin o eroare). Fără `session`, deschide `Session(app.core.db.engine)`.

Subcomenzi:
- `register <cale> [--structured] [--type T] [--title] [--author] [--split/--no-split]` — folder: originale = fișierele cu extensie suportată, fără cele din `converted/` (restul „sărit (format nesuportat)”); `--structured` = toate `*.md`. `AlreadyRegistered` → „sărit”, nu eroare. `--title` interzis pe folder.
- `prepare <țintă> [--force]` — țintă folder = documentele cu `original_path` sub folder; fără `--force` doar cele `uploaded` (restul „sărit”).
- `review <țintă>`, `validate <țintă>`
- `status [<țintă>]` — tabel `slug | tip | stadiu | avertismente | eroare` pe stdout.

- [ ] **Step 1: Teste**

```python
import logging
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.knowledge_prep.__main__ import main
from app.knowledge_prep.models import Document, Stage

CONVERTERS = {".txt": lambda p: f"conținut din {p.name}\n", ".ogg": lambda p: "transcriere\n"}


def make_file(tmp_path: Path, name: str, content: str = "brut") -> Path:
    f = tmp_path / "baza" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


def run(session: Session, *argv: str) -> int:
    return main(list(argv), session=session, converters=CONVERTERS)


def stages(session: Session) -> dict[str, Stage]:
    return {d.slug: d.stage for d in session.exec(select(Document))}


def test_register_folder_skips_converted_and_unsupported_and_is_idempotent(session, tmp_path, caplog):
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


def test_register_structured_folder(session, tmp_path):
    make_file(tmp_path, "extracted/bucata_1.md", "## Nuci\n")
    make_file(tmp_path, "extracted/bucata_2.md", "## Gât\n")

    assert run(session, "register", "--structured", "--type", "conversatie", str(tmp_path / "baza")) == 0

    assert stages(session) == {"bucata-1": Stage.structured, "bucata-2": Stage.structured}


def test_register_rejects_title_for_folder(session, tmp_path):
    make_file(tmp_path, "a.txt")
    with pytest.raises(SystemExit):
        run(session, "register", str(tmp_path / "baza"), "--title", "X")


def test_full_flow_prepare_review_status(session, tmp_path, capsys):
    src = make_file(tmp_path, "a.txt")
    run(session, "register", str(src), "--no-split")

    assert run(session, "prepare", "a") == 0
    assert run(session, "review", str(src)) == 0
    md = tmp_path / "baza" / "converted" / "a.md"
    md.write_text(md.read_text() + "editat\n")
    capsys.readouterr()

    assert run(session, "status") == 0
    out = capsys.readouterr().out
    assert "a" in out and "reviewed" in out and "modificat după review" in out


def test_prepare_folder_only_uploaded_without_force(session, tmp_path, caplog):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "sub/b.txt")
    folder = str(tmp_path / "baza")
    run(session, "register", folder)
    run(session, "prepare", str(tmp_path / "baza" / "a.txt"))

    with caplog.at_level(logging.INFO):
        assert run(session, "prepare", folder) == 0

    assert stages(session) == {"a": Stage.structured, "b": Stage.structured}
    assert "sărit" in caplog.text


def test_prepare_folder_continues_after_a_failure(session, tmp_path):
    make_file(tmp_path, "a.txt")
    make_file(tmp_path, "b.ogg")
    run(session, "register", str(tmp_path / "baza"))

    def broken(_p):
        raise RuntimeError("API căzut")

    code = main(["prepare", str(tmp_path / "baza")], session=session,
                converters={".txt": broken, ".ogg": CONVERTERS[".ogg"]})

    assert code == 1
    assert stages(session) == {"a": Stage.uploaded, "b": Stage.structured}


def test_single_target_rule_violation_returns_1(session, tmp_path, caplog):
    run(session, "register", str(make_file(tmp_path, "a.txt")))
    assert run(session, "validate", "a") == 1
    assert "indexed" in caplog.text
    assert run(session, "review", "nu-exista") == 1
```

- [ ] **Step 2: Verifică că pică**

Run: `uv run pytest app/knowledge_prep/tests/test_cli.py -q`
Expected: FAIL (ModuleNotFoundError `app.knowledge_prep.__main__`).

- [ ] **Step 3: Implementează `__main__.py`**

Structura:

```python
"""CLI-ul pipeline-ului de cunoștințe: register → prepare → review → (index) → validate, plus status.

Rulare din backend/:
    uv run python -m app.knowledge_prep <comandă> <țintă> [opțiuni]
<țintă> = id, slug sau cale (a originalului ori a markdown-ului). Spec: docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md
"""

def build_parser() -> argparse.ArgumentParser: ...   # subparsers dest="command", required=True
def _register(args, session, converters, parser) -> int: ...
def _prepare(args, session, converters) -> int: ...
def _single(fn, args, session) -> int: ...          # review / validate
def _status(args, session) -> int: ...
def _run_each(items, action, describe) -> int:       # buclă tolerantă: loghează, numără eșecurile
def main(argv=None, *, session=None, converters=CONVERTERS) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if session is None:
        from app.core.db import engine  # doar la rulare reală: cere .env + Postgres
        with Session(engine) as s:
            return _dispatch(args, s, converters)
    return _dispatch(args, session, converters)
```

`status` scrie pe `sys.stdout` (ruff interzice `print`), coloane aliniate cu lățimea maximă a fiecărei coloane. `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 4: Rulează testele + lint**

Run: `uv run pytest app/knowledge_prep -q && uv run ruff check app/knowledge_prep && uv run ruff format --check app/knowledge_prep && uv run ty check app/knowledge_prep`
Expected: PASS, fără erori.

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge_prep
git commit -m "feat(knowledge_prep): CLI with register/prepare/review/validate/status"
```

---

### Task 6: Docs + verificare reală

**Files:**
- Modify: `/home/alex/projects/RAG/AGENTS.md` (secțiunile „Comenzi” și „Corpus / knowledge”)
- Modify: `calea_de_autovindecare/docs/taskuri.md` (status faza 1) — doar dacă userul confirmă

- [ ] **Step 1: AGENTS.md**

Înlocuiește comanda `convert` cu:

```bash
# Registrul de cunoștințe (knowledge_prep): register → prepare → review → index → validate
uv run python -m app.knowledge_prep register "<fișier|folder>" [--structured] [--type regim] [--title "..."] [--author "..."] [--split/--no-split]
uv run python -m app.knowledge_prep prepare <id|slug|cale|folder> [--force]
uv run python -m app.knowledge_prep review|validate <id|slug|cale>
uv run python -m app.knowledge_prep status [<țintă>]
```

și bullet-ul despre `status: draft/reviewed` cu: stadiul stă în tabelul `document` (Postgres, vizibil în Adminer :8080); front-matter-ul e doar informativ; `review` salvează hash-ul corpului.

- [ ] **Step 2: Verificare reală (necesită Postgres pornit)**

```bash
docker compose up -d db          # din rădăcina full-stack
uv run alembic upgrade head && uv run alembic check
uv run python -m app.knowledge_prep register --structured "../../calea_de_autovindecare/knowledge/calea-de-vindecare/audio DRI/converted"
uv run python -m app.knowledge_prep status
```

Expected: documentele audio apar `structured`, cu originalul `.ogg` legat.
