# Restructurarea la `prepare` (faza 2) — plan de implementare

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `prepare` corectează și structurează markdown-ul convertit (diacritice, caractere stricate, titluri `#`-`####`, liste) cu Gemini, pe ferestre de ~8k tokeni, verificând că textul nu se pierde și nu se adaugă.

**Architecture:** Modul nou `app/knowledge_prep/restructure.py`, fără DB: `restructure(body, title) -> str`. Taie corpul în ferestre la granițe de paragraf, trimite fiecare fereastră cu calea de titluri de până atunci, verifică rezultatul cu `pdf.missing_fraction` în ambele sensuri și reîncearcă o dată. `registry.prepare` îl cheamă după `convert`, implicit pentru tot ce nu e PDF; `--restructure/--no-restructure` suprascrie. Restructuratorul se injectează ca și convertoarele, ca testele să nu cheme Gemini.

**Tech Stack:** Python 3.14, google-genai (prin `converters/gemini.py::generate`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` (secțiunile „Restructurarea”, „Comenzi”, faza 2).

## Global Constraints

- Comenzile se rulează din `full-stack-fastapi-template/backend/` (`uv run ...`).
- **Când:** implicit pentru `.txt`, `.md` și audio; nu pentru PDF. `--restructure/--no-restructure` suprascrie.
- **Model:** `GEMINI_MODEL` din `.env`, prin `converters/gemini.py::generate` (fără thinking, retry cu `stop_after_attempt(5)`). Niciun `@retry` nou.
- **Promptul** e cel din spike (`calea_de_autovindecare/.claude/skills/structurare-carte/prompt.md`), fără nota specifică cărții „Dăruim lumină și iubire”.
- **Ferestre** de ~8k tokeni (`WINDOW_CHARS = 24_000`), tăiate la granițe de paragraf; fiecare apel primește calea de titluri de la finalul ferestrei anterioare.
- **Verificare:** `missing_fraction` lipsă și adăugat, prag 1% pe fiecare, per fereastră; la depășire o reîncercare; apoi încercarea mai bună, precedată de `<!-- VERIFICĂ: fereastra N, lipsă X%, adăugat Y% -->`.
- Pas eșuat → `last_error`, stadiul nu se schimbă (deja asigurat de `record_step`).
- `rag_simple` rămâne neatins. Mesajele pentru user sunt în română.

## Review Focus

1. **Gemini întoarce textul între ```` ```markdown ```` ... ```` ``` ````** (s-a văzut la `ruleaza.sh`): gardurile se scot, nu ajung în `.md` și nu strică verificarea.
2. **Un paragraf mai lung decât o fereastră** (transcriere audio fără rânduri goale): devine o fereastră singură, nu se pierde și nu intră în buclă.
3. **Răspuns gol la ambele încercări:** fereastra rămâne cu textul nerestructurat, precedat de `VERIFICĂ`, nu se pierde și `prepare` nu cade.
4. **Testele și CLI-ul nu cheamă Gemini** când se dau convertoare fake: fiecare apel `prepare`/`main` din teste primește și un restructurator fake.
5. **`ş`/`ţ` cu sedilă în răspuns** se normalizează la virgulă dedesubt (`diacritics.normalize`), ca la PDF.

---

## Structura fișierelor

| Fișier | Rol |
|---|---|
| Create `backend/app/knowledge_prep/restructure.py` | promptul, ferestrele, calea de titluri, verificarea, `restructure()` |
| Create `backend/app/knowledge_prep/tests/test_restructure.py` | teste cu `FakeClient` |
| Modify `backend/app/knowledge_prep/registry.py` | `prepare(..., restructure=None, restructurer=...)` |
| Modify `backend/app/knowledge_prep/__main__.py` | `prepare --restructure/--no-restructure`; `main(..., restructurer=...)` |
| Modify `backend/app/knowledge_prep/converters/text.py` | docstring: structurarea e în `restructure.py` |
| Modify `backend/app/knowledge_prep/tests/test_registry.py`, `test_cli.py` | restructurator fake; teste noi pentru implicit/override |
| Modify `RAG/AGENTS.md` (în afara repo-ului) | `prepare [--restructure/--no-restructure]` |

---

### Task 1: `restructure.py`

**Files:**
- Create: `backend/app/knowledge_prep/restructure.py`
- Test: `backend/app/knowledge_prep/tests/test_restructure.py`

**Interfaces:**
- Consumes: `converters.gemini.client_and_model, generate`; `converters.pdf.missing_fraction`; `converters.diacritics.normalize`.
- Produces: `restructure(body: str, title: str, client: genai.Client | None = None, model: str | None = None) -> str`; `split_windows(text: str, max_chars: int | None = None) -> list[str]` (implicit `WINDOW_CHARS`, citit la apel); `heading_path(markdown: str) -> str`; `Restructurer = Callable[[str, str], str]`.

- [ ] **Step 1: Testele**

```python
from types import SimpleNamespace
from typing import cast

from google import genai

from app.knowledge_prep.restructure import heading_path, restructure, split_windows

PARA = "Grâul încolțit curăță sângele și dă putere organismului slăbit"


class FakeClient:
    """Întoarce pe rând textele din `replies`; `None` = răspuns gol."""

    def __init__(self, replies: list[str | None]):
        self.messages: list[str] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.messages.append(contents[1])
        return SimpleNamespace(text=self._replies.pop(0))


def run(body: str, client: FakeClient, title: str = "Grâul încolțit") -> str:
    return restructure(body, title, cast(genai.Client, client), "gemini-2.5-flash")


def test_split_windows_cuts_at_paragraphs_and_keeps_long_paragraph_alone():
    paras = ["a" * 40, "b" * 40, "c" * 150, "d" * 40]
    windows = split_windows("\n\n".join(paras) + "\n", max_chars=100)
    assert windows == [f"{'a' * 40}\n\n{'b' * 40}", "c" * 150, "d" * 40]


def test_heading_path_keeps_last_header_per_level():
    md = "# A\n\n## B\n\n### C\n\ntext\n\n## D\n\ntext #nu\n"
    assert heading_path(md) == "# A > ## D"
    assert heading_path("fără titluri\n") == ""


def test_restructure_passes_heading_path_to_next_window(monkeypatch):
    monkeypatch.setattr("app.knowledge_prep.restructure.WINDOW_CHARS", 80)
    first, second = f"{PARA} unu", f"{PARA} doi"
    client = FakeClient([f"# Grâul încolțit\n\n## Beneficii\n\n{first}", second])

    out = run(f"{first}\n\n{second}\n", client)

    assert out == f"# Grâul încolțit\n\n## Beneficii\n\n{first}\n\n{second}\n"
    assert "bucata 1 din 2" in client.messages[0]
    assert "„Grâul încolțit”" in client.messages[0]
    assert "# Grâul încolțit > ## Beneficii" in client.messages[1]
    assert f"<text>\n{second}\n</text>" in client.messages[1]


def test_restructure_strips_code_fences_and_normalizes_cedilla():
    client = FakeClient([f"```markdown\n# Titlu\n\n{PARA} şi ţara\n```"])
    assert run(f"{PARA} si tara\n", client) == f"# Titlu\n\n{PARA} și țara\n"


def test_restructure_retries_window_with_missing_text_then_accepts():
    body = f"{PARA}\n\nAl doilea paragraf important despre germeni si vitamine\n"
    client = FakeClient([PARA, body])
    assert run(body, client) == body
    assert len(client.messages) == 2


def test_restructure_marks_window_still_off_after_retry_keeping_better_attempt():
    body = f"{PARA}\n\nAl doilea paragraf important despre germeni si vitamine\n"
    added = f"{body}\n\nO frază inventată de model, care nu era în original deloc.\n"
    client = FakeClient([PARA, added])
    out = run(body, client)
    assert out.startswith("<!-- VERIFICĂ: fereastra 1, lipsă 0.0%, adăugat ")
    assert "O frază inventată" in out


def test_restructure_keeps_original_window_when_gemini_returns_nothing():
    client = FakeClient([None, "  "])
    out = run(f"{PARA}\n", client)
    assert out == (
        "<!-- VERIFICĂ: fereastra 1, Gemini nu a întors nimic; "
        f"mai jos e textul nerestructurat -->\n\n{PARA}\n"
    )
```

- [ ] **Step 2:** `uv run pytest app/knowledge_prep/tests/test_restructure.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementarea**

```python
"""Corectarea și structurarea markdown-ului convertit cu Gemini (spec 2026-09-25, „Restructurarea”).

Corpul se taie în ferestre de ~8k tokeni, la granițe de paragraf. Fiecare fereastră primește
calea de titluri de la finalul celei anterioare, ca ierarhia să continue în tot documentul.
Rezultatul se verifică pe litere față de fereastră (`pdf.missing_fraction`, în ambele sensuri):
peste prag se reîncearcă o dată, apoi se păstrează încercarea mai bună, marcată `VERIFICĂ`.
"""

import logging
import re
from collections.abc import Callable

from google import genai

from app.knowledge_prep.converters.diacritics import normalize
from app.knowledge_prep.converters.gemini import client_and_model, generate
from app.knowledge_prep.converters.pdf import missing_fraction

logger = logging.getLogger(__name__)

# (corp, titlu) → corp restructurat; `prepare` îl primește injectat, ca testele să nu cheme Gemini.
Restructurer = Callable[[str, str], str]

# Promptul din spike-ul din 2026-09-25 (calea_de_autovindecare/.claude/skills/structurare-carte/prompt.md).
PROMPT = """..."""  # textul complet: vezi Step 3a

WINDOW_CHARS = 24_000  # ~8k tokeni de text românesc
MAX_DIFF = 0.01  # lipsă sau adăugat, per fereastră

_HEADER = re.compile(r"^(#{1,4}) \S.*$", re.M)
_FENCE = re.compile(r"\A```[\w-]*\n(.*?)\n?```\s*\Z", re.S)


def split_windows(text: str, max_chars: int | None = None) -> list[str]:
    """Paragrafe consecutive până la `max_chars`; un paragraf mai lung rămâne singur."""
    max_chars = max_chars or WINDOW_CHARS
    windows: list[str] = []
    current = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if current and len(current) + 2 + len(para) > max_chars:
            windows.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        windows.append(current)
    return windows


def heading_path(markdown: str) -> str:
    """Ultimul titlu de pe fiecare nivel, ex. `# A > ## B > ### C`."""
    path: dict[int, str] = {}
    for m in _HEADER.finditer(markdown):
        level = len(m[1])
        path = {k: v for k, v in path.items() if k < level}
        path[level] = m[0].strip()
    return " > ".join(path[k] for k in sorted(path))


def _message(window: str, i: int, total: int, title: str, done: str) -> str:
    if i == 1:
        where = (
            f"Aceasta e bucata 1 din {total}, începutul documentului „{title}”. "
            "Pune singurul `#` cu titlul documentului."
        )
    else:
        where = (
            f"Aceasta e bucata {i} din {total} a documentului „{title}”. "
            f"Calea de titluri de până acum: {heading_path(done) or '(niciuna)'}. "
            "Continuă ierarhia de acolo."
        )
    return f"{where}\n\n<text>\n{window}\n</text>"


def _clean(text: str) -> str:
    text = text.strip()
    if fenced := _FENCE.match(text):
        text = fenced[1].strip()
    return normalize(text)


def _restructure_window(
    client: genai.Client, model: str, window: str, message: str, n: int
) -> str:
    best, best_diff = None, (1.0, 1.0)
    for attempt in (1, 2):
        reply = generate(client, model, [PROMPT, message])
        if not reply or not reply.strip():
            logger.warning("  fereastra %d: răspuns gol (încercarea %d)", n, attempt)
            continue
        text = _clean(reply)
        diff = (missing_fraction(window, text), missing_fraction(text, window))
        if max(diff) <= MAX_DIFF:
            return text
        logger.warning(
            "  fereastra %d: lipsă %.1f%%, adăugat %.1f%% (încercarea %d)",
            n, diff[0] * 100, diff[1] * 100, attempt,
        )
        if max(diff) < max(best_diff):
            best, best_diff = text, diff

    if best is None:
        return (
            f"<!-- VERIFICĂ: fereastra {n}, Gemini nu a întors nimic; "
            f"mai jos e textul nerestructurat -->\n\n{window}"
        )
    return (
        f"<!-- VERIFICĂ: fereastra {n}, lipsă {best_diff[0]:.1%}, "
        f"adăugat {best_diff[1]:.1%} -->\n\n{best}"
    )


def restructure(
    body: str,
    title: str,
    client: genai.Client | None = None,
    model: str | None = None,
) -> str:
    client, model = client_and_model(client, model)
    windows = split_windows(body)
    parts: list[str] = []
    for i, window in enumerate(windows, 1):
        logger.info("restructurez „%s”: fereastra %d/%d cu %s ...", title, i, len(windows), model)
        message = _message(window, i, len(windows), title, "\n\n".join(parts))
        parts.append(_restructure_window(client, model, window, message, i))
    return "\n\n".join(parts) + "\n"
```

- [ ] **Step 3a: `PROMPT`** = textul din `prompt.md`, fără paragraful „NOTĂ (specifică cărții …)”, cu prima frază generalizată: „Primești o bucată dintr-un text în limba română (carte, articol sau transcriere), obținut prin conversie automată (PDF, OCR, transcriere audio sau text copiat).” și diacriticele corectate în regula „NU MODIFICA” („poți”, „Dacă”).

- [ ] **Step 4:** `uv run pytest app/knowledge_prep/tests/test_restructure.py -q` → PASS; `uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep`.

- [ ] **Step 5: Commit** `feat(knowledge_prep): restructure markdown with Gemini in verified windows`

---

### Task 2: `prepare` cu restructurare

**Files:**
- Modify: `backend/app/knowledge_prep/registry.py` (`prepare`)
- Modify: `backend/app/knowledge_prep/__main__.py` (`build_parser`, `_prepare`, `_dispatch`, `main`)
- Modify: `backend/app/knowledge_prep/converters/text.py` (docstring)
- Test: `backend/app/knowledge_prep/tests/test_registry.py`, `backend/app/knowledge_prep/tests/test_cli.py`
- Modify: `RAG/AGENTS.md`

**Interfaces:**
- Consumes: `restructure.restructure`, `restructure.Restructurer`, `converters.pdf.PDF_EXTENSIONS`.
- Produces: `registry.prepare(session, doc, *, force=False, restructure: bool | None = None, converters=CONVERTERS, restructurer: Restructurer = restructure.restructure) -> Path`; `main(argv, *, session=None, converters=CONVERTERS, restructurer=restructure.restructure) -> int`.

- [ ] **Step 1: Teste în `test_registry.py`** — `fake_restructure(body, title) -> f"# {title}\n\n{body}"`; helperul local `prep(session, doc, **kw)` cheamă `registry.prepare(..., converters=CONVERTERS, restructurer=fake_restructure, **kw)` și înlocuiește toate apelurile existente (testul de scriere așteaptă acum corpul `# Regim DRI - Anemie\n\nconținut din …`). Teste noi:

```python
def test_prepare_restructures_text_and_audio_but_not_pdf_by_default(
    session: Session, tmp_path: Path
):
    converters = {**CONVERTERS, ".pdf": fake_converter}
    txt = reg(session, make_file(tmp_path, "a.txt"))
    pdf = registry.register(session, make_file(tmp_path, "b.pdf"), converters=converters)

    txt_md = registry.prepare(session, txt, converters=converters, restructurer=fake_restructure)
    pdf_md = registry.prepare(session, pdf, converters=converters, restructurer=fake_restructure)

    assert split_frontmatter(txt_md.read_text())[1].startswith("# a\n\n")
    assert split_frontmatter(pdf_md.read_text())[1] == "conținut din b.pdf\n"


def test_prepare_restructure_flag_overrides_default(session: Session, tmp_path: Path):
    converters = {**CONVERTERS, ".pdf": fake_converter}
    txt = reg(session, make_file(tmp_path, "a.txt"))
    pdf = registry.register(session, make_file(tmp_path, "b.pdf"), converters=converters)

    txt_md = registry.prepare(
        session, txt, converters=converters, restructurer=fake_restructure, restructure=False
    )
    pdf_md = registry.prepare(
        session, pdf, converters=converters, restructurer=fake_restructure, restructure=True
    )

    assert split_frontmatter(txt_md.read_text())[1] == "conținut din a.txt\n"
    assert split_frontmatter(pdf_md.read_text())[1].startswith("# b\n\n")


def test_prepare_restructure_failure_records_error_and_writes_nothing(
    session: Session, tmp_path: Path
):
    doc = reg(session, make_file(tmp_path, "a.txt"))

    def broken(_body: str, _title: str) -> str:
        raise RuntimeError("Gemini căzut")

    with pytest.raises(RuntimeError):
        registry.prepare(session, doc, converters=CONVERTERS, restructurer=broken)
    assert (doc.stage, doc.last_error) == (Stage.uploaded, "RuntimeError: Gemini căzut")
    assert not Path(doc.markdown_path).exists()
```

- [ ] **Step 2: Teste în `test_cli.py`** — `run()` trimite `restructurer=fake_restructure`; și celelalte apeluri directe `main(...)` din fișier. Test nou:

```python
def test_prepare_no_restructure_flag(session: Session, tmp_path: Path):
    make_file(tmp_path, "a.txt")
    assert run(session, "register", str(tmp_path / "baza")) == 0
    assert run(session, "prepare", "a", "--no-restructure") == 0
    md = tmp_path / "baza" / "converted" / "a.md"
    assert md.read_text().endswith("\nconținut din a.txt\n")
```

- [ ] **Step 3:** rulează testele → FAIL (`unexpected keyword argument 'restructurer'`).

- [ ] **Step 4: `registry.prepare`**

```python
    original = Path(doc.original_path)
    if restructure is None:
        restructure = original.suffix.lower() not in PDF_EXTENSIONS  # promptul PDF structurează deja
    with record_step(session, doc):
        body = convert(original, converters=converters)
        if restructure:
            body = restructurer(body, doc.title)
        ...
```

Parametrul `restructurer: Restructurer = restructure_markdown`, cu `from app.knowledge_prep.restructure import Restructurer, restructure as restructure_markdown` (numele `restructure` e parametrul bool). Docstring-ul lui `prepare` menționează restructurarea.

- [ ] **Step 5: CLI** — în `build_parser`, pe `prep`:

```python
    prep.add_argument(
        "--restructure",
        action=argparse.BooleanOptionalAction,
        help="corectură + structurare cu Gemini; implicit da, în afară de PDF",
    )
```

`_prepare(args, session, converters, restructurer)` trimite `restructure=args.restructure, restructurer=restructurer`; `main(..., restructurer: Restructurer = restructure_markdown)` îl dă prin `_dispatch`.

- [ ] **Step 6:** `text.py` docstring: „Corectura și structurarea (titluri, diacritice lipsă) le face `restructure.py`, la `prepare`.” `AGENTS.md`: `prepare <id|slug|cale|folder> [--force] [--restructure/--no-restructure]`.

- [ ] **Step 7:** `uv run pytest app/knowledge_prep -q` → PASS; ruff + ty curate.

- [ ] **Step 8: Commit** `feat(knowledge_prep): prepare restructures non-PDF sources (--restructure/--no-restructure)`

---

### Task 3: Verificare reală

- [ ] `Graul_incoltit_Valeriu_Popa.md` (carti/converted, ~10k caractere, fără diacritice): se copiază ca original `.md` într-un folder temporar sub `carti/` **doar dacă userul confirmă**; altfel se rulează `restructure()` direct din Python pe corpul lui și se compară cu rezultatul spike-ului (diacritice, liste, titluri, marcaje `VERIFICĂ`, procentele din log).
- [ ] Raport către user: timpul, procentele lipsă/adăugat, numărul de marcaje.
