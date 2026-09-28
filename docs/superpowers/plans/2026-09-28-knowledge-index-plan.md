# Indexarea (faza 3) — plan de implementare

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `index <țintă> [--no-questions] [--force]` împarte markdown-ul revizuit în chunk-uri, generează 1-5 întrebări per chunk cu Gemini, scrie `chunk`/`question` în Postgres și vectorii-copii (`content` + `question`) în `knowledge_vectors`, apoi trece documentul în `indexed`.

**Architecture:** Patru module noi, fiecare testabil singur: `chunking.py` (markdown → `ChunkDraft`-uri, fără I/O), `questions.py` (Gemini cu output structurat → listă de întrebări), `vectors.py` (`VectorIndex`: protocol + implementarea `PgVectorIndex` peste `PGVectorStore`), `index.py` (regulile de stadiu + orchestrarea, cu `record_step` din registru). Tot ce cheamă API-uri (întrebări, embeddings) se face în memorie **înainte** de orice scriere; apoi tabelele relaționale se înlocuiesc într-o tranzacție, apoi vectorii documentului se șterg după `document_id` și se adaugă cei noi, apoi `stage=indexed`. Întrebările și vectorii se injectează (`questioner=`, `vectors=`) ca testele să nu cheme Gemini, OpenAI sau pgvector.

**Tech Stack:** Python 3.14, SQLModel 0.0.42, langchain-postgres 0.0.18 (`PGEngine`, `PGVectorStore`, `Column`), langchain-text-splitters 1.1.2 (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`), tiktoken 0.14 (`cl100k_base`), langchain-openai 1.6.2 (`OpenAIEmbeddings`), google-genai 2.8.0 (`response_schema` Pydantic → `response.parsed`), tenacity, pytest. **Nicio dependență nouă**: toate sunt deja în `backend/pyproject.toml` (tiktoken vine prin langchain-openai și e folosit deja de `rag_simple`).

**Spec:** `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` (secțiunile „Decizii”, „Modelul de date” — `chunk`, `question`, „Tabelul de vectori” —, „Stadii”, „Comenzi”, „Indexarea”, „Testare”; faza 3).

## Global Constraints

- Comenzile se rulează din `full-stack-fastapi-template/backend/` (`uv run ...`). Testele: `uv run pytest app/knowledge_prep` (nu `backend/tests/`, care pică la import). Lint: `uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep`.
- `index` cere `stage >= reviewed` și hash-ul curent egal cu `reviewed_hash`. Altfel refuză cu mesajul „markdown modificat după review; rulează din nou `review`”.
- Dacă documentul era `validated`, comanda cere `--force` și avertizează că întrebările editate manual se pierd.
- Un pas eșuat scrie `last_error` și nu schimbă stadiul; un pas reușit golește `last_error` (`registry.record_step`). Regulile încălcate ridică `RegistryError` fără să atingă registrul (convenția din faza 1).
- Chunking `split=false`: un singur chunk cu tot corpul; `heading_path` = titlul documentului; avertisment peste 8000 de tokeni.
- Chunking `split=true`: titluri `#`-`####`, headerele rămân în text; `MIN_CHUNK_TOKENS = 80` (se unesc cu următoarea secțiune din același părinte); `MAX_CHUNK_TOKENS = 1000` (se împart cu `RecursiveCharacterTextSplitter`, overlap 150, păstrând `heading_path`); marcajele `<!-- VERIFICĂ -->` se scot.
- Întrebări: un apel Gemini per chunk, output structurat (listă de str), primind titlul documentului, `heading_path` și textul; 1-5 întrebări în română, la care chunk-ul răspunde complet, formulate ca de un om, care numesc subiectul; pentru `split=false`, acoperă tot documentul. Model: `GEMINI_MODEL` din `.env`, fără thinking.
- `knowledge_vectors`: `PGVectorStore`, `openai:text-embedding-3-large`, 3072 dimensiuni; `metadata_columns`: `doc_id` (chunk), `document_id`, `kind` (`content | question`). Id-ul vectorului = `chunk.id` pentru `content`, `question.id` pentru `question`.
- Textul embedded pentru `content` = `" > ".join(heading_path)` + rând gol + `chunk.text`; pentru `question` = doar textul întrebării.
- Orice apel LLM/API nou: `@retry(..., stop=stop_after_attempt(N), before_sleep=..., reraise=True)`. Niciodată `@retry(wait=...)` fără `stop=`.
- `rag_simple` rămâne neatins (nici nu se importă din el). Fără modele locale grele (WSL cu RAM puțin). Nu se editează `.env`.
- Mesajele pentru user sunt în română, ca în codul existent.

## Review Focus

1. **`#` care nu e titlu:** linii `# ...` în blocuri de cod (```` ``` ````) și `#hashtag` fără spațiu nu deschid secțiuni noi. Test: `test_code_fence_and_hashtag_are_not_headers` (Task 1).
2. **Reindexarea după editare + `review`:** chunk-urile, întrebările și vectorii vechi ai documentului dispar, iar ale altor documente rămân neatinse. Teste: `test_reindex_replaces_old_rows_and_vectors_only_for_that_document` (Task 4, cu fake) și `test_replace_document_swaps_only_that_documents_vectors` (Task 3, pe Postgres).
3. **Eșec după commit-ul relațional** (scrierea în pgvector cade): stadiul rămâne cel vechi, `last_error` e scris, iar o nouă rulare `index` reface totul curat. Test: `test_vector_write_failure_keeps_stage_and_rerun_recovers` (Task 4).
4. **Gemini întoarce JSON invalid, întrebări goale, duplicate sau mai mult de 5:** chunk-ul se indexează oricum (cu vectorul `content`), lista se curăță, iar lipsa întrebărilor apare ca avertisment. Teste: `test_cleans_empty_duplicate_and_extra_questions`, `test_unparseable_reply_returns_no_questions` (Task 2), `test_chunk_without_questions_is_reported` (Task 4).
5. **Markdown care după scoaterea marcajelor `VERIFICĂ` e gol:** `index` eșuează cu `last_error` clar, fără să scrie nimic și fără să șteargă indexul vechi. Teste: `test_empty_body_raises` (Task 1), `test_empty_markdown_records_error` (Task 4).

---

## Decizii de implementare (interpretări ale spec-ului)

1. **Nicio migrare Alembic nouă pentru `chunk`/`question`.** Tabelele există deja, din migrarea fazei 1 (`7b3f1c2a9d10_add_knowledge_registry.py`, aplicată pe baza de dev), cu exact coloanele din spec. Faza 3 doar le populează.
2. **`knowledge_vectors` îl creează `PGEngine.init_vectorstore_table` la prima folosire, nu Alembic** (ca `rag_simple.vectorstore.ensure_table`). Motive: schema e a lui langchain_postgres (`langchain_id`, `content`, `embedding vector(N)`, `langchain_metadata`), iar `PGVectorStore.create_sync` o validează; testul de integrare creează tabele de test cu același cod, deci calea de producție e testată. `alembic/env.py` exclude tabelele `knowledge_vectors*` din autogenerate, ca `rag_simple_chunks`. Fără index ANN: pgvector indexează `vector` doar până la 2000 de dimensiuni, iar la corpusul de acum (mii de rânduri) scanarea exactă e suficientă.
3. **Împărțirea pe titluri e un parser propriu (~30 de rânduri), nu `MarkdownHeaderTextSplitter`.** Verificat pe langchain-text-splitters 1.1.2: cu `strip_headers=False`, splitter-ul lipește liniile cu `"  \n"` și **scoate rândurile goale** (`"# A\n\nText."` → `"# A  \nText."`). Asta strică `chunk.text` (textul din DB nu mai e cel din markdown, editabil prin REST în faza 5) și îi ia lui `RecursiveCharacterTextSplitter` separatorul `"\n\n"`, deci secțiunile mari s-ar tăia în mijlocul paragrafelor. Parserul propriu păstrează textul exact și ignoră `#` din blocurile de cod. `RecursiveCharacterTextSplitter` rămâne pentru secțiunile mari, cum cere spec-ul.
4. **„Același părinte” la unirea secțiunilor mici:** secțiunea mică `S` se unește cu următoarea `N` dacă `N` e soră sau descendentă a lui `S` (`N.heading_path` începe cu `S.heading_path[:-1]`). Calea chunk-ului unit e prefixul comun, cu o excepție: dacă `S` conține doar titluri (ex. `# Carte` urmat direct de `## Capitol`), chunk-ul ia calea lui `N`. Textul dinaintea primului titlu (și orice chunk fără titluri) primește `heading_path = [titlul documentului]`.
5. **Ordinea scrierii diferă puțin de spec (pasul 4), în favoarea recuperării:** embeddings-urile se calculează **înainte** de tranzacția relațională (un eșec OpenAI nu mai lasă chunk-uri fără vectori), iar vectorii vechi se șterg **după** commit-ul relațional, cu `PGVectorStore.delete(filter={"document_id": ...})` în loc de lista de id-uri. Filtrul prinde și vectorii rămași de la o rulare întreruptă și nu depinde de chunk-urile care tocmai au fost șterse. Regula din spec rămâne: eșec după commit-ul relațional → stadiul anterior + `last_error`; o nouă rulare reface tot.
6. **Tokenii se numără cu tiktoken `cl100k_base`** (tokenizer-ul lui `text-embedding-3-large`). Prima rulare pe o mașină nouă descarcă fișierul encoding-ului (cache în `$TIKTOKEN_CACHE_DIR` sau `/tmp/data-gym-cache`); `rag_simple` are deja aceeași dependență.
7. **Un chunk pentru care Gemini nu întoarce întrebări valide** se indexează doar cu vectorul `content` și apare ca avertisment. Un singur chunk ratat nu oprește indexarea unei cărți întregi (întrebările se pot adăuga prin REST în faza 5).
8. **Re-`index` pe un document `indexed` (nevalidat)** e permis fără `--force` (spec-ul cere `--force` doar pentru `validated`), dar loghează câte întrebări existente se înlocuiesc.
9. **`status` nu se schimbă.** Afișează deja stadiul, „modificat după indexare” (compară cu `indexed_hash`) și `last_error`; `index` doar completează `indexed_hash`.
10. **Embeddings:** `OpenAIEmbeddings(model="text-embedding-3-large", api_key=settings.OPENAI_API_KEY)` direct (fără `os.environ` ca în `rag_simple/config.py`). Loturile peste 300k tokeni pe request le împarte deja langchain-openai (`MAX_TOKENS_PER_REQUEST`).

## Structura fișierelor

| Fișier | Rol |
|---|---|
| Create `backend/app/knowledge_prep/chunking.py` | `ChunkDraft`, `chunk_markdown`, `count_tokens`, `embedding_text`, constantele de mărime |
| Create `backend/app/knowledge_prep/questions.py` | `PROMPT`, `GeneratedQuestions`, `generate_questions`, `QuestionGenerator` |
| Modify `backend/app/knowledge_prep/converters/gemini.py` | `generate_structured` (retry ca `generate`); `_log_retry` → `log_retry` (refolosit) |
| Create `backend/app/knowledge_prep/vectors.py` | `VectorEntry`, `VectorIndex` (Protocol), `PgVectorIndex` |
| Modify `backend/app/alembic/env.py` | exclude `knowledge_vectors*` din autogenerate |
| Create `backend/app/knowledge_prep/index.py` | `check_indexable`, `index_document`, `IndexReport` |
| Modify `backend/app/knowledge_prep/__main__.py` | subcomanda `index`; `main(..., vectors=, questioner=)` |
| Create `backend/app/knowledge_prep/tests/test_chunking.py`, `test_questions.py`, `test_vectors_pg.py`, `test_index.py`, `index_fakes.py` | teste + fake-uri comune |
| Modify `backend/app/knowledge_prep/tests/test_cli.py` | `run()` cu fake-uri pentru index; teste noi |
| Modify `backend/app/knowledge_prep/README.md`, spec-ul, `RAG/AGENTS.md`, `calea_de_autovindecare/docs/taskuri.md` | docs (Task 6) |

---

### Task 1: `chunking.py`

**Files:**
- Create: `backend/app/knowledge_prep/chunking.py`
- Test: `backend/app/knowledge_prep/tests/test_chunking.py`

**Interfaces:**
- Consumes: nimic din task-urile anterioare (`tiktoken`, `langchain_text_splitters`).
- Produces: `ChunkDraft(heading_path: list[str], text: str)` (dataclass); `chunk_markdown(body: str, title: str, split: bool) -> tuple[list[ChunkDraft], list[str]]` (chunk-uri, avertismente; `ValueError("markdown gol: ...")` pe corp gol); `count_tokens(text: str) -> int`; `embedding_text(heading_path: list[str], text: str) -> str`; constantele `MIN_CHUNK_TOKENS = 80`, `MAX_CHUNK_TOKENS = 1000`, `CHUNK_OVERLAP_TOKENS = 150`, `UNSPLIT_WARN_TOKENS = 8000`.

- [ ] **Step 1: Testele**

```python
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
    chunks, _ = chunk_markdown(f"# Carte\n\n## Mare\n\n{big}\n", "Titlu doc", split=True)
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
```

- [ ] **Step 2:** `uv run pytest app/knowledge_prep/tests/test_chunking.py -q` → FAIL (`ModuleNotFoundError: app.knowledge_prep.chunking`).

- [ ] **Step 3: Implementarea**

```python
"""Împărțirea markdown-ului revizuit în chunk-uri (spec 2026-09-25, „Indexarea”, pasul 2).

`split=false`: tot corpul e un singur chunk, cu `heading_path = [titlul documentului]`.
`split=true`: o secțiune la fiecare titlu `#`-`####` (headerele rămân în text); secțiunile mici
se lipesc de următoarea din același părinte, cele mari se taie cu `RecursiveCharacterTextSplitter`.

Împărțirea pe titluri e proprie, nu `MarkdownHeaderTextSplitter`: acela scoate rândurile goale
și lipește liniile cu "  \\n", deci textul din DB n-ar mai fi cel din markdown, iar secțiunile
mari nu s-ar mai putea tăia la granițe de paragraf.
"""

import re
from dataclasses import dataclass
from functools import cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_TOKENS = 80
MAX_CHUNK_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 150
UNSPLIT_WARN_TOKENS = 8000  # limita lui text-embedding-3-large e 8191
ENCODING = "cl100k_base"  # tokenizer-ul lui text-embedding-3-large

_HEADER = re.compile(r"^(#{1,4})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_VERIFY = re.compile(r"[ \t]*<!--\s*VERIFICĂ.*?-->[ \t]*", re.S)


@dataclass
class ChunkDraft:
    heading_path: list[str]
    text: str


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def embedding_text(heading_path: list[str], text: str) -> str:
    """Textul embedded pentru vectorul `content`: calea de titluri, rând gol, chunk-ul."""
    return f"{' > '.join(heading_path)}\n\n{text}"


def _strip_verify_markers(body: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", _VERIFY.sub("", body))


def _split_sections(body: str) -> list[ChunkDraft]:
    """O secțiune la fiecare titlu `#`-`####` din afara blocurilor de cod; textul rămâne neatins."""
    sections: list[ChunkDraft] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(ChunkDraft([h for _, h in path], text))
        lines.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        header = None if in_fence else _HEADER.match(line)
        if header:
            flush()
            level = len(header[1])
            path = [(lvl, h) for lvl, h in path if lvl < level] + [
                (level, header[2].strip())
            ]
        lines.append(line)
    flush()
    return sections


def _headers_only(text: str) -> bool:
    return all(_HEADER.match(line) for line in text.splitlines() if line.strip())


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return prefix


def _merge_small(sections: list[ChunkDraft]) -> list[ChunkDraft]:
    """Sub MIN_CHUNK_TOKENS, o secțiune se lipește de următoarea dacă aceea e soră sau descendentă."""
    merged: list[ChunkDraft] = []
    pending: ChunkDraft | None = None
    for section in sections:
        if pending is not None:
            parent = pending.heading_path[:-1]
            if section.heading_path[: len(parent)] == parent:
                path = (
                    section.heading_path
                    if _headers_only(pending.text)
                    else _common_prefix(pending.heading_path, section.heading_path)
                )
                section = ChunkDraft(path, f"{pending.text}\n\n{section.text}")
            else:
                merged.append(pending)
            pending = None
        if count_tokens(section.text) < MIN_CHUNK_TOKENS:
            pending = section
        else:
            merged.append(section)
    if pending is not None:
        merged.append(pending)
    return merged


def _split_large(sections: list[ChunkDraft]) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    out: list[ChunkDraft] = []
    for section in sections:
        if count_tokens(section.text) <= MAX_CHUNK_TOKENS:
            out.append(section)
        else:
            out.extend(
                ChunkDraft(list(section.heading_path), piece)
                for piece in splitter.split_text(section.text)
            )
    return out


def chunk_markdown(
    body: str, title: str, split: bool
) -> tuple[list[ChunkDraft], list[str]]:
    """Corpul markdown (fără front-matter) → (chunk-uri în ordine, avertismente)."""
    text = _strip_verify_markers(body).strip()
    if not text:
        raise ValueError("markdown gol: nimic de indexat")
    if not split:
        warnings = []
        tokens = count_tokens(text)
        if tokens > UNSPLIT_WARN_TOKENS:
            warnings.append(
                f"documentul întreg are {tokens} tokeni (> {UNSPLIT_WARN_TOKENS}); "
                "vectorul `content` va fi o medie pe bucăți (imprecis), întrebările nu sunt afectate"
            )
        return [ChunkDraft([title], text)], warnings
    chunks = _split_large(_merge_small(_split_sections(text)))
    for chunk in chunks:
        if not chunk.heading_path:
            chunk.heading_path = [title]
    return chunks, []
```

- [ ] **Step 4:** `uv run pytest app/knowledge_prep/tests/test_chunking.py -q` → PASS; `uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep` curate. Dacă `test_fixture_sizes` pică, ajustează numărul de repetări din `MID` (nu constantele).

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge_prep/chunking.py backend/app/knowledge_prep/tests/test_chunking.py
git commit -m "feat(knowledge_prep): chunk reviewed markdown by headers with size limits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `questions.py` + `gemini.generate_structured`

**Files:**
- Modify: `backend/app/knowledge_prep/converters/gemini.py`
- Create: `backend/app/knowledge_prep/questions.py`
- Test: `backend/app/knowledge_prep/tests/test_questions.py`

**Interfaces:**
- Consumes: `converters.gemini.client_and_model`, `thinking_config`.
- Produces: `gemini.generate_structured[M: BaseModel](client: genai.Client, model: str, contents: list, schema: type[M]) -> M | None`; `gemini.log_retry(retry_state)` (fostul `_log_retry`); `questions.GeneratedQuestions` (Pydantic, `questions: list[str]`); `questions.generate_questions(title: str, heading_path: list[str], text: str, whole_document: bool, client: genai.Client | None = None, model: str | None = None) -> list[str]`; `questions.QuestionGenerator = Callable[[str, list[str], str, bool], list[str]]`; `questions.MAX_QUESTIONS = 5`.

- [ ] **Step 1: Testele**

```python
from types import SimpleNamespace
from typing import cast

from google import genai

from app.knowledge_prep.questions import GeneratedQuestions, generate_questions


class FakeClient:
    """Întoarce pe rând `parsed` din `replies`; `None` = JSON care nu s-a validat."""

    def __init__(self, replies: list[GeneratedQuestions | None]):
        self.requests: list[dict] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(parsed=self._replies.pop(0))


def ask(client: FakeClient, *, whole: bool = False) -> list[str]:
    return generate_questions(
        "Grâul încolțit",
        ["Grâul încolțit", "Contraindicații"],
        "Text chunk.",
        whole,
        client=cast(genai.Client, client),
        model="gemini-2.5-flash",
    )


def test_sends_title_heading_path_text_and_json_schema():
    client = FakeClient(
        [GeneratedQuestions(questions=["Cine nu are voie să consume grâu încolțit?"])]
    )
    assert ask(client) == ["Cine nu are voie să consume grâu încolțit?"]
    request = client.requests[0]
    message = request["contents"][1]
    assert "„Grâul încolțit”" in message
    assert "Grâul încolțit > Contraindicații" in message
    assert "<text>\nText chunk.\n</text>" in message
    assert "documentul întreg" not in message
    config = request["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is GeneratedQuestions
    assert config.thinking_config.thinking_budget == 0


def test_whole_document_asks_to_cover_everything():
    client = FakeClient([GeneratedQuestions(questions=["Ce conține regimul?"])])
    ask(client, whole=True)
    assert "documentul întreg" in client.requests[0]["contents"][1]


def test_cleans_empty_duplicate_and_extra_questions():
    raw = [
        "  Ce este grâul încolțit? ",
        "",
        "ce este grâul  încolțit?",
        "Q2?",
        "Q3?",
        "Q4?",
        "Q5?",
        "Q6?",
    ]
    client = FakeClient([GeneratedQuestions(questions=raw)])
    assert ask(client) == ["Ce este grâul încolțit?", "Q2?", "Q3?", "Q4?", "Q5?"]


def test_unparseable_reply_returns_no_questions():
    assert ask(FakeClient([None])) == []
```

- [ ] **Step 2:** `uv run pytest app/knowledge_prep/tests/test_questions.py -q` → FAIL (`ModuleNotFoundError: app.knowledge_prep.questions`).

- [ ] **Step 3: `converters/gemini.py`** — redenumește `_log_retry` în `log_retry` (și în decoratorul lui `generate`), actualizează docstring-ul modulului („audio, pdf, restructurare, întrebări”) și adaugă:

```python
from pydantic import BaseModel


@retry(
    retry=retry_if_exception_type(errors.APIError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=log_retry,
    reraise=True,
)
def generate_structured[M: BaseModel](
    client: genai.Client, model: str, contents: list, schema: type[M]
) -> M | None:
    """Răspuns JSON validat pe `schema` (`response.parsed`); `None` dacă nu se validează."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=thinking_config(model),
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = response.parsed
    return parsed if isinstance(parsed, schema) else None
```

(google-genai 2.8.0: cu `response_schema` = clasă Pydantic, SDK-ul face `schema.model_validate_json(text)` și pune `None` în `parsed` la `ValidationError`/`JSONDecodeError`, fără excepție — de aici `M | None`.)

- [ ] **Step 4: `questions.py`**

```python
"""Întrebările la care răspunde un chunk, generate cu Gemini (spec 2026-09-25, „Indexarea”, pasul 3).

Fiecare întrebare devine un vector-copil (`kind=question`) al chunk-ului: o întrebare a
utilizatorului formulată altfel decât textul găsește totuși chunk-ul (multi-vector retrieval).
"""

import logging
from collections.abc import Callable

from google import genai
from pydantic import BaseModel, Field

from app.knowledge_prep.converters.gemini import client_and_model, generate_structured

logger = logging.getLogger(__name__)

# (titlul documentului, heading_path, textul chunk-ului, e documentul întreg?) → întrebări.
# `index` îl primește injectat, ca testele să nu cheme Gemini.
QuestionGenerator = Callable[[str, list[str], str, bool], list[str]]

MAX_QUESTIONS = 5

PROMPT = """Primești un fragment dintr-un text în limba română din corpusul „Calea de Autovindecare” (cărți, regimuri alimentare, note și transcrieri despre vindecarea naturală).

Scrie între 1 și 5 întrebări la care fragmentul răspunde complet. Reguli:
1. Întrebările sunt în limba română, formulate cum ar întreba un om care caută informația, nu copiate din text.
2. Fiecare întrebare se înțelege singură, fără context: numește explicit subiectul (ex. „grâul încolțit”, „regimul pentru anemie”), niciodată „acesta”, „textul”, „autorul” sau „fragmentul”.
3. Răspunsul complet la fiecare întrebare se găsește în fragment. Nu pune întrebări la care fragmentul răspunde doar parțial.
4. Întrebările sunt diferite între ele și acoperă ideile principale. Mai bine mai puține și bune decât multe și slabe.
5. Nu adăuga informații care nu sunt în fragment.
"""


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(
        description="1-5 întrebări în limba română, la care fragmentul răspunde complet"
    )


def _message(title: str, heading_path: list[str], text: str, whole_document: bool) -> str:
    scope = (
        "Fragmentul e documentul întreg: întrebările trebuie să acopere tot documentul, "
        "nu doar începutul.\n"
        if whole_document
        else ""
    )
    return (
        f"Documentul: „{title}”\nSecțiunea: {' > '.join(heading_path)}\n{scope}"
        f"\n<text>\n{text}\n</text>"
    )


def _clean(questions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        q = " ".join(q.split())
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out[:MAX_QUESTIONS]


def generate_questions(
    title: str,
    heading_path: list[str],
    text: str,
    whole_document: bool,
    client: genai.Client | None = None,
    model: str | None = None,
) -> list[str]:
    client, model = client_and_model(client, model)
    reply = generate_structured(
        client,
        model,
        [PROMPT, _message(title, heading_path, text, whole_document)],
        GeneratedQuestions,
    )
    if reply is None:
        logger.warning("  Gemini nu a întors întrebări valide pentru „%s”", " > ".join(heading_path))
        return []
    return _clean(reply.questions)
```

- [ ] **Step 5:** `uv run pytest app/knowledge_prep -q` → PASS (inclusiv `test_pdf`, `test_audio`, `test_restructure` după redenumirea lui `log_retry`); ruff + ty curate.

- [ ] **Step 6: Commit**

```bash
git add backend/app/knowledge_prep/converters/gemini.py backend/app/knowledge_prep/questions.py backend/app/knowledge_prep/tests/test_questions.py
git commit -m "feat(knowledge_prep): generate questions per chunk with Gemini structured output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `vectors.py` — `knowledge_vectors` peste `PGVectorStore`

**Files:**
- Create: `backend/app/knowledge_prep/vectors.py`
- Modify: `backend/app/alembic/env.py` (`include_object`)
- Test: `backend/app/knowledge_prep/tests/test_vectors_pg.py` (integrare, pe Postgres cu pgvector)

**Interfaces:**
- Consumes: `converters.gemini.log_retry` (Task 2).
- Produces: `VectorEntry(id: uuid.UUID, doc_id: uuid.UUID, document_id: uuid.UUID, kind: Literal["content", "question"], text: str)` (frozen dataclass); `VectorIndex` (Protocol) cu `embed(texts: list[str]) -> list[list[float]]` și `replace_document(document_id: uuid.UUID, entries: Sequence[VectorEntry], embeddings: Sequence[list[float]]) -> None`; `PgVectorIndex(url: str, embeddings: Embeddings, *, table_name: str = TABLE_NAME, vector_size: int = EMBEDDING_DIMENSIONS)` cu atributele `store: PGVectorStore`, `table_name: str` și `PgVectorIndex.from_settings()`; constantele `TABLE_NAME = "knowledge_vectors"`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS = 3072`, `METADATA_COLUMNS`.

**Cum rulează testul de integrare:** baza de dev din `.env` (`docker compose up -d db`, imaginea `pgvector/pgvector:pg18`, extensia `vector` e instalată). Testul creează un tabel `knowledge_vectors_test_<hex>` cu vectori de 8 dimensiuni și `DeterministicFakeEmbedding` (fără OpenAI) și îl șterge la final. Dacă baza nu răspunde (sau `.env` lipsește), modulul e **sărit** (`skipif`), așa că `uv run pytest app/knowledge_prep` rămâne verde și fără Postgres.

- [ ] **Step 1: Testele**

```python
"""Integrare pe Postgres cu pgvector (spec, „Testare”): ștergerea și refacerea vectorilor unui document.

Se sare dacă baza din `.env` nu răspunde (`docker compose up -d db` din rădăcina full-stack).
"""

import uuid
from collections.abc import Iterator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, text

from app.knowledge_prep.vectors import PgVectorIndex, VectorEntry


def _pg_url() -> str | None:
    try:
        from app.core.config import settings

        url = str(settings.DATABASE_URL)
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
        return url
    except Exception:
        return None


PG_URL = _pg_url()
pytestmark = pytest.mark.skipif(
    PG_URL is None, reason="Postgres cu pgvector indisponibil (docker compose up -d db)"
)
SIZE = 8


def make_index(table: str) -> PgVectorIndex:
    assert PG_URL is not None
    return PgVectorIndex(
        PG_URL, DeterministicFakeEmbedding(size=SIZE), table_name=table, vector_size=SIZE
    )


@pytest.fixture
def pg() -> Iterator[PgVectorIndex]:
    assert PG_URL is not None
    table = f"knowledge_vectors_test_{uuid.uuid4().hex[:8]}"
    yield make_index(table)
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    engine.dispose()


def entries_for(document_id: uuid.UUID, n_chunks: int = 2) -> list[VectorEntry]:
    out = []
    for i in range(n_chunks):
        chunk_id = uuid.uuid4()
        out.append(VectorEntry(chunk_id, chunk_id, document_id, "content", f"chunk {i}"))
        out.append(VectorEntry(uuid.uuid4(), chunk_id, document_id, "question", f"întrebarea {i}?"))
    return out


def put(pg: PgVectorIndex, document_id: uuid.UUID, entries: list[VectorEntry]) -> None:
    pg.replace_document(document_id, entries, pg.embed([e.text for e in entries]))


def stored(pg: PgVectorIndex, entries: list[VectorEntry]) -> dict[str, dict]:
    return {str(d.id): d.metadata for d in pg.store.get_by_ids([str(e.id) for e in entries])}


def test_replace_document_swaps_only_that_documents_vectors(pg: PgVectorIndex):
    a, b = uuid.uuid4(), uuid.uuid4()
    old_a, only_b = entries_for(a), entries_for(b)
    put(pg, a, old_a)
    put(pg, b, only_b)

    new_a = entries_for(a, n_chunks=3)
    put(pg, a, new_a)

    assert stored(pg, old_a) == {}
    assert len(stored(pg, new_a)) == 6
    b_rows = stored(pg, only_b)
    assert len(b_rows) == 4
    question = only_b[1]
    meta = b_rows[str(question.id)]
    assert (str(meta["doc_id"]), str(meta["document_id"]), meta["kind"]) == (
        str(question.doc_id),
        str(b),
        "question",
    )


def test_kind_filter_works_and_existing_table_is_reused(pg: PgVectorIndex):
    a = uuid.uuid4()
    put(pg, a, entries_for(a))
    docs = pg.store.similarity_search("chunk", k=10, filter={"kind": "content"})
    assert len(docs) == 2
    assert {d.metadata["kind"] for d in docs} == {"content"}

    again = make_index(pg.table_name)  # tabelul există: nu se recreează, datele rămân
    assert len(again.store.similarity_search("chunk", k=10)) == 4


def test_replace_document_with_no_entries_only_deletes(pg: PgVectorIndex):
    a = uuid.uuid4()
    entries = entries_for(a)
    put(pg, a, entries)
    pg.replace_document(a, [], [])
    assert stored(pg, entries) == {}
```

- [ ] **Step 2:** `uv run pytest app/knowledge_prep/tests/test_vectors_pg.py -v` → FAIL (`ModuleNotFoundError: app.knowledge_prep.vectors`). Cu `docker compose up -d db` pornit, altfel testele ar fi doar sărite.

- [ ] **Step 3: `vectors.py`**

```python
"""Tabelul `knowledge_vectors`: vectorii-copii ai chunk-urilor (spec 2026-09-25, „Tabelul de vectori”).

Un rând per vector: `content` (id = chunk.id) sau `question` (id = question.id). Coloanele
filtrabile: `doc_id` (chunk-ul părinte, cheia lui MultiVectorRetriever din faza 4),
`document_id` și `kind`.

Tabelul îl creează `PGEngine.init_vectorstore_table` la prima folosire (ca în `rag_simple`), nu
Alembic: schema e a lui langchain_postgres, iar `alembic/env.py` îl exclude din autogenerate.
Fără index ANN: pgvector indexează `vector` doar până la 2000 de dimensiuni (aici 3072), iar la
câteva mii de rânduri scanarea exactă e suficientă.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import openai
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from pydantic import SecretStr
from sqlalchemy import create_engine, inspect
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.knowledge_prep.converters.gemini import log_retry

TABLE_NAME = "knowledge_vectors"
EMBEDDING_MODEL = "text-embedding-3-large"  # același model ca rag_simple
EMBEDDING_DIMENSIONS = 3072
METADATA_COLUMNS = ["doc_id", "document_id", "kind"]


@dataclass(frozen=True)
class VectorEntry:
    id: uuid.UUID  # chunk.id pentru content, question.id pentru question
    doc_id: uuid.UUID  # chunk-ul părinte
    document_id: uuid.UUID
    kind: Literal["content", "question"]
    text: str  # textul embedded


class VectorIndex(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def replace_document(
        self,
        document_id: uuid.UUID,
        entries: Sequence[VectorEntry],
        embeddings: Sequence[list[float]],
    ) -> None: ...


def _metadata_columns() -> list[Column]:
    # obiecte noi la fiecare apel: init_vectorstore_table le modifică numele pe loc
    return [
        Column("doc_id", "UUID", nullable=False),
        Column("document_id", "UUID", nullable=False),
        Column("kind", "VARCHAR(20)", nullable=False),
    ]


def _table_exists(url: str, table_name: str) -> bool:
    engine = create_engine(url)
    try:
        return inspect(engine).has_table(table_name)
    finally:
        engine.dispose()


class PgVectorIndex:
    def __init__(
        self,
        url: str,
        embeddings: Embeddings,
        *,
        table_name: str = TABLE_NAME,
        vector_size: int = EMBEDDING_DIMENSIONS,
    ):
        self.table_name = table_name
        self._embeddings = embeddings
        engine = PGEngine.from_connection_string(url=url)
        # init_vectorstore_table face CREATE TABLE simplu: doar dacă nu există deja
        if not _table_exists(url, table_name):
            engine.init_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
                metadata_columns=_metadata_columns(),
            )
        self.store = PGVectorStore.create_sync(
            engine=engine,
            embedding_service=embeddings,
            table_name=table_name,
            metadata_columns=METADATA_COLUMNS,
        )

    @classmethod
    def from_settings(cls) -> "PgVectorIndex":
        """Baza și cheia OpenAI din `.env`."""
        from app.core.config import settings

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, api_key=SecretStr(settings.OPENAI_API_KEY)
        )
        return cls(str(settings.DATABASE_URL), embeddings)

    @retry(
        retry=retry_if_exception_type(openai.APIError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(3),
        before_sleep=log_retry,
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts) if texts else []

    def replace_document(
        self,
        document_id: uuid.UUID,
        entries: Sequence[VectorEntry],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Șterge toți vectorii documentului (după `document_id`, deci și resturi de la o rulare întreruptă), apoi îi adaugă pe cei noi."""
        self.store.delete(filter={"document_id": str(document_id)})
        if not entries:
            return
        self.store.add_embeddings(
            texts=[e.text for e in entries],
            embeddings=list(embeddings),
            metadatas=[
                {"doc_id": str(e.doc_id), "document_id": str(e.document_id), "kind": e.kind}
                for e in entries
            ],
            ids=[str(e.id) for e in entries],
        )
```

- [ ] **Step 4: `alembic/env.py`** — tabelele vectoriale (și cele de test rămase după un test întrerupt) nu intră în autogenerate:

```python
# Tabele gestionate în afara Alembic (create de LangChain direct); nu le include în autogenerate
_ALEMBIC_EXCLUDE_TABLES = {"rag_simple_chunks"}
_ALEMBIC_EXCLUDE_PREFIXES = ("knowledge_vectors",)  # knowledge_prep/vectors.py (+ tabelele de test)


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and (
        name in _ALEMBIC_EXCLUDE_TABLES or name.startswith(_ALEMBIC_EXCLUDE_PREFIXES)
    ):
        return False
    return True
```

- [ ] **Step 5:** `uv run pytest app/knowledge_prep/tests/test_vectors_pg.py -v` → 3 PASS (cu `db` pornit). Oprește temporar baza (`docker compose stop db`) sau rulează cu `DATABASE_URL` invalid doar dacă vrei să vezi `SKIPPED`; nu e obligatoriu. Apoi `uv run pytest app/knowledge_prep -q` → PASS; ruff + ty curate. Verifică și că nu a rămas niciun tabel de test: `uv run python -c "from sqlalchemy import inspect; from app.core.db import engine; print([t for t in inspect(engine).get_table_names() if t.startswith('knowledge_vectors')])"` → `[]`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/knowledge_prep/vectors.py backend/app/knowledge_prep/tests/test_vectors_pg.py backend/app/alembic/env.py
git commit -m "feat(knowledge_prep): knowledge_vectors store with per-document replace

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `index.py` — regulile și orchestrarea

**Files:**
- Create: `backend/app/knowledge_prep/index.py`
- Create: `backend/app/knowledge_prep/tests/index_fakes.py`
- Test: `backend/app/knowledge_prep/tests/test_index.py`

**Interfaces:**
- Consumes: `chunking.chunk_markdown`, `chunking.embedding_text` (Task 1); `questions.QuestionGenerator` (Task 2); `vectors.VectorEntry`, `vectors.VectorIndex` (Task 3); `registry.RegistryError`, `registry.body_hash`, `registry.record_step`; `frontmatter.strip_frontmatter`; `models.Chunk`, `Question`, `Document`, `Stage`.
- Produces: `IndexReport(chunks: int, questions: int, warnings: list[str])`; `check_indexable(doc: Document, *, force: bool) -> None` (ridică `RegistryError`); `index_document(session: Session, doc: Document, *, vectors: VectorIndex, questioner: QuestionGenerator | None, force: bool = False) -> IndexReport` (`questioner=None` = `--no-questions`). În teste: `index_fakes.FakeVectors(fail: str | None = None)` (`fail` ∈ `"embed"`, `"write"`; `.rows: dict[uuid.UUID, VectorEntry]`) și `index_fakes.FakeQuestions(per_chunk: int = 2)` (`.calls`).

- [ ] **Step 1: Fake-urile comune (`tests/index_fakes.py`)**

```python
"""Fake-uri pentru indexare: vectori în memorie și întrebări deterministe (fără OpenAI, Gemini, pgvector)."""

import uuid
from collections.abc import Sequence

from app.knowledge_prep.vectors import VectorEntry


class FakeVectors:
    """`fail="embed"`: cade la embeddings; `fail="write"`: șterge vectorii vechi, apoi cade."""

    def __init__(self, fail: str | None = None):
        self.rows: dict[uuid.UUID, VectorEntry] = {}
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail == "embed":
            raise RuntimeError("OpenAI căzut")
        return [[float(len(t))] for t in texts]

    def replace_document(
        self,
        document_id: uuid.UUID,
        entries: Sequence[VectorEntry],
        embeddings: Sequence[list[float]],
    ) -> None:
        assert len(entries) == len(embeddings)
        self.rows = {k: e for k, e in self.rows.items() if e.document_id != document_id}
        if self.fail == "write":
            raise RuntimeError("pgvector căzut")
        self.rows.update({e.id: e for e in entries})

    def of(self, document_id: uuid.UUID, kind: str | None = None) -> list[VectorEntry]:
        return [
            e
            for e in self.rows.values()
            if e.document_id == document_id and (kind is None or e.kind == kind)
        ]


class FakeQuestions:
    def __init__(self, per_chunk: int = 2):
        self.per_chunk = per_chunk
        self.calls: list[tuple[str, list[str], str, bool]] = []

    def __call__(
        self, title: str, heading_path: list[str], text: str, whole_document: bool
    ) -> list[str]:
        self.calls.append((title, list(heading_path), text, whole_document))
        return [
            f"Întrebarea {i} despre {heading_path[-1]}?"
            for i in range(1, self.per_chunk + 1)
        ]
```

- [ ] **Step 2: Testele (`tests/test_index.py`)**

```python
from pathlib import Path

import pytest
from sqlmodel import Session, col, select

from app.knowledge_prep import registry
from app.knowledge_prep.frontmatter import FrontMatter, render, slugify
from app.knowledge_prep.index import index_document
from app.knowledge_prep.models import Chunk, Document, Question, Stage
from app.knowledge_prep.registry import RegistryError
from app.knowledge_prep.tests.index_fakes import FakeQuestions, FakeVectors

MID = " ".join(["Grâul încolțit dă putere organismului slăbit."] * 10)
BOOK = f"# Grâul încolțit\n\n## Beneficii\n\n{MID}\n\n## Contraindicații\n\n{MID}\n"


def make_doc(
    session: Session,
    tmp_path: Path,
    body: str = BOOK,
    *,
    title: str = "Grâul încolțit",
    doc_type: str = "carte",
    review: bool = True,
) -> Document:
    md = tmp_path / f"{slugify(title)}.md"
    fm = FrontMatter(title=title, slug=slugify(title), type=doc_type, author=None)
    md.write_text(render(fm, body), encoding="utf-8")
    doc = registry.register_structured(session, md)
    return registry.review(session, doc) if review else doc


def chunks_of(session: Session, doc: Document) -> list[Chunk]:
    return list(
        session.exec(
            select(Chunk).where(Chunk.document_id == doc.id).order_by(col(Chunk.position))
        )
    )


def questions_of(session: Session, doc: Document) -> list[Question]:
    return list(
        session.exec(
            select(Question)
            .join(Chunk, col(Question.chunk_id) == col(Chunk.id))
            .where(Chunk.document_id == doc.id)
        )
    )


def edit_and_review(session: Session, doc: Document, extra: str) -> None:
    md = Path(doc.markdown_path)
    md.write_text(md.read_text(encoding="utf-8") + extra, encoding="utf-8")
    registry.review(session, doc)


def test_index_writes_chunks_questions_vectors_and_marks_indexed(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors, questioner = FakeVectors(), FakeQuestions()

    report = index_document(session, doc, vectors=vectors, questioner=questioner)

    chunks = chunks_of(session, doc)
    assert [c.heading_path for c in chunks] == [
        ["Grâul încolțit", "Beneficii"],
        ["Grâul încolțit", "Contraindicații"],
    ]
    assert [c.position for c in chunks] == [0, 1]
    assert chunks[0].text.startswith("# Grâul încolțit\n\n## Beneficii")  # fără front-matter
    questions = questions_of(session, doc)
    assert (report.chunks, report.questions, len(questions)) == (2, 4, 4)
    assert questioner.calls[0][:2] == ("Grâul încolțit", ["Grâul încolțit", "Beneficii"])
    assert questioner.calls[0][3] is False

    content = {e.id: e for e in vectors.of(doc.id, "content")}
    assert set(content) == {c.id for c in chunks}
    assert content[chunks[0].id].doc_id == chunks[0].id
    assert content[chunks[0].id].text == f"Grâul încolțit > Beneficii\n\n{chunks[0].text}"
    by_id = {e.id: e for e in vectors.of(doc.id, "question")}
    assert set(by_id) == {q.id for q in questions}
    assert all(by_id[q.id].doc_id == q.chunk_id and by_id[q.id].text == q.text for q in questions)

    assert (doc.stage, doc.indexed_hash, doc.last_error) == (
        Stage.indexed,
        doc.reviewed_hash,
        None,
    )
    assert registry.hash_warnings(doc) == []


def test_no_questions_indexes_only_content(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    report = index_document(session, doc, vectors=vectors, questioner=None)
    assert report.questions == 0 and questions_of(session, doc) == []
    assert {e.kind for e in vectors.rows.values()} == {"content"}
    assert doc.stage == Stage.indexed


def test_unsplit_document_is_one_chunk_with_whole_document_questions(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path, title="Regim DRI - Anemie", doc_type="regim")
    assert doc.split is False
    questioner = FakeQuestions()
    index_document(session, doc, vectors=FakeVectors(), questioner=questioner)
    [chunk] = chunks_of(session, doc)
    assert chunk.heading_path == ["Regim DRI - Anemie"]
    assert chunk.text == BOOK.strip()
    assert questioner.calls[0][3] is True


def test_reindex_replaces_old_rows_and_vectors_only_for_that_document(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    other = make_doc(session, tmp_path, title="Regim DRI - Anemie", doc_type="regim")
    vectors = FakeVectors()
    index_document(session, other, vectors=vectors, questioner=FakeQuestions())
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    old_ids = {e.id for e in vectors.of(doc.id)}
    other_ids = {e.id for e in vectors.of(other.id)}

    edit_and_review(session, doc, f"\n## Mod de folosire\n\n{MID}\n")
    report = index_document(session, doc, vectors=vectors, questioner=FakeQuestions())

    assert len(chunks_of(session, doc)) == 3
    assert len(questions_of(session, doc)) == 6
    assert len(session.exec(select(Question)).all()) == 6 + 2
    assert len(session.exec(select(Chunk)).all()) == 3 + 1
    new_ids = {e.id for e in vectors.of(doc.id)}
    assert not (old_ids & new_ids) and len(new_ids) == 3 + 6
    assert {e.id for e in vectors.of(other.id)} == other_ids
    assert any("se înlocuiesc 4 întrebări" in w for w in report.warnings)


def test_index_requires_reviewed_and_unchanged_markdown(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path, review=False)
    with pytest.raises(RegistryError, match="index cere 'reviewed'"):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())

    registry.review(session, doc)
    md = Path(doc.markdown_path)
    md.write_text(md.read_text(encoding="utf-8") + "\nadăugat\n", encoding="utf-8")
    with pytest.raises(
        RegistryError, match="markdown modificat după review; rulează din nou `review`"
    ):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())
    assert chunks_of(session, doc) == []
    assert (doc.stage, doc.last_error) == (Stage.reviewed, None)


def test_validated_document_needs_force(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    registry.validate(session, doc)

    with pytest.raises(RegistryError, match="--force"):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert doc.stage == Stage.validated

    index_document(session, doc, vectors=vectors, questioner=FakeQuestions(), force=True)
    assert doc.stage == Stage.indexed


def test_embedding_failure_keeps_previous_index_and_records_error(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors()
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    old_chunks = {c.id for c in chunks_of(session, doc)}
    edit_and_review(session, doc, f"\n## Nou\n\n{MID}\n")

    vectors.fail = "embed"
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())

    assert (doc.stage, doc.last_error) == (Stage.reviewed, "RuntimeError: OpenAI căzut")
    assert {c.id for c in chunks_of(session, doc)} == old_chunks  # nimic scris
    assert {e.doc_id for e in vectors.of(doc.id, "content")} == old_chunks


def test_vector_write_failure_keeps_stage_and_rerun_recovers(
    session: Session, tmp_path: Path
):
    doc = make_doc(session, tmp_path)
    vectors = FakeVectors(fail="write")
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert (doc.stage, doc.last_error) == (Stage.reviewed, "RuntimeError: pgvector căzut")
    assert len(chunks_of(session, doc)) == 2  # commit-ul relațional s-a făcut

    vectors.fail = None
    index_document(session, doc, vectors=vectors, questioner=FakeQuestions())
    assert (doc.stage, doc.last_error) == (Stage.indexed, None)
    assert len(chunks_of(session, doc)) == 2
    assert len(vectors.of(doc.id)) == 2 + 4


def test_question_failure_writes_nothing(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)

    def broken(_t: str, _h: list[str], _x: str, _w: bool) -> list[str]:
        raise RuntimeError("Gemini căzut")

    vectors = FakeVectors()
    with pytest.raises(RuntimeError):
        index_document(session, doc, vectors=vectors, questioner=broken)
    assert (doc.stage, doc.last_error) == (Stage.reviewed, "RuntimeError: Gemini căzut")
    assert chunks_of(session, doc) == [] and vectors.rows == {}


def test_empty_markdown_records_error(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path, body="<!-- VERIFICĂ: fereastra 1 -->\n")
    with pytest.raises(ValueError):
        index_document(session, doc, vectors=FakeVectors(), questioner=FakeQuestions())
    assert doc.stage == Stage.reviewed
    assert doc.last_error is not None and doc.last_error.startswith("ValueError: markdown gol")


def test_chunk_without_questions_is_reported(session: Session, tmp_path: Path):
    doc = make_doc(session, tmp_path)
    report = index_document(
        session, doc, vectors=FakeVectors(), questioner=FakeQuestions(per_chunk=0)
    )
    assert doc.stage == Stage.indexed
    assert "chunk 1: nicio întrebare generată" in report.warnings
```

- [ ] **Step 3:** `uv run pytest app/knowledge_prep/tests/test_index.py -q` → FAIL (`ModuleNotFoundError: app.knowledge_prep.index`).

- [ ] **Step 4: `index.py`**

```python
"""Indexarea unui document revizuit: chunk-uri, întrebări, vectori (spec 2026-09-25, „Indexarea”).

Ordinea e aleasă ca un eșec să lase cât mai puțin de curățat:
1. tot ce cere API-uri (întrebări Gemini, embeddings OpenAI) se face în memorie, înainte de scriere;
2. `chunk` și `question` se înlocuiesc într-o singură tranzacție;
3. vectorii documentului se șterg (după `document_id`) și se adaugă cei noi;
4. abia apoi `stage=indexed`, `indexed_hash=reviewed_hash`.
Un eșec la pasul 3 lasă chunk-urile noi fără vectori, stadiul vechi și `last_error`; o nouă
rulare `index` reface totul. Regulile încălcate ridică `RegistryError` fără să atingă registrul.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlmodel import Session, col, delete, select

from app.knowledge_prep.chunking import chunk_markdown, embedding_text
from app.knowledge_prep.frontmatter import strip_frontmatter
from app.knowledge_prep.models import Chunk, Document, Question, Stage
from app.knowledge_prep.questions import QuestionGenerator
from app.knowledge_prep.registry import RegistryError, body_hash, record_step
from app.knowledge_prep.vectors import VectorEntry, VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    chunks: int = 0
    questions: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        logger.warning("atenție: %s", message)
        self.warnings.append(message)


def check_indexable(doc: Document, *, force: bool) -> None:
    if doc.stage.rank < Stage.reviewed.rank:
        raise RegistryError(
            f"{doc.slug} e '{doc.stage}'; index cere 'reviewed' (rulează întâi review)."
        )
    md = Path(doc.markdown_path)
    if not md.is_file():
        raise RegistryError(f"lipsește markdown-ul: {md}")
    if body_hash(md) != doc.reviewed_hash:
        raise RegistryError("markdown modificat după review; rulează din nou `review`")
    if doc.stage == Stage.validated and not force:
        raise RegistryError(
            f"{doc.slug} e 'validated': reindexarea înlocuiește toate întrebările, "
            "inclusiv cele editate manual. Folosește --force."
        )


def _chunk_ids(doc: Document):
    return select(Chunk.id).where(Chunk.document_id == doc.id)


def index_document(
    session: Session,
    doc: Document,
    *,
    vectors: VectorIndex,
    questioner: QuestionGenerator | None,
    force: bool = False,
) -> IndexReport:
    """`reviewed`+ cu markdown nemodificat → chunk-uri, întrebări, vectori → `indexed`.

    `questioner=None` (`--no-questions`) indexează doar vectorii `content`: baseline-ul.
    """
    check_indexable(doc, force=force)
    report = IndexReport()
    replaced = len(
        session.exec(select(Question.id).where(col(Question.chunk_id).in_(_chunk_ids(doc)))).all()
    )
    if replaced:
        report.warn(f"se înlocuiesc {replaced} întrebări existente (editările manuale se pierd)")

    with record_step(session, doc):
        body = strip_frontmatter(Path(doc.markdown_path).read_text(encoding="utf-8"))
        drafts, warnings = chunk_markdown(body, doc.title, doc.split)
        for w in warnings:
            report.warn(w)
        chunks = [
            Chunk(document_id=doc.id, position=i, heading_path=d.heading_path, text=d.text)
            for i, d in enumerate(drafts)
        ]

        questions: list[Question] = []
        if questioner is not None:
            for i, chunk in enumerate(chunks, 1):
                logger.info("întrebări pentru „%s”: chunk %d/%d ...", doc.title, i, len(chunks))
                texts = questioner(doc.title, chunk.heading_path, chunk.text, not doc.split)
                if not texts:
                    report.warn(f"chunk {i}: nicio întrebare generată")
                questions += [Question(chunk_id=chunk.id, text=t) for t in texts]

        entries = [
            VectorEntry(c.id, c.id, doc.id, "content", embedding_text(c.heading_path, c.text))
            for c in chunks
        ] + [VectorEntry(q.id, q.chunk_id, doc.id, "question", q.text) for q in questions]
        logger.info("embeddings pentru %d vectori ...", len(entries))
        embeddings = vectors.embed([e.text for e in entries])

        # relațional, o singură tranzacție; delete explicit pe question (SQLite nu face cascade)
        session.exec(delete(Question).where(col(Question.chunk_id).in_(_chunk_ids(doc))))
        session.exec(delete(Chunk).where(col(Chunk.document_id) == doc.id))
        session.add_all(chunks)
        session.flush()
        session.add_all(questions)
        session.commit()

        vectors.replace_document(doc.id, entries, embeddings)
        doc.stage = Stage.indexed
        doc.indexed_hash = doc.reviewed_hash

    report.chunks, report.questions = len(chunks), len(questions)
    return report
```

Note pentru implementator: `document_id` și `id` se citesc **înainte** de `session.commit()` în `entries`; după commit, `doc` e expirat și se reîncarcă singur la `doc.id`/`doc.reviewed_hash`. Dacă `ty` se plânge de tipul de retur al lui `_chunk_ids`, adnotează-l ca `SelectOfScalar[uuid.UUID]` (`from sqlmodel.sql.expression import SelectOfScalar`).

- [ ] **Step 5:** `uv run pytest app/knowledge_prep -q` → PASS; ruff + ty curate.

- [ ] **Step 6: Commit**

```bash
git add backend/app/knowledge_prep/index.py backend/app/knowledge_prep/tests/index_fakes.py backend/app/knowledge_prep/tests/test_index.py
git commit -m "feat(knowledge_prep): index reviewed documents into chunks, questions and vectors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CLI `index <țintă> [--no-questions] [--force]`

**Files:**
- Modify: `backend/app/knowledge_prep/__main__.py` (docstring, `build_parser`, `_index` nou, `_dispatch`, `main`)
- Test: `backend/app/knowledge_prep/tests/test_cli.py`

**Interfaces:**
- Consumes: `index.index_document`, `IndexReport` (Task 4); `questions.QuestionGenerator`, `generate_questions` (Task 2); `vectors.VectorIndex`, `PgVectorIndex.from_settings` (Task 3); `tests/index_fakes.FakeVectors`, `FakeQuestions`.
- Produces: `main(argv=None, *, session=None, converters=CONVERTERS, restructurer=restructure, vectors: VectorIndex | None = None, questioner: QuestionGenerator = generate_questions) -> int`. `vectors=None` → `PgVectorIndex.from_settings()`, construit doar la `index`.

- [ ] **Step 1: Teste în `test_cli.py`** — `run()` trimite mereu fake-uri pentru index, ca nicio rulare din teste să nu atingă OpenAI/Gemini/pgvector:

```python
from app.knowledge_prep.tests.index_fakes import FakeQuestions, FakeVectors

MID = " ".join(["Grâul încolțit dă putere organismului slăbit."] * 10)


def run(
    session: Session,
    *argv: str,
    converters: Mapping[str, Converter] = CONVERTERS,
    vectors: FakeVectors | None = None,
) -> int:
    return main(
        list(argv),
        session=session,
        converters=converters,
        restructurer=fake_restructure,
        vectors=vectors or FakeVectors(),
        questioner=FakeQuestions(),
    )


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
    assert run(session, "index", "carte", vectors=vectors) == 1  # validated, fără --force
    assert stages(session) == {"carte": Stage.validated}
    assert run(session, "index", "carte", "--force", vectors=vectors) == 0
    assert {e.kind for e in vectors.rows.values()} == {"content", "question"}

    capsys.readouterr()
    assert run(session, "status", "carte") == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.split() == ["carte", "carte", "indexed"]  # fără avertismente, fără eroare


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
```

Celelalte apeluri directe `main(...)` din `test_cli.py` (dacă există) nu au nevoie de schimbare: `vectors`/`questioner` se folosesc doar la `index`.

- [ ] **Step 2:** `uv run pytest app/knowledge_prep/tests/test_cli.py -q` → FAIL (`unexpected keyword argument 'vectors'`).

- [ ] **Step 3: `__main__.py`**

Docstring: `register → prepare → review → index → validate, plus status.` Importuri:

```python
from app.knowledge_prep.index import index_document
from app.knowledge_prep.questions import QuestionGenerator, generate_questions
from app.knowledge_prep.vectors import PgVectorIndex, VectorIndex
```

În `build_parser`, după `review`:

```python
    idx = sub.add_parser(
        "index", help="chunking + întrebări + embeddings → 'indexed'"
    )
    idx.add_argument("target", help="id, slug sau cale")
    idx.add_argument(
        "--questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-questions: doar vectorii 'content' (baseline)",
    )
    idx.add_argument(
        "--force",
        action="store_true",
        help="reindexează un document 'validated' (întrebările editate manual se pierd)",
    )
```

Funcția nouă:

```python
def _index(
    args: argparse.Namespace,
    session: Session,
    vectors: VectorIndex | None,
    questioner: QuestionGenerator,
) -> int:
    try:
        doc = registry.resolve(session, args.target)
        report = index_document(
            session,
            doc,
            vectors=vectors if vectors is not None else PgVectorIndex.from_settings(),
            questioner=questioner if args.questions else None,
            force=args.force,
        )
    except Exception as e:  # regulă încălcată sau pas eșuat (last_error e deja scris)
        logger.error("EROARE %s: %s", args.target, e)
        return 1
    logger.info(
        "%s: %s (%d chunk-uri, %d întrebări, %d avertismente)",
        doc.slug,
        doc.stage,
        report.chunks,
        report.questions,
        len(report.warnings),
    )
    return 0
```

`_dispatch(args, session, converters, restructurer, vectors, questioner, parser)` primește `case "index": return _index(args, session, vectors, questioner)`; `main(..., vectors: VectorIndex | None = None, questioner: QuestionGenerator = generate_questions)` le dă mai departe în ambele ramuri. În lista de loggere zgomotoase adaugă `"openai"`.

- [ ] **Step 4:** `uv run pytest app/knowledge_prep -q` → PASS; ruff + ty curate. `uv run python -m app.knowledge_prep index --help` afișează `--questions/--no-questions` și `--force`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge_prep/__main__.py backend/app/knowledge_prep/tests/test_cli.py
git commit -m "feat(knowledge_prep): CLI index command (--no-questions, --force)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Docs

**Files:**
- Modify: `backend/app/knowledge_prep/README.md`
- Modify: `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` („Indexarea”, „Faze de implementare”)
- Modify: `/home/alex/projects/RAG/AGENTS.md` (în afara oricărui repo, nu se comite)
- Modify: `/home/alex/projects/RAG/calea_de_autovindecare/docs/taskuri.md` (repo git separat)

- [ ] **Step 1: README-ul modulului**
  - „Starea fazelor”: faza 3 → `✅ (data)`, faza 4 → „următoarea”; planul fazei 3 în lista de planuri de sus.
  - „Fluxul” / „Comenzi”: `uv run python -m app.knowledge_prep index <id|slug|cale> [--no-questions] [--force]`; `OPENAI_API_KEY` necesar pentru `index`, pe lângă cheile Gemini.
  - „Harta modulului”: `chunking.py`, `questions.py`, `vectors.py`, `index.py`, `tests/index_fakes.py`.
  - Secțiune nouă „Indexarea”, în stilul secțiunii „Restructurarea” (ce rezolvă, cum funcționează, de ce așa, cum se reglează), cu:
    - ordinea pașilor și de ce (API-uri înainte de scriere; ștergerea vectorilor după `document_id`; ce rămâne în urmă la fiecare tip de eșec și că o nouă rulare `index` repară);
    - regulile de chunking, inclusiv de ce parserul de titluri e propriu (exemplul `"# A\n\nText." → "# A  \nText."` cu `MarkdownHeaderTextSplitter`) și regula „același părinte” / secțiuni doar cu titluri;
    - schema `knowledge_vectors` (coloane, id-uri, textul embedded) și de ce nu e în Alembic; fără index ANN (limita de 2000 de dimensiuni);
    - regula `validated` → `--force` și avertismentul „se înlocuiesc N întrebări”;
    - limitări: fără cache pe întrebări (o carte întreruptă se plătește din nou), un apel Gemini per chunk (secvențial), tiktoken descarcă `cl100k_base` la prima rulare;
    - tabel „Cum se reglează”: `MIN_CHUNK_TOKENS`, `MAX_CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `UNSPLIT_WARN_TOKENS`, `PROMPT` din `questions.py`, `MAX_QUESTIONS`.
  - „Testare”: fake-urile `FakeVectors`/`FakeQuestions`; `main(...)` primește și `vectors=`, `questioner=` (dă-le fake în teste); `test_vectors_pg.py` rulează pe baza de dev și e sărit fără Postgres, creând și ștergând tabele `knowledge_vectors_test_*`.
- [ ] **Step 2: Spec-ul** — în „Faze de implementare”: `3. **Indexarea:** ... ✅ <data>`. La finalul secțiunii „Indexarea”, o notă ca la restructurare: `> **Implementat (<data>)**, cu precizări: parser propriu de titluri în loc de MarkdownHeaderTextSplitter (păstrează textul exact); embeddings înainte de scrierea relațională și ștergerea vectorilor după document_id; knowledge_vectors creat de PGEngine, nu de Alembic. Detalii: backend/app/knowledge_prep/README.md.`
- [ ] **Step 3: `RAG/AGENTS.md`** — în blocul „Comenzi”, sub `review|validate`: `uv run python -m app.knowledge_prep index <id|slug|cale> [--no-questions] [--force]  # chunk-uri + întrebări Gemini + vectori în knowledge_vectors`; fluxul din comentariu devine `register → prepare → review → index → validate`; notează că `index` cere și `OPENAI_API_KEY`.
- [ ] **Step 4: Commit în `full-stack-fastapi-template`**

```bash
git add backend/app/knowledge_prep/README.md docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md
git commit -m "docs(knowledge_prep): indexing (phase 3) in module README; mark phase 3 done in spec

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: `taskuri.md` (repo separat)** — rândul 5 („Stare implementare”): fazele 1-3 gata, următoarea e faza 4 (retrieval + evaluare); la „### 3. Modelul de date + indexarea”, bifează schema și indexarea, cu trimitere la README pentru abateri (fără BM25 încă: vine în faza 4; „reconstruirea completă din fișiere” rămâne nebifată: `index` e per document).

```bash
git -C /home/alex/projects/RAG/calea_de_autovindecare status --short docs/taskuri.md
```

Fișierul are deja modificări necomise ale userului (la 2026-09-28: `MM docs/taskuri.md`). **Nu comite peste ele fără acord:** dacă `git diff docs/taskuri.md` arată și altceva decât schimbările de la acest pas, întreabă userul dacă le comiți împreună. Altfel:

```bash
git -C /home/alex/projects/RAG/calea_de_autovindecare add docs/taskuri.md
git -C /home/alex/projects/RAG/calea_de_autovindecare commit -m "docs(taskuri): phase 3 (indexing) done

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Verificare reală (manuală, opțională — cere `OPENAI_API_KEY`, `GEMINI_API_KEY` și Postgres)

Nu se rulează automat. Costă puțin (câteva zeci de apeluri Gemini Flash + embeddings), dar **scrie în baza de dev**, iar pasul cu regimul scrie și în corpus: cere acordul userului înainte.

- [ ] **Step 1: Carte (`split=true`), baseline** — din `backend/`, cu `docker compose up -d db` pornit:

```bash
uv run python -m app.knowledge_prep status graul-incoltit-valeriu-popa     # reviewed, fără avertismente
uv run python -m app.knowledge_prep index graul-incoltit-valeriu-popa --no-questions
```

Dacă `status` arată „modificat după review”, userul revizuiește fișierul și rulează `review` întâi.

- [ ] **Step 2: Carte, cu întrebări** (reindexare peste baseline):

```bash
uv run python -m app.knowledge_prep index graul-incoltit-valeriu-popa
uv run python - <<'EOF'
from sqlalchemy import text
from app.core.db import engine
with engine.connect() as c:
    print(c.execute(text("select kind, count(*) from knowledge_vectors group by kind")).all())
    rows = c.execute(text("""
        select c.position, c.heading_path, length(c.text), q.text
        from chunk c join document d on d.id = c.document_id
        left join question q on q.chunk_id = c.id
        where d.slug = 'graul-incoltit-valeriu-popa' order by c.position"""))
    for r in rows:
        print(r)
EOF
```

De verificat: numărul de vectori = chunk-uri + întrebări (reindexarea nu a lăsat resturi); `heading_path` are sens; nicio secțiune tăiată aberant; întrebările sunt în română, numesc subiectul și au răspuns în chunk.

- [ ] **Step 3: Căutare de probă** (doar ca sanity check, retrieval-ul propriu-zis e faza 4):

```bash
uv run python - <<'EOF'
from app.knowledge_prep.vectors import PgVectorIndex
vi = PgVectorIndex.from_settings()
for q in ["Cine nu are voie să mănânce grâu încolțit?", "Cum se încolțește grâul?"]:
    print("==", q)
    for d in vi.store.similarity_search(q, k=5):
        print(d.metadata["kind"], "|", d.page_content[:100].replace("\n", " "))
EOF
```

- [ ] **Step 4: Regim (`split=false`)** — **doar cu acordul userului** (`prepare` scrie în `regimuri DRI/converted/`):

```bash
uv run python -m app.knowledge_prep register "../../calea_de_autovindecare/knowledge/calea-de-vindecare/regimuri DRI/Regim DRI - Anemie.md" --type regim
uv run python -m app.knowledge_prep prepare regim-dri-anemie
# userul revizuiește regimuri DRI/converted/regim-dri-anemie.md în IDE
uv run python -m app.knowledge_prep review regim-dri-anemie
uv run python -m app.knowledge_prep index regim-dri-anemie
```

De verificat: un singur chunk, `heading_path = ["Regim DRI - Anemie"]`, fără avertismentul de 8000 de tokeni (un regim are ~3k), întrebări care acoperă tot regimul, nu doar începutul.

- [ ] **Step 5:** `uv run alembic check` nu propune operații pe `knowledge_vectors`.

- [ ] **Step 6: Raport și README** — raport către user: timpii, numărul de chunk-uri și distribuția de tokeni, 5-10 întrebări reprezentative, problemele observate (de prompt vs. de cod). Adaugă „Rezultate reale (<data>)” în secțiunea „Indexarea” din README (ca la restructurare) și comite:

```bash
git add backend/app/knowledge_prep/README.md
git commit -m "docs(knowledge_prep): real indexing results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
