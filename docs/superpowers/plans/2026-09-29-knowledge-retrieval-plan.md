# Retrieval (faza 4, fără evaluare) — plan de implementare

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pentru o întrebare, întoarce top `k` chunk-uri din indexul fazei 3, în trei moduri: baseline (doar vectorii `content`), multi-vector (`content` + `question`) și hibrid (multi-vector sau baseline + BM25 pe `chunk.text`, combinate prin reciprocal rank fusion pe `chunk.id`), plus o comandă CLI `search` pentru verificare manuală.

**Architecture:** Trei module noi în `app/knowledge_prep/`, fiecare testabil singur: `docstore.py` (`ChunkDocStore`, părinții lui `MultiVectorRetriever`, citiți din tabelul `chunk` + `document`), `lexical.py` (tokenizare fără diacritice + `ChunkBM25`, index BM25 în memorie peste `chunk.text`, cu `rank_bm25`) și `retrieval.py` (`reciprocal_rank_fusion`, `KnowledgeRetriever(BaseRetriever)` și fabrica `build_retriever`). Partea semantică e `MultiVectorRetriever` din `langchain_classic` peste `PgVectorIndex.store` (`knowledge_vectors`), cu `id_key="doc_id"` și `search_kwargs["k"] = k * CHILDREN_PER_CHUNK`; baseline-ul adaugă `filter={"kind": "content"}`. Vectorstore-ul se injectează (`build_retriever(vectorstore, bind, ...)`), ca testele să ruleze cu un fake în memorie pe SQLite.

**Tech Stack:** Python 3.14, SQLModel 0.0.42, langchain-core 1.6.3 (`BaseStore`, `BaseRetriever`, `VectorStore`, `DeterministicFakeEmbedding`), langchain-classic 1.0.8 (`langchain_classic.retrievers.multi_vector.MultiVectorRetriever`), langchain-postgres 0.0.18 (`PGVectorStore` prin `PgVectorIndex` din faza 3), rank-bm25 0.2.2 (`BM25Okapi`), tenacity, pytest. **O singură schimbare de dependențe:** `langchain-classic` devine dependență directă (azi e instalat doar tranzitiv, prin `langchain-community`, care e în sunset — decizia 5 din spec). `rank-bm25` e deja dependență directă.

**Spec:** `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` (secțiunile „Decizii” — 5, 9, 11 —, „Tabelul de vectori”, „Retrieval”, „Faze de implementare” — faza 4 —, „Testare”).

## Global Constraints

- Comenzile se rulează din `full-stack-fastapi-template/backend/` (`uv run ...`). Testele: `uv run pytest app/knowledge_prep` (nu `backend/tests/`, care pică la import). Lint: `uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep`.
- `ChunkDocStore(BaseStore[str, Document])`: `mget` citește din tabelul `chunk` și întoarce `Document(page_content=text, metadata={chunk_id, document_id, slug, title, heading_path})`. `mset`/`mdelete` ridică `NotImplementedError`: scrierea se face doar prin indexare.
- `MultiVectorRetriever(vectorstore=knowledge_vectors, docstore=ChunkDocStore(), id_key="doc_id")`. `search_kwargs["k"]` pentru copii e mai mare decât numărul de chunk-uri dorit.
- Baseline = același retriever, cu `search_kwargs["filter"] = {"kind": "content"}`.
- Hibrid = BM25 peste `chunk.text`, combinat cu rezultatele semantice prin reciprocal rank fusion pe `chunk.id`.
- **Excluse din acest plan:** evaluarea (metricile din `rag_simple/evaluation/`, seturile de întrebări de test, comparația baseline vs. multi-vector), rescrierea query-ului, reranking-ul, REST-ul (faza 5).
- `rag_simple` rămâne neatins (nici nu se importă din el). Nu se folosește `langchain-community` (pachet în sunset). Fără modele locale grele (WSL cu RAM puțin). Nu se editează `.env`.
- Orice apel API nou: `@retry(..., stop=stop_after_attempt(N), before_sleep=..., reraise=True)`. Niciodată `@retry(wait=...)` fără `stop=`.
- Mesajele pentru user și docstring-urile sunt în română, ca în codul existent.

## Review Focus

1. **Id-urile părinților vin din Postgres ca `uuid.UUID`, nu ca `str`** (coloana `doc_id` e `UUID`; `PGVectorStore` pune în metadate valoarea brută a rândului, iar `MultiVectorRetriever` o dă neschimbată lui `docstore.mget`). `mget` trebuie să accepte ambele. Teste: `test_mget_accepts_uuid_objects` (Task 1), `test_docstore_reads_postgres_rows_and_uuid_keys` și `test_multi_vector_finds_chunk_through_its_question` (Task 4, pe Postgres real).
2. **Întrebări fără diacritice sau cu sedilă (`ş/ţ`) față de un text cu virgulă (`ș/ț`), și invers:** BM25 le găsește la fel. Teste: `test_tokenize_lowercases_and_strips_diacritics`, `test_search_ignores_diacritics_both_ways` (Task 2).
3. **Un chunk cu multe întrebări nu ocupă toate locurile:** cu 5 întrebări + `content` pentru chunk-ul 1, `k=2` întoarce tot două chunk-uri distincte. Test: `test_many_questions_of_one_chunk_do_not_crowd_out_the_others` (Task 3).
4. **Chunk-uri dispărute după construirea retriever-ului** (reindexare cât timp un retriever trăiește; vectori orfani de la o indexare întreruptă): se sar, fără excepție, iar rezultatul se completează din ceilalți candidați. Test: `test_chunks_missing_from_db_are_skipped` (Task 3).
5. **Index gol, query doar din punctuație, `-k 0`:** listă goală / mesaj clar, fără excepții din `rank_bm25` (împărțire la zero pe corpus gol). Teste: `test_empty_index_returns_nothing`, `test_k_must_be_positive` (Task 3), `test_query_without_words_and_empty_corpus_return_nothing` (Task 2), `test_search_rejects_non_positive_k` și `test_search_rejects_blank_query` (Task 5).

---

## Decizii de implementare (interpretări ale spec-ului)

1. **Locul modulului: în `app/knowledge_prep/`, ca trei fișiere plate (`docstore.py`, `lexical.py`, `retrieval.py`), nu un pachet nou.** Spec-ul lasă numele deschis. Retrieval-ul citește exact tabelele pe care le scrie `knowledge_prep` (`chunk`, `document`, `knowledge_vectors`), refolosește `PgVectorIndex`, `MAX_QUESTIONS`, `embedding_text`, CLI-ul și fixture-urile/fake-urile din `tests/`; un pachet separat ar importa oricum toate acestea din `knowledge_prep`. REST-ul (faza 5) și evaluarea vor importa `build_retriever` de aici. `rag_simple` nu se atinge (AGENTS.md). Un viitor agent (LangGraph) merge într-un folder nou, conform AGENTS.md, și doar importă `build_retriever`. Numele `knowledge_prep` e puțin îngust pentru retrieval, dar redenumirea pachetului e în afara acestui plan.
2. **Hibridul: BM25 cu `rank_bm25` în memorie peste `chunk.text`, nu `hybrid_search_config` din `PGVectorStore`.** Motive:
   - `hybrid_search_config` face full-text pe coloana `content` a lui `knowledge_vectors`, adică pe **copii** (textele întrebărilor + `heading_path` + chunk). Fuziunea lui se face pe id-uri de copii, înainte de reducerea la părinți, deci un chunk cu 5 întrebări ar fi favorizat de 6 ori, iar spec-ul cere RRF pe `chunk.id`.
   - Scorul lui e `ts_rank` din Postgres (nu BM25) și fuziunea implicită e o sumă ponderată de scoruri normalizate, nu RRF.
   - Cere o coloană `tsvector` + index GIN pe `knowledge_vectors`, tabel pe care îl creează `PGEngine`, nu Alembic (decizia 2 din planul fazei 3). Configurația `romanian` există în Postgres, dar nu scoate diacriticele fără `unaccent` și o configurație proprie — adică o migrare nouă.
   - Corpusul e mic (36 de chunk-uri azi, câteva mii în viitorul apropiat): un index în memorie se construiește în milisecunde. `rank_bm25` e deja dependență directă și e folosit de `rag_simple`.
   - Nu `BM25Retriever` din `langchain_community`: pachet în sunset (decizia 5), și e doar un înveliș peste `rank_bm25`.
3. **Tokenizarea BM25:** litere mici, NFKD, fără semnele combinante (deci `ă→a`, `â/î→a/i`, `ș/ş→s`, `ț/ţ→t`), apoi `\w+`. Aceeași funcție pe corpus și pe query. Motiv: utilizatorii scriu des fără diacritice, iar unele texte vechi au sedilă. **Fără stemming și fără stopwords** în acest plan: `grâu` ≠ `grâul` pentru BM25 (partea semantică le acoperă). Se reevaluează după evaluare (întrebare deschisă, vezi finalul).
4. **Indexul BM25 se construiește o dată, la `build_retriever`,** din toate chunk-urile din DB. BM25 întoarce doar id-uri; documentele se citesc mereu prin `ChunkDocStore.mget`, deci un chunk șters între timp dispare din rezultate (nu apare text vechi). Un chunk **nou** nu apare în BM25 până la un nou `build_retriever` — limită documentată în README; CLI-ul construiește retriever-ul la fiecare rulare.
5. **Numărul de copii ceruți:** `k * CHILDREN_PER_CHUNK`, cu `CHILDREN_PER_CHUNK = MAX_QUESTIONS + 1 = 6` (un vector `content` + cel mult 5 întrebări per chunk). Așa, chiar dacă toți copiii găsiți ar veni în grupuri complete, rămân cel puțin `k` părinți distincți (când există). Același număr se folosește pentru candidații BM25. La corpusul actual (sute de vectori, scanare exactă) costul e neglijabil.
6. **Modurile sunt două opțiuni ortogonale, nu un enum:** `questions: bool` (multi-vector vs. baseline) și `hybrid: bool` (cu/fără BM25). Evaluarea de mai târziu va avea nevoie de toate patru combinațiile. CLI-ul le expune ca `--questions/--no-questions` (același nume ca la `index`) și `--hybrid/--no-hybrid`; implicit ambele pornite.
7. **`KnowledgeRetriever` e un `BaseRetriever` din langchain-core** (se poate folosi direct ca tool într-un agent). Întoarce cel mult `k` documente, ordonate după RRF (sau după ordinea semantică, fără hibrid). RRF cu constanta `RRF_K = 60` (ca în `rag_simple` și în articolul original); la egalitate, câștigă ordinea listei semantice.
8. **Retry pe embedding-ul query-ului:** căutarea semantică (care cheamă OpenAI pentru embedding-ul întrebării) e împachetată în `_semantic_search`, cu `@retry` pe `openai.APIError`, `stop_after_attempt(3)`, ca `PgVectorIndex.embed`.
9. **Nu se filtrează după stadiul documentului.** Un document readus în `structured` de `prepare --force` își păstrează chunk-urile și vectorii până la următorul `index` (spec, „Stadii”), deci rămâne căutabil. Documentat în README.
10. **CLI `search`: da**, e ieftin (~40 de rânduri) și e singurul mod de a vedea retrieval-ul pe corpusul real înainte de evaluare. Nu e un pas din pipeline și nu schimbă registrul.
11. **Testele de integrare pe Postgres** scriu rânduri reale în `document`/`chunk` (slug `retrieval-test-<hex>`) și le șterg în `finally`, plus un tabel `knowledge_vectors_test_<hex>` șters la final. Helper-ul de conexiune din `test_vectors_pg.py` se mută în `tests/pg.py`, ca să fie folosit de ambele fișiere.

## Structura fișierelor

| Fișier | Rol |
|---|---|
| Create `backend/app/knowledge_prep/docstore.py` | `Bind`, `ChunkDocStore`, `chunk_document` |
| Create `backend/app/knowledge_prep/lexical.py` | `tokenize`, `ChunkBM25` |
| Create `backend/app/knowledge_prep/retrieval.py` | `DEFAULT_K`, `RRF_K`, `CHILDREN_PER_CHUNK`, `reciprocal_rank_fusion`, `KnowledgeRetriever`, `build_retriever` |
| Modify `backend/pyproject.toml`, `uv.lock` (rădăcina full-stack) | `langchain-classic` dependență directă |
| Modify `backend/app/knowledge_prep/__main__.py` | subcomanda `search`; `main(..., vectorstore=)` |
| Create `backend/app/knowledge_prep/tests/retrieval_fakes.py` | `add_document`, `child`, `FakeChildStore` |
| Create `backend/app/knowledge_prep/tests/pg.py` | `PG_URL`, `requires_pg`, `has_table`, `drop_table` (mutate din `test_vectors_pg.py`) |
| Create `backend/app/knowledge_prep/tests/test_docstore.py`, `test_lexical.py`, `test_retrieval.py`, `test_retrieval_pg.py` | teste |
| Modify `backend/app/knowledge_prep/tests/test_vectors_pg.py` | folosește `tests/pg.py` |
| Modify `backend/app/knowledge_prep/tests/test_cli.py` | `run(..., vectorstore=)`; teste `search` |
| Modify `backend/app/knowledge_prep/README.md`, spec-ul, `RAG/AGENTS.md`, `calea_de_autovindecare/docs/taskuri.md` | docs (Task 7) |

---

### Task 1: `docstore.py` — `ChunkDocStore`

**Files:**
- Create: `backend/app/knowledge_prep/docstore.py`
- Create: `backend/app/knowledge_prep/tests/retrieval_fakes.py`
- Test: `backend/app/knowledge_prep/tests/test_docstore.py`

**Interfaces:**
- Consumes: `models.Chunk`, `models.Document`, `DocumentType`, `Stage` (faza 1); fixture-ul `session` din `tests/conftest.py` (SQLite in-memory, `StaticPool`).
- Produces:
  - `Bind = Engine | Connection` (alias de tip, în `docstore.py`);
  - `chunk_document(chunk: Chunk, doc: models.Document) -> langchain_core.documents.Document`;
  - `ChunkDocStore(bind: Bind)` cu `mget(keys: Sequence[str]) -> list[Document | None]` (acceptă și `uuid.UUID`; ordinea cheilor; `None` pentru cheie inexistentă sau invalidă), `yield_keys(*, prefix: str | None = None) -> Iterator[str]`, `mset`/`mdelete` → `NotImplementedError`;
  - metadatele: `{"chunk_id": str, "document_id": str, "slug": str, "title": str, "heading_path": list[str]}`; `Document.id = chunk_id`;
  - în `tests/retrieval_fakes.py`: `add_document(session: Session, slug: str, sections: list[tuple[list[str], str]]) -> list[Chunk]` (titlul documentului = `f"Titlu {slug}"`, commit + refresh).

- [ ] **Step 1: Helper-ul de test** — `backend/app/knowledge_prep/tests/retrieval_fakes.py` (se completează în Task 3 cu `child` și `FakeChildStore`):

```python
"""Fake-uri și helper-e pentru testele de retrieval (fără pgvector, fără OpenAI)."""

from sqlmodel import Session

from app.knowledge_prep import models
from app.knowledge_prep.models import Chunk, DocumentType, Stage


def add_document(
    session: Session, slug: str, sections: list[tuple[list[str], str]]
) -> list[Chunk]:
    """Un document `indexed` cu chunk-urile date ca (heading_path, text), în ordine."""
    doc = models.Document(
        slug=slug,
        title=f"Titlu {slug}",
        type=DocumentType.carte,
        split=True,
        stage=Stage.indexed,
        markdown_path=f"/nu-exista/{slug}.md",
    )
    session.add(doc)
    session.flush()
    chunks = [
        Chunk(document_id=doc.id, position=i, heading_path=path, text=text)
        for i, (path, text) in enumerate(sections)
    ]
    session.add_all(chunks)
    session.commit()
    for c in chunks:
        session.refresh(c)
    return chunks
```

- [ ] **Step 2: Testele** — `backend/app/knowledge_prep/tests/test_docstore.py`:

```python
import uuid

import pytest
from sqlmodel import Session

from app.knowledge_prep.docstore import ChunkDocStore
from app.knowledge_prep.tests.retrieval_fakes import add_document

SECTIONS = [(["Grâul", "A"], "text a"), (["Grâul", "B"], "text b")]


def test_mget_returns_documents_in_key_order_with_metadata(session: Session):
    a, b = add_document(session, "grau", SECTIONS)
    store = ChunkDocStore(session.get_bind())

    got = store.mget([str(b.id), str(a.id)])

    assert [d.page_content for d in got if d] == ["text b", "text a"]
    first = got[0]
    assert first is not None
    assert first.id == str(b.id)
    assert first.metadata == {
        "chunk_id": str(b.id),
        "document_id": str(b.document_id),
        "slug": "grau",
        "title": "Titlu grau",
        "heading_path": ["Grâul", "B"],
    }


def test_mget_missing_and_invalid_keys_are_none(session: Session):
    (a,) = add_document(session, "grau", SECTIONS[:1])
    store = ChunkDocStore(session.get_bind())

    got = store.mget([str(uuid.uuid4()), "nu-e-uuid", str(a.id)])

    assert got[:2] == [None, None]
    third = got[2]
    assert third is not None and third.page_content == "text a"
    assert store.mget([]) == []


def test_mget_accepts_uuid_objects(session: Session):
    # PGVectorStore pune în metadate `doc_id` ca uuid.UUID (coloana e UUID),
    # iar MultiVectorRetriever îl dă neschimbat lui mget.
    (a,) = add_document(session, "grau", SECTIONS[:1])
    (got,) = ChunkDocStore(session.get_bind()).mget([a.id])  # type: ignore[list-item]
    assert got is not None and got.metadata["chunk_id"] == str(a.id)


def test_store_is_read_only(session: Session):
    store = ChunkDocStore(session.get_bind())
    with pytest.raises(NotImplementedError, match="index"):
        store.mset([])
    with pytest.raises(NotImplementedError, match="index"):
        store.mdelete([])


def test_yield_keys_lists_chunk_ids(session: Session):
    a, b = add_document(session, "grau", SECTIONS)
    store = ChunkDocStore(session.get_bind())
    assert set(store.yield_keys()) == {str(a.id), str(b.id)}
    assert list(store.yield_keys(prefix=str(a.id))) == [str(a.id)]
```

- [ ] **Step 3: Rulează testele, trebuie să pice**

Run: `uv run pytest app/knowledge_prep/tests/test_docstore.py -v`
Expected: FAIL cu `ModuleNotFoundError: No module named 'app.knowledge_prep.docstore'`

- [ ] **Step 4: Implementarea** — `backend/app/knowledge_prep/docstore.py`:

```python
"""Părinții lui MultiVectorRetriever: chunk-urile din tabelul `chunk` (spec 2026-09-25, „Retrieval”).

Doar citire: chunk-urile se scriu numai prin `index` (faza 3), deci `mset`/`mdelete` ridică
`NotImplementedError`. Fiecare apel deschide o sesiune scurtă pe `bind`, ca retriever-ul să poată
trăi mai mult decât o sesiune (ex. într-un server).
"""

import uuid
from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from sqlalchemy import Connection, Engine
from sqlmodel import Session, col, select

from app.knowledge_prep import models
from app.knowledge_prep.models import Chunk

Bind = Engine | Connection

_READ_ONLY = "ChunkDocStore e doar pentru citire: chunk-urile se scriu prin `index`."


def _as_uuid(key: object) -> uuid.UUID | None:
    # PGVectorStore întoarce `doc_id` ca uuid.UUID, restul lumii ca str
    if isinstance(key, uuid.UUID):
        return key
    try:
        return uuid.UUID(str(key))
    except ValueError:
        return None


def chunk_document(chunk: Chunk, doc: models.Document) -> Document:
    return Document(
        id=str(chunk.id),
        page_content=chunk.text,
        metadata={
            "chunk_id": str(chunk.id),
            "document_id": str(doc.id),
            "slug": doc.slug,
            "title": doc.title,
            "heading_path": list(chunk.heading_path),
        },
    )


class ChunkDocStore(BaseStore[str, Document]):
    def __init__(self, bind: Bind):
        self.bind = bind

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        """Chunk-urile cerute, în ordinea cheilor; `None` pentru o cheie inexistentă sau invalidă."""
        wanted = [_as_uuid(k) for k in keys]
        ids = {u for u in wanted if u is not None}
        if not ids:
            return [None] * len(keys)
        with Session(self.bind) as s:
            rows = s.exec(
                select(Chunk, models.Document)
                .join(models.Document, col(Chunk.document_id) == col(models.Document.id))
                .where(col(Chunk.id).in_(ids))
            ).all()
            found = {chunk.id: chunk_document(chunk, doc) for chunk, doc in rows}
        return [found.get(u) if u is not None else None for u in wanted]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        raise NotImplementedError(_READ_ONLY)

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError(_READ_ONLY)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        with Session(self.bind) as s:
            ids = [str(i) for i in s.exec(select(Chunk.id).order_by(col(Chunk.id)))]
        return (i for i in ids if prefix is None or i.startswith(prefix))
```

- [ ] **Step 5: Rulează testele, trebuie să treacă**

Run: `uv run pytest app/knowledge_prep/tests/test_docstore.py -v`
Expected: 5 passed

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep
git add backend/app/knowledge_prep/docstore.py backend/app/knowledge_prep/tests/retrieval_fakes.py backend/app/knowledge_prep/tests/test_docstore.py
git commit -m "feat(knowledge_prep): ChunkDocStore, read-only parent store over the chunk table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Dacă `ty` nu acceptă `# type: ignore[list-item]` din test, folosește `# type: ignore` simplu; nu schimba semnătura lui `mget`.)

---

### Task 2: `lexical.py` — BM25 peste `chunk.text`

**Files:**
- Create: `backend/app/knowledge_prep/lexical.py`
- Test: `backend/app/knowledge_prep/tests/test_lexical.py`

**Interfaces:**
- Consumes: `Bind` din `docstore.py` (Task 1); `add_document` din `tests/retrieval_fakes.py` (Task 1); `rank_bm25.BM25Okapi` (`BM25Okapi(tokenized_corpus)`, `get_scores(tokenized_query) -> numpy array`, verificat în documentația rank_bm25).
- Produces:
  - `tokenize(text: str) -> list[str]`;
  - `ChunkBM25(chunk_ids: list[str], texts: list[str])`, `ChunkBM25.from_db(bind: Bind) -> ChunkBM25`, `ChunkBM25.search(query: str, k: int) -> list[str]` (id-uri de chunk ca `str`, cel mai bun primul, doar scoruri `> 0`).

- [ ] **Step 1: Testele** — `backend/app/knowledge_prep/tests/test_lexical.py`:

```python
from sqlmodel import Session

from app.knowledge_prep.lexical import ChunkBM25, tokenize
from app.knowledge_prep.tests.retrieval_fakes import add_document

TEXTS = {
    "a": "Grâul încolțit dă putere organismului.",
    "b": "Ceaiul de mușețel calmează stomacul.",
    "c": "Sucul de morcovi şi ţelină pentru ficat.",  # sedilă, ca în textele vechi
    "d": "Postul negru durează o zi.",
}


def bm25() -> ChunkBM25:
    return ChunkBM25(list(TEXTS), list(TEXTS.values()))


def test_tokenize_lowercases_and_strips_diacritics():
    assert tokenize("Grâul ÎNCOLȚIT, şi ţelina!") == ["graul", "incoltit", "si", "telina"]
    assert tokenize("?! — ...") == []


def test_search_ignores_diacritics_both_ways():
    index = bm25()
    assert index.search("graul incoltit", 3) == ["a"]
    assert index.search("țelină", 3) == ["c"]  # virgulă în query, sedilă în text
    assert index.search("MUSETEL", 3) == ["b"]


def test_search_ranks_and_limits_to_k():
    index = bm25()
    ranked = index.search("sucul postul ceaiul", 4)
    assert set(ranked) == {"b", "c", "d"}
    assert len(index.search("sucul postul ceaiul", 2)) == 2


def test_query_without_words_and_empty_corpus_return_nothing():
    assert bm25().search("?!", 3) == []
    assert bm25().search("necunoscut", 3) == []
    assert ChunkBM25([], []).search("grâu", 3) == []


def test_from_db_indexes_all_chunks(session: Session):
    chunks = add_document(
        session,
        "grau",
        [(["G", str(i)], text) for i, text in enumerate(TEXTS.values())],
    )
    index = ChunkBM25.from_db(session.get_bind())
    assert index.search("morcovi", 3) == [str(chunks[2].id)]
```

- [ ] **Step 2: Rulează testele, trebuie să pice**

Run: `uv run pytest app/knowledge_prep/tests/test_lexical.py -v`
Expected: FAIL cu `ModuleNotFoundError: No module named 'app.knowledge_prep.lexical'`

- [ ] **Step 3: Implementarea** — `backend/app/knowledge_prep/lexical.py`:

```python
"""BM25 peste `chunk.text` (spec 2026-09-25, „Retrieval” → „Hibrid”).

Index în memorie cu `rank_bm25`, construit o dată din toate chunk-urile din DB. Întoarce doar
id-uri: documentele se citesc prin `ChunkDocStore`, deci un chunk șters între timp nu apare.
De ce nu `hybrid_search_config` din PGVectorStore: vezi planul fazei 4, decizia 2.
"""

import re
import unicodedata

from rank_bm25 import BM25Okapi
from sqlmodel import Session, col, select

from app.knowledge_prep.docstore import Bind
from app.knowledge_prep.models import Chunk

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Cuvinte cu litere mici, fără diacritice: `Grâul` = `graul`, `ș`/`ş` = `s`, `ț`/`ţ` = `t`.

    Utilizatorii scriu des fără diacritice, iar unele texte au sedilă în loc de virgulă.
    Fără stemming: `grâu` ≠ `grâul` (partea semantică acoperă formele flexionate).
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WORD.findall(plain)


class ChunkBM25:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        # BM25Okapi împarte la lungimea medie: pe corpus gol ar da eroare
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    @classmethod
    def from_db(cls, bind: Bind) -> ChunkBM25:
        with Session(bind) as s:
            rows = s.exec(
                select(Chunk.id, Chunk.text).order_by(
                    col(Chunk.document_id), col(Chunk.position)
                )
            ).all()
        return cls([str(i) for i, _ in rows], [t for _, t in rows])

    def search(self, query: str, k: int) -> list[str]:
        """Cele mai bune `k` chunk-uri cu scor pozitiv (niciun cuvânt comun → listă goală)."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunk_ids[i] for i in ranked[:k] if scores[i] > 0]
```

Notă pentru implementator: `BM25Okapi` dă IDF 0 unui cuvânt prezent în exact jumătate din documente (și IDF negativ, înlocuit cu `epsilon * average_idf`, peste jumătate). De aceea testele folosesc un corpus de 4 texte în care cuvintele căutate apar într-unul singur; pe un corpus de 2 chunk-uri, un cuvânt dintr-unul are scor 0 și nu e întors.

- [ ] **Step 4: Rulează testele, trebuie să treacă**

Run: `uv run pytest app/knowledge_prep/tests/test_lexical.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep
git add backend/app/knowledge_prep/lexical.py backend/app/knowledge_prep/tests/test_lexical.py
git commit -m "feat(knowledge_prep): in-memory BM25 over chunk text, diacritics-insensitive

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `retrieval.py` — multi-vector, baseline, hibrid

**Files:**
- Modify: `backend/pyproject.toml`, `uv.lock` (prin `uv add`)
- Create: `backend/app/knowledge_prep/retrieval.py`
- Modify: `backend/app/knowledge_prep/tests/retrieval_fakes.py` (adaugă `child`, `FakeChildStore`)
- Test: `backend/app/knowledge_prep/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `ChunkDocStore`, `Bind` (Task 1); `ChunkBM25` (Task 2); `MAX_QUESTIONS` din `questions.py`; `log_retry` din `converters/gemini.py`; `VectorEntry` din `vectors.py`; `MultiVectorRetriever(vectorstore=..., docstore=..., id_key=..., search_kwargs=...)` din `langchain_classic.retrievers.multi_vector` (verificat în sursa 1.0.8: cheamă `vectorstore.similarity_search(query, **search_kwargs)`, păstrează ordinea primei apariții a fiecărui `metadata[id_key]`, apoi `docstore.mget(ids)` și scoate `None`-urile).
- Produces:
  - `DEFAULT_K = 5`, `RRF_K = 60`, `CHILDREN_PER_CHUNK = MAX_QUESTIONS + 1`;
  - `reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]`;
  - `KnowledgeRetriever(BaseRetriever)` cu câmpurile `semantic: MultiVectorRetriever`, `docstore: ChunkDocStore`, `lexical: ChunkBM25 | None = None`, `k: int = DEFAULT_K`; `.invoke(query: str) -> list[Document]` (metadatele din Task 1);
  - `build_retriever(vectorstore: VectorStore, bind: Bind, *, k: int = DEFAULT_K, questions: bool = True, hybrid: bool = True) -> KnowledgeRetriever` (`ValueError` pentru `k < 1`);
  - în `tests/retrieval_fakes.py`: `child(chunk: Chunk, kind: str, text: str = "copil") -> Document` și `FakeChildStore(children: list[Document])` cu `.calls: list[dict]` (`{"query", "k", "filter"}`) și `FakeChildStore.from_entries(entries: Iterable[VectorEntry]) -> FakeChildStore`.

- [ ] **Step 1: Dependența directă**

```bash
uv add "langchain-classic>=1.0.8"
uv run python -c "from langchain_classic.retrievers.multi_vector import MultiVectorRetriever; print('ok')"
```

Expected: `ok`; `backend/pyproject.toml` are `"langchain-classic>=1.0.8"` în `dependencies`, iar `uv.lock` (în rădăcina full-stack) e actualizat.

- [ ] **Step 2: Fake-ul de vectorstore** — adaugă în `backend/app/knowledge_prep/tests/retrieval_fakes.py` (importurile noi sus, restul la final):

```python
from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.knowledge_prep.vectors import VectorEntry


def child(chunk: Chunk, kind: str, text: str = "copil") -> Document:
    """Un vector-copil cum îl întoarce PGVectorStore: `doc_id`/`document_id` ca uuid.UUID."""
    return Document(
        page_content=text,
        metadata={"doc_id": chunk.id, "document_id": chunk.document_id, "kind": kind},
    )


class FakeChildStore(VectorStore):
    """Copiii în ordinea dată (primul = cel mai apropiat), indiferent de query.

    Filtrul se aplică la fel ca în PGVectorStore pentru egalități simple (`{"kind": "content"}`).
    """

    def __init__(self, children: list[Document]):
        self.children = children
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_entries(cls, entries: Iterable[VectorEntry]) -> FakeChildStore:
        return cls(
            [
                Document(
                    page_content=e.text,
                    metadata={
                        "doc_id": e.doc_id,
                        "document_id": e.document_id,
                        "kind": e.kind,
                    },
                )
                for e in entries
            ]
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        flt = kwargs.get("filter")
        self.calls.append({"query": query, "k": k, "filter": flt})
        hits = [
            d
            for d in self.children
            if all(d.metadata.get(key) == v for key, v in (flt or {}).items())
        ]
        return hits[:k]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FakeChildStore:
        raise NotImplementedError
```

- [ ] **Step 3: Testele** — `backend/app/knowledge_prep/tests/test_retrieval.py`:

```python
import pytest
from langchain_core.documents import Document
from sqlmodel import Session

from app.knowledge_prep.models import Chunk
from app.knowledge_prep.retrieval import (
    CHILDREN_PER_CHUNK,
    build_retriever,
    reciprocal_rank_fusion,
)
from app.knowledge_prep.tests.retrieval_fakes import (
    FakeChildStore,
    add_document,
    child,
)


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


@pytest.fixture
def chunks(session: Session) -> list[Chunk]:
    return add_document(
        session,
        "grau",
        [
            (["Grâul", "Beneficii"], "Grâul încolțit dă putere."),
            (["Grâul", "Preparare"], "Boabele se țin la apă o zi."),
            (["Grâul", "Contraindicații"], "Nu se dă copiilor mici."),
            (["Grâul", "Istoric"], "Metoda zxqvortex a fost descrisă de Popa."),
        ],
    )


def test_rrf_sums_reciprocal_ranks():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert reciprocal_rank_fusion([["x", "y"]]) == ["x", "y"]
    assert reciprocal_rank_fusion([["x"], ["y"]]) == ["x", "y"]  # egalitate: prima listă
    assert reciprocal_rank_fusion([]) == []


def test_multi_vector_dedupes_children_and_asks_for_more_children_than_k(
    session: Session, chunks: list[Chunk]
):
    c1, c2, c3, _ = chunks
    store = FakeChildStore(
        [
            child(c2, "question"),
            child(c2, "content"),
            child(c1, "question"),
            child(c3, "content"),
        ]
    )
    retriever = build_retriever(store, session.get_bind(), k=2, hybrid=False)

    assert ids(retriever.invoke("orice")) == [str(c2.id), str(c1.id)]
    assert store.calls == [
        {"query": "orice", "k": 2 * CHILDREN_PER_CHUNK, "filter": None}
    ]


def test_baseline_searches_only_content_vectors(session: Session, chunks: list[Chunk]):
    c1, c2, c3, _ = chunks
    store = FakeChildStore(
        [
            child(c2, "question"),
            child(c1, "question"),
            child(c3, "content"),
            child(c2, "content"),
        ]
    )
    retriever = build_retriever(
        store, session.get_bind(), k=5, questions=False, hybrid=False
    )

    assert ids(retriever.invoke("orice")) == [str(c3.id), str(c2.id)]
    assert store.calls[0]["filter"] == {"kind": "content"}


def test_many_questions_of_one_chunk_do_not_crowd_out_the_others(
    session: Session, chunks: list[Chunk]
):
    c1, c2, *_ = chunks
    children = [child(c1, "question") for _ in range(5)]
    children += [child(c1, "content"), child(c2, "content")]
    retriever = build_retriever(
        FakeChildStore(children), session.get_bind(), k=2, hybrid=False
    )
    assert ids(retriever.invoke("orice")) == [str(c1.id), str(c2.id)]


def test_hybrid_adds_bm25_hits_by_rrf_on_chunk_id(session: Session, chunks: list[Chunk]):
    c1, c2, c3, c4 = chunks
    store = FakeChildStore(
        [child(c2, "content"), child(c1, "content"), child(c3, "content")]
    )
    bind = session.get_bind()

    # semantic [c2, c1, c3] + BM25 [c4] → RRF: c2 (1/61, primul la egalitate), c4 (1/61), ...
    assert ids(build_retriever(store, bind, k=2).invoke("zxqvortex")) == [
        str(c2.id),
        str(c4.id),
    ]
    assert ids(build_retriever(store, bind, k=2, hybrid=False).invoke("zxqvortex")) == [
        str(c2.id),
        str(c1.id),
    ]


def test_chunks_missing_from_db_are_skipped(session: Session, chunks: list[Chunk]):
    c1, c2, c3, c4 = chunks
    store = FakeChildStore(
        [child(c1, "content"), child(c2, "content"), child(c3, "content")]
    )
    retriever = build_retriever(store, session.get_bind(), k=2)  # BM25 îl conține pe c4
    session.delete(c1)  # vector orfan
    session.delete(c4)  # dispărut din BM25 după construire
    session.commit()

    assert ids(retriever.invoke("zxqvortex")) == [str(c2.id), str(c3.id)]


def test_empty_index_returns_nothing(session: Session):
    retriever = build_retriever(FakeChildStore([]), session.get_bind(), k=3)
    assert retriever.invoke("grâu") == []


def test_k_must_be_positive(session: Session):
    with pytest.raises(ValueError, match="k"):
        build_retriever(FakeChildStore([]), session.get_bind(), k=0)
```

- [ ] **Step 4: Rulează testele, trebuie să pice**

Run: `uv run pytest app/knowledge_prep/tests/test_retrieval.py -v`
Expected: FAIL cu `ModuleNotFoundError: No module named 'app.knowledge_prep.retrieval'`

- [ ] **Step 5: Implementarea** — `backend/app/knowledge_prep/retrieval.py`:

```python
"""Retrieval peste indexul fazei 3 (spec 2026-09-25, „Retrieval”).

- **multi-vector:** `MultiVectorRetriever` peste `knowledge_vectors` (copiii `content` + `question`),
  cu părinții din `ChunkDocStore`;
- **baseline:** același retriever, doar cu copiii `content` (`filter={"kind": "content"}`);
- **hibrid:** oricare dintre cele două + BM25 pe `chunk.text` (`lexical.py`), combinate prin
  reciprocal rank fusion pe `chunk.id`.

Evaluarea, rescrierea query-ului și reranking-ul nu sunt aici (vin după evaluare).
"""

import logging
from collections.abc import Sequence
from typing import Any

import openai
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.knowledge_prep.converters.gemini import log_retry
from app.knowledge_prep.docstore import Bind, ChunkDocStore
from app.knowledge_prep.lexical import ChunkBM25
from app.knowledge_prep.questions import MAX_QUESTIONS

logger = logging.getLogger(__name__)

DEFAULT_K = 5
RRF_K = 60  # constanta din articolul RRF (și din rag_simple)
# un vector `content` + cel mult MAX_QUESTIONS întrebări per chunk: cu k * 6 copii rămân
# cel puțin k părinți distincți, chiar dacă toți copiii unui chunk vin grupați
CHILDREN_PER_CHUNK = MAX_QUESTIONS + 1


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[str]:
    """Id-urile din toate listele, după suma 1 / (k + rang); la egalitate, ordinea primei apariții."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


@retry(
    retry=retry_if_exception_type(openai.APIError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(3),
    before_sleep=log_retry,
    reraise=True,
)
def _semantic_search(semantic: MultiVectorRetriever, query: str) -> list[Document]:
    # embedding-ul întrebării e un apel OpenAI (în PGVectorStore)
    return semantic.invoke(query)


class KnowledgeRetriever(BaseRetriever):
    """Top `k` chunk-uri pentru o întrebare, ca `Document`-uri cu metadatele din `ChunkDocStore`."""

    semantic: MultiVectorRetriever
    docstore: ChunkDocStore
    lexical: ChunkBM25 | None = None
    k: int = DEFAULT_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        semantic_ids = [
            d.metadata["chunk_id"] for d in _semantic_search(self.semantic, query)
        ]
        if self.lexical is None:
            ranked = semantic_ids
        else:
            lexical_ids = self.lexical.search(query, self.k * CHILDREN_PER_CHUNK)
            ranked = reciprocal_rank_fusion([semantic_ids, lexical_ids])
        # BM25 poate avea id-uri de chunk-uri șterse după construirea lui: mget le dă None
        docs = [d for d in self.docstore.mget(ranked) if d is not None]
        return docs[: self.k]


def build_retriever(
    vectorstore: VectorStore,
    bind: Bind,
    *,
    k: int = DEFAULT_K,
    questions: bool = True,
    hybrid: bool = True,
) -> KnowledgeRetriever:
    """`questions=False` = baseline (doar vectorii `content`); `hybrid=True` adaugă BM25.

    `vectorstore` e în producție `PgVectorIndex.from_settings().store`; `bind` e engine-ul
    (sau conexiunea) Postgres. Indexul BM25 se construiește acum, din toate chunk-urile.
    """
    if k < 1:
        raise ValueError(f"k trebuie să fie cel puțin 1 (primit {k})")
    docstore = ChunkDocStore(bind)
    search_kwargs: dict[str, Any] = {"k": k * CHILDREN_PER_CHUNK}
    if not questions:
        search_kwargs["filter"] = {"kind": "content"}
    semantic = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs=search_kwargs,
    )
    return KnowledgeRetriever(
        semantic=semantic,
        docstore=docstore,
        lexical=ChunkBM25.from_db(bind) if hybrid else None,
        k=k,
    )
```

- [ ] **Step 6: Rulează testele, trebuie să treacă**

Run: `uv run pytest app/knowledge_prep/tests/test_retrieval.py -v`
Expected: 8 passed

Dacă `MultiVectorRetriever(...)` refuză `docstore=ChunkDocStore(...)` la validarea pydantic, verifică întâi că `ChunkDocStore` moștenește `BaseStore[str, Document]` (Task 1); nu trece la `InMemoryStore`.

- [ ] **Step 7: Toată suita + lint + commit**

```bash
uv run pytest app/knowledge_prep
uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep
git add backend/pyproject.toml uv.lock backend/app/knowledge_prep/retrieval.py backend/app/knowledge_prep/tests/retrieval_fakes.py backend/app/knowledge_prep/tests/test_retrieval.py
git commit -m "feat(knowledge_prep): multi-vector, baseline and hybrid (BM25 + RRF) retrieval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`uv.lock` e în rădăcina `full-stack-fastapi-template/`: din `backend/`, calea pentru `git add` e `../uv.lock`.)

---

### Task 4: Integrare pe Postgres (`ChunkDocStore`, BM25, `MultiVectorRetriever` pe `PGVectorStore`)

**Files:**
- Create: `backend/app/knowledge_prep/tests/pg.py`
- Modify: `backend/app/knowledge_prep/tests/test_vectors_pg.py` (folosește `tests/pg.py`)
- Test: `backend/app/knowledge_prep/tests/test_retrieval_pg.py`

**Interfaces:**
- Consumes: `ChunkDocStore` (Task 1), `ChunkBM25` (Task 2), `build_retriever` (Task 3), `add_document` (Task 1); `PgVectorIndex(url, embeddings, *, table_name, vector_size)`, `VectorEntry`, `replace_document`, `embed` (faza 3); `embedding_text` din `chunking.py`; `DeterministicFakeEmbedding(size=8)` (același text → același vector, deci distanța 0 pentru query identic).
- Produces: `tests/pg.py` cu `PG_URL: str | None`, `requires_pg` (marker `skipif`), `has_table(name: str) -> bool` (False fără Postgres), `drop_table(name: str) -> None`.

- [ ] **Step 1: Helper-ul comun** — `backend/app/knowledge_prep/tests/pg.py` (conținutul lui `_pg_url` se mută din `test_vectors_pg.py`, neschimbat):

```python
"""Baza Postgres din `.env` pentru testele de integrare; se sar dacă nu răspunde (`docker compose up -d db`)."""

import pytest
from sqlalchemy import create_engine, inspect, text


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
requires_pg = pytest.mark.skipif(
    PG_URL is None, reason="Postgres cu pgvector indisponibil (docker compose up -d db)"
)


def has_table(name: str) -> bool:
    if PG_URL is None:
        return False
    engine = create_engine(PG_URL)
    try:
        return inspect(engine).has_table(name)
    finally:
        engine.dispose()


def drop_table(name: str) -> None:
    assert PG_URL is not None
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
    engine.dispose()
```

În `test_vectors_pg.py`: șterge `_pg_url`, `PG_URL = _pg_url()` și `pytestmark = pytest.mark.skipif(...)`; pune în loc

```python
from app.knowledge_prep.tests.pg import PG_URL, drop_table, requires_pg

pytestmark = requires_pg
```

iar în fixture-ul `pg`, blocul de după `yield` devine `drop_table(table)`. Scoate importurile rămase nefolosite (`ruff check` le arată).

- [ ] **Step 2: Verifică refactorizarea**

Run: `uv run pytest app/knowledge_prep/tests/test_vectors_pg.py -v`
Expected: 3 passed (cu `docker compose up -d db` pornit) sau 3 skipped (fără Postgres) — același rezultat ca înainte.

- [ ] **Step 3: Testele de integrare** — `backend/app/knowledge_prep/tests/test_retrieval_pg.py`:

```python
"""Integrare pe Postgres (spec, „Testare”): ChunkDocStore, BM25 și MultiVectorRetriever pe date reale.

Scrie un document `retrieval-test-<hex>` în tabelele reale `document`/`chunk` (șters la final,
chunk-urile în cascadă) și un tabel propriu `knowledge_vectors_test_<hex>` (șters la final).
Nu atinge `knowledge_vectors`. Se sare fără Postgres sau fără migrarea fazei 1.
"""

import uuid
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import Engine, create_engine
from sqlmodel import Session

from app.knowledge_prep import models
from app.knowledge_prep.chunking import embedding_text
from app.knowledge_prep.docstore import ChunkDocStore
from app.knowledge_prep.lexical import ChunkBM25
from app.knowledge_prep.models import Chunk
from app.knowledge_prep.retrieval import build_retriever
from app.knowledge_prep.tests.pg import PG_URL, drop_table, has_table, requires_pg
from app.knowledge_prep.tests.retrieval_fakes import add_document
from app.knowledge_prep.vectors import PgVectorIndex, VectorEntry

pytestmark = [
    requires_pg,
    pytest.mark.skipif(
        not has_table("chunk"), reason="lipsește tabelul chunk (alembic upgrade head)"
    ),
]
SIZE = 8
QUESTION = "Ce cereală se macină pentru testul de retrieval?"


def ids(docs: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in docs]


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert PG_URL is not None
    e = create_engine(PG_URL)
    yield e
    e.dispose()


@pytest.fixture
def chunks(engine: Engine) -> Iterator[list[Chunk]]:
    slug = f"retrieval-test-{uuid.uuid4().hex[:8]}"
    with Session(engine) as s:
        made = add_document(
            s,
            slug,
            [
                (["Test", "Ovăz"], "Ovăzul zxqvortex se fierbe încet."),
                (["Test", "Orz"], "Orzul se macină."),
                (["Test", "Mei"], "Meiul se spală bine."),
            ],
        )
        s.expunge_all()
    try:
        yield made
    finally:
        with Session(engine) as s:
            doc = s.get(models.Document, made[0].document_id)
            if doc is not None:
                s.delete(doc)  # chunk-urile: ON DELETE CASCADE
                s.commit()


@pytest.fixture
def index(chunks: list[Chunk]) -> Iterator[PgVectorIndex]:
    assert PG_URL is not None
    table = f"knowledge_vectors_test_{uuid.uuid4().hex[:8]}"
    idx = PgVectorIndex(
        PG_URL, DeterministicFakeEmbedding(size=SIZE), table_name=table, vector_size=SIZE
    )
    document_id = chunks[0].document_id
    entries = [
        VectorEntry(c.id, c.id, document_id, "content", embedding_text(c.heading_path, c.text))
        for c in chunks
    ]
    entries.append(VectorEntry(uuid.uuid4(), chunks[1].id, document_id, "question", QUESTION))
    idx.replace_document(document_id, entries, idx.embed([e.text for e in entries]))
    try:
        yield idx
    finally:
        drop_table(table)


def test_docstore_reads_postgres_rows_and_uuid_keys(engine: Engine, chunks: list[Chunk]):
    a = chunks[0]
    got = ChunkDocStore(engine).mget([a.id, "nu-e-uuid", str(uuid.uuid4())])  # type: ignore[list-item]
    assert got[1:] == [None, None]
    first = got[0]
    assert first is not None
    assert first.page_content == "Ovăzul zxqvortex se fierbe încet."
    assert first.metadata["heading_path"] == ["Test", "Ovăz"]
    assert first.metadata["slug"].startswith("retrieval-test-")


def test_bm25_from_postgres_finds_rare_word(engine: Engine, chunks: list[Chunk]):
    assert ChunkBM25.from_db(engine).search("zxqvortex", 3) == [str(chunks[0].id)]


def test_multi_vector_finds_chunk_through_its_question(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    retriever = build_retriever(index.store, engine, k=1, hybrid=False)
    assert ids(retriever.invoke(QUESTION)) == [str(chunks[1].id)]


def test_baseline_finds_chunk_by_its_content_vector(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    target = chunks[2]
    retriever = build_retriever(index.store, engine, k=1, questions=False, hybrid=False)
    query = embedding_text(target.heading_path, target.text)
    assert ids(retriever.invoke(query)) == [str(target.id)]


def test_hybrid_brings_in_the_bm25_hit(
    engine: Engine, index: PgVectorIndex, chunks: list[Chunk]
):
    retriever = build_retriever(index.store, engine, k=2)
    assert str(chunks[0].id) in ids(retriever.invoke("zxqvortex"))
```

- [ ] **Step 4: Rulează testele** (cu `docker compose up -d db` din rădăcina full-stack)

Run: `uv run pytest app/knowledge_prep/tests/test_retrieval_pg.py app/knowledge_prep/tests/test_vectors_pg.py -v`
Expected: 8 passed. Fără Postgres: 8 skipped.

Dacă `test_multi_vector_finds_chunk_through_its_question` pică cu listă goală, cauza probabilă e tipul lui `doc_id` (uuid vs. str) între `PGVectorStore` și `mget` — repară în `docstore._as_uuid`, nu în test.

Verifică și că nu rămân resturi: `uv run python -c "from sqlalchemy import create_engine, text; from app.core.config import settings; e=create_engine(str(settings.DATABASE_URL)); print(e.connect().execute(text(\"select count(*) from document where slug like 'retrieval-test-%'\")).scalar())"` → `0`.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep
git add backend/app/knowledge_prep/tests/pg.py backend/app/knowledge_prep/tests/test_vectors_pg.py backend/app/knowledge_prep/tests/test_retrieval_pg.py
git commit -m "test(knowledge_prep): retrieval integration on Postgres (docstore, BM25, multi-vector)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CLI `search "<întrebare>" [-k N] [--no-questions] [--no-hybrid]`

**Files:**
- Modify: `backend/app/knowledge_prep/__main__.py`
- Modify: `backend/app/knowledge_prep/tests/test_cli.py`

**Interfaces:**
- Consumes: `build_retriever`, `DEFAULT_K` (Task 3); `FakeChildStore.from_entries` (Task 3); `PgVectorIndex.from_settings().store` (faza 3); `FakeVectors.rows` (faza 3, `dict[uuid.UUID, VectorEntry]`).
- Produces: `main(argv, *, session=None, converters=..., restructurer=..., vectors=None, questioner=..., vectorstore: VectorStore | None = None) -> int`; output pe stdout, câte un rezultat pe două rânduri: `"{i}. {slug} — {heading_path unit cu ' > '}"` și `"   {primele 200 de caractere, spațiile comprimate}"`; `"niciun rezultat"` când lista e goală.

- [ ] **Step 1: Testele** — în `backend/app/knowledge_prep/tests/test_cli.py`:

Importuri noi:

```python
from langchain_core.vectorstores import VectorStore

from app.knowledge_prep.tests.retrieval_fakes import FakeChildStore
```

`run()` primește și `vectorstore` (restul neschimbat):

```python
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
```

Teste noi, la final:

```python
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
        session, "search", "zxqvortex", "--no-questions", "--no-hybrid", vectorstore=store
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
```

Verifică la Step 4 că secțiunile au peste `MIN_CHUNK_TOKENS` (80) și rezultă exact 3 chunk-uri (`# Carte` fără text se unește cu `## Grâul`, cu calea `["Carte", "Grâul"]`); dacă nu, mărește multiplicatorul din `section`, nu schimba chunking-ul.

- [ ] **Step 2: Rulează testele, trebuie să pice**

Run: `uv run pytest app/knowledge_prep/tests/test_cli.py -v`
Expected: FAIL — `main() got an unexpected keyword argument 'vectorstore'` (toate testele care folosesc `run`).

- [ ] **Step 3: Implementarea** — în `backend/app/knowledge_prep/__main__.py`:

Docstring-ul modulului: prima linie devine `"""CLI-ul pipeline-ului de cunoștințe: register → prepare → review → index → validate, plus status și search.`

Importuri noi:

```python
from langchain_core.vectorstores import VectorStore

from app.knowledge_prep.retrieval import DEFAULT_K, build_retriever
```

Constantă și validator, după `logger = ...`:

```python
SNIPPET_CHARS = 200


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"trebuie să fie cel puțin 1 (primit {n})")
    return n
```

În `build_parser`, înainte de `return p`:

```python
    srch = sub.add_parser(
        "search", help="caută chunk-uri pentru o întrebare (verificare manuală)"
    )
    srch.add_argument("query", help="întrebarea, între ghilimele")
    srch.add_argument(
        "-k", type=_positive_int, default=DEFAULT_K, help="câte chunk-uri (implicit 5)"
    )
    srch.add_argument(
        "--questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-questions: doar vectorii 'content' (baseline)",
    )
    srch.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-hybrid: fără BM25",
    )
```

Funcția nouă, după `_index`:

```python
def _search(
    args: argparse.Namespace, session: Session, vectorstore: VectorStore | None
) -> int:
    query = args.query.strip()
    if not query:
        logger.error("EROARE: întrebarea e goală")
        return 1
    try:
        retriever = build_retriever(
            vectorstore
            if vectorstore is not None
            else PgVectorIndex.from_settings().store,
            session.get_bind(),
            k=args.k,
            questions=args.questions,
            hybrid=args.hybrid,
        )
        docs = retriever.invoke(query)
    except Exception as e:
        logger.exception("EROARE: %s", e)
        return 1
    if not docs:
        sys.stdout.write("niciun rezultat\n")
        return 0
    for i, d in enumerate(docs, 1):
        m = d.metadata
        snippet = " ".join(d.page_content.split())[:SNIPPET_CHARS]
        sys.stdout.write(
            f"{i}. {m['slug']} — {' > '.join(m['heading_path'])}\n   {snippet}\n"
        )
    return 0
```

`_dispatch` primește `vectorstore: VectorStore | None` (după `questioner`) și are un caz nou, **înainte** de `case _:`:

```python
        case "search":
            return _search(args, session, vectorstore)
```

`main` primește `vectorstore: VectorStore | None = None` (ultimul parametru keyword) și îl dă mai departe în ambele apeluri `_dispatch(...)`.

- [ ] **Step 4: Rulează testele, trebuie să treacă**

Run: `uv run pytest app/knowledge_prep -v`
Expected: toate trec (testele Postgres: passed sau skipped).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check app/knowledge_prep && uv run ruff format app/knowledge_prep && uv run ty check app/knowledge_prep
git add backend/app/knowledge_prep/__main__.py backend/app/knowledge_prep/tests/test_cli.py
git commit -m "feat(knowledge_prep): CLI search (baseline/multi-vector, optional BM25 hybrid)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Verificare reală (manuală, opțională — cere `OPENAI_API_KEY` și Postgres cu datele fazei 3)

Doar citește din DB (plus un embedding OpenAI per căutare, cost neglijabil). Nu scrie în registru sau în corpus. La 2026-09-29, baza de dev are 36 de chunk-uri și 178 de vectori.

- [ ] **Step 1: Cele patru combinații pe aceeași întrebare** — din `backend/`:

```bash
Q="Cine nu are voie să mănânce grâu încolțit?"
uv run python -m app.knowledge_prep search "$Q" --no-questions --no-hybrid   # baseline pur
uv run python -m app.knowledge_prep search "$Q" --no-hybrid                  # multi-vector
uv run python -m app.knowledge_prep search "$Q" --no-questions               # baseline + BM25
uv run python -m app.knowledge_prep search "$Q"                              # multi + BM25
```

- [ ] **Step 2: Fără diacritice și cu un cuvânt-cheie rar** (BM25 trebuie să ajute):

```bash
uv run python -m app.knowledge_prep search "graul incoltit contraindicatii"
uv run python -m app.knowledge_prep search "<un termen rar din corpus, ales de user>" -k 3
```

- [ ] **Step 3: Raport către user** — ce chunk-uri apar în fiecare mod, diferențele observate, orice rezultat evident greșit. **Nu e evaluarea** (aceea vine cu setul de întrebări de test); doar confirmă că modurile funcționează cap-coadă pe datele reale. Observațiile intră în README la Task 7 („Rezultate reale”).

---

### Task 7: Docs

**Files:**
- Modify: `backend/app/knowledge_prep/README.md`
- Modify: `docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md` („Retrieval”, „Faze de implementare”)
- Modify: `/home/alex/projects/RAG/AGENTS.md` (în afara oricărui repo git, nu se comite)
- Modify: `/home/alex/projects/RAG/calea_de_autovindecare/docs/taskuri.md` (**repo git separat**)

- [ ] **Step 1: README-ul modulului**
  - Lista de planuri de sus: adaugă `2026-09-29-knowledge-retrieval-plan.md` pentru faza 4 (retrieval).
  - „Starea fazelor”: faza 4 → `retrieval ✅ (<data>), evaluarea în așteptare`.
  - „Comenzi”: `uv run python -m app.knowledge_prep search "<întrebare>" [-k N] [--no-questions] [--no-hybrid]` (cere `OPENAI_API_KEY` pentru embedding-ul întrebării).
  - „Harta modulului”: `docstore.py`, `lexical.py`, `retrieval.py`, `tests/retrieval_fakes.py`, `tests/pg.py`.
  - Secțiune nouă „Retrieval (`docstore.py`, `lexical.py`, `retrieval.py`)”, în stilul secțiunii „Indexarea”, cu:
    - ce rezolvă și cele patru combinații (`questions` × `hybrid`), cu `build_retriever(vectorstore, bind, k=, questions=, hybrid=)` ca punct de intrare pentru cod (evaluare, REST, un viitor agent);
    - cum funcționează: `MultiVectorRetriever` cu `id_key="doc_id"` și `k * CHILDREN_PER_CHUNK` copii (de ce 6), `ChunkDocStore` doar-citire (de ce acceptă și `uuid.UUID`), BM25 în memorie pe `chunk.text`, RRF pe `chunk.id` cu `RRF_K = 60`, trunchiere la `k`;
    - de ce BM25 cu `rank_bm25` și nu `hybrid_search_config` (rezumatul deciziei 2 din plan) și tokenizarea fără diacritice;
    - limitări: fără stemming (`grâu` ≠ `grâul` la BM25); indexul BM25 nu vede chunk-uri noi până la un nou `build_retriever` (CLI-ul îl reface la fiecare rulare); un cuvânt prezent în ≥ jumătate din chunk-uri are scor BM25 ≤ 0 (irelevant la corpus real, relevant în teste mici); documentele readuse în `structured` rămân căutabile până la următorul `index`; fără filtrare după stadiu;
    - „Rezultate reale” din Task 6 (sau „de completat” dacă Task 6 nu s-a rulat);
    - tabel „Cum se reglează”: `DEFAULT_K`, `CHILDREN_PER_CHUNK`, `RRF_K`, `tokenize`.
  - „Testare”: `FakeChildStore` (copii în ordine fixă, filtru pe egalitate, înregistrează apelurile) și `add_document`; `main(...)` primește și `vectorstore=` (dă-l fake în teste; `run()` din `test_cli.py` o face); `test_retrieval_pg.py` scrie un document `retrieval-test-*` în `document`/`chunk` reale și îl șterge, plus un tabel `knowledge_vectors_test_*`.
- [ ] **Step 2: Spec-ul**
  - „Faze de implementare”, punctul 4: `4. **Retrieval + evaluare:** ... ✅ retrieval <data> (evaluarea în așteptare)`.
  - La finalul secțiunii „Retrieval”, o notă ca la fazele 2 și 3:
    `> **Implementat (<data>) — doar retrieval-ul; evaluarea rămâne pentru după setul de întrebări de test.** Modulele: docstore.py, lexical.py, retrieval.py în knowledge_prep. Hibridul folosește BM25 cu rank_bm25, în memorie, peste chunk.text, fără diacritice (nu hybrid_search_config: acela lucrează pe copii, cu ts_rank și sumă ponderată). Detalii: backend/app/knowledge_prep/README.md.`
- [ ] **Step 3: `RAG/AGENTS.md`** — în blocul „Comenzi”, sub `index`: `uv run python -m app.knowledge_prep search "<întrebare>" [-k N] [--no-questions] [--no-hybrid]  # verificare manuală a retrieval-ului (baseline / multi-vector / + BM25)`.
- [ ] **Step 4: Commit în `full-stack-fastapi-template`**

```bash
git add backend/app/knowledge_prep/README.md docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md
git commit -m "docs(knowledge_prep): retrieval (phase 4) in module README; mark retrieval done in spec

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: `taskuri.md` (repo separat, `calea_de_autovindecare`)**
  - Rândul 5 („Stare implementare”): fazele 1-3 gata; din faza 4, retrieval-ul (baseline, multi-vector, BM25 + RRF, comanda `search`) e gata, evaluarea urmează.
  - La „### 3. …”, rândul „Fără BM25 încă — vine în faza 4”: BM25 există acum (în memorie, din tabelul `chunk`, la retrieval).
  - La „### 4. Retrieval + evaluare”: adaugă `- [x] Retrieval: baseline / multi-vector / hibrid BM25 + RRF (comanda `search`); detalii în README-ul knowledge_prep.`; „Evaluare retrieval pe noul model” rămâne nebifată.

```bash
git -C /home/alex/projects/RAG/calea_de_autovindecare status --short docs/taskuri.md
```

La 2026-09-29 fișierul are modificări necomise ale userului (`MM docs/taskuri.md`). **Nu comite peste ele fără acord:** dacă `git -C /home/alex/projects/RAG/calea_de_autovindecare diff docs/taskuri.md` arată și altceva decât schimbările de la acest pas, întreabă userul dacă le comiți împreună. Altfel:

```bash
git -C /home/alex/projects/RAG/calea_de_autovindecare add docs/taskuri.md
git -C /home/alex/projects/RAG/calea_de_autovindecare commit -m "docs(taskuri): phase 4 retrieval done, evaluation pending

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Întrebări deschise (nu blochează planul)

1. **Stemming pentru BM25 în română** (ex. `snowballstemmer`, pur Python, are română): ar potrivi `grâu`/`grâul`/`grâului`. E o dependență nouă; se decide după evaluare, dacă BM25 ratează forme flexionate. `tokenize` e singurul loc de schimbat.
2. **BM25 pentru un retriever cu viață lungă** (server LangGraph, REST): azi indexul se construiește la `build_retriever`. Dacă apare un proces de durată, fie se reconstruiește după fiecare `index`, fie la fiecare N minute; decizia ține de faza 5 / agent.
3. **Titlul documentului în BM25:** spec-ul cere BM25 pe `chunk.text`; chunk-urile unei cărți conțin doar titlurile lor, nu titlul cărții. Dacă evaluarea arată că lipsa lui contează, se poate indexa `embedding_text(heading_path, text)` în loc de `chunk.text`.
