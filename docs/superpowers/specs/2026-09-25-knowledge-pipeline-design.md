# Pipeline de cunoștințe: registru, restructurare, indexare multi-vector — design

Data: 2026-09-25
Status: draft, în review la user

## Context și motivație

`backend/app/knowledge_prep/` convertește azi sursele (PDF, audio, `.txt`/`.md`) în
`<folderul sursei>/converted/<slug>.md`, cu front-matter și `status: draft | reviewed`
(`calea_de_autovindecare/docs/taskuri.md`, taskul 1). Drumul spre DB e încă cel din
`rag_simple/ingest.py`: citește `*.md` dintr-un folder fix, împarte doar pe `###`-`#####`,
scrie în `rag_simple_chunks` + `chunks.json` și mută fișierele în `ingested/`.

Problemele pe care le rezolvă designul:

- **Căile fixe și stadiul greu de urmărit.** Nu se vede într-un loc ce documente există,
  în ce stadiu sunt și dacă fișierul s-a schimbat după ultimul pas.
- **Textul nu e curățat și structurat.** `text.py` face doar curățare mecanică; fișiere ca
  `carti/Graul_incoltit.md` nu au diacritice, au glyph-uri de font în loc de buline și
  titluri lipite în text.
- **Un singur vector per chunk.** Pentru regimuri (documente întregi, ~3k tokeni) și pentru
  pasaje formulate altfel decât întrebările utilizatorilor, vrem mai multe întrebări care
  duc la același chunk (multi-vector retrieval, decizia 3 din `taskuri.md`).

Spike-ul din 2026-09-25 a rulat promptul de restructurare (secțiunea „Restructurarea”) pe
`Graul_incoltit.md` cu Gemini 2.5 Flash fără thinking: 12 s, 0,77% text lipsă și 0,73%
adăugat (verificarea pe litere din `pdf.py`), diacritice, liste și titluri recuperate
corect. Userul a acceptat rezultatul ca suficient de bun.

## Decizii

1. **Fișierele rămân pe disc; DB-ul e registrul.** Originalul (opțional) și markdown-ul
   curățat și structurat stau pe disc și se editează în IDE. Postgres ține registrul
   (id, stadiu, căi, hash-uri), chunk-urile și întrebările.
2. **Registrul decide stadiul.** Front-matter-ul nu mai are `status`. El se scrie la
   `prepare` doar pentru lizibilitate; pipeline-ul îl ignoră (se scoate la indexare).
3. **Originalul e opțional.** Un document poate intra direct ca markdown deja structurat
   (ex. extrasele WhatsApp din `whatsapp/extracted/`). Fișierul rămâne unde e; registrul
   reține calea lui.
4. **Indexarea și întrebările sunt un singur pas.** Fiecare chunk are ca vectori-copii
   propriul text (`kind=content`) și 1-N întrebări (`kind=question`), generate de LLM.
   `--no-questions` indexează doar `content`: acesta e baseline-ul.
5. **`MultiVectorRetriever` din `langchain_classic`**, cu copiii în `PGVectorStore` și
   părinții (chunk-urile) într-un tabel propriu `chunk`, expus printr-un `BaseStore` mic.
   Nu folosim `SQLStore` din `langchain-community` (pachet în sunset).
6. **Documentele „fără chunk”** (`split=false`: regimuri, note etc.) devin un singur
   chunk cu 1-N întrebări. Același cod ca pentru cărți.
7. **Chunk-urile și întrebările stau doar în DB** și se editează prin REST (faza 5).
   Reindexarea șterge chunk-urile, întrebările și vectorii vechi ai documentului și îi
   creează din nou; editările manuale ale întrebărilor se pierd (comanda avertizează).
   *(Revizuit 2026-09-30: reindexarea e incrementală: se refac doar chunk-urile schimbate,
   iar întrebările celor neschimbate rămân. Vezi „Indexarea”.)*
8. **Întrebările se fac embedding imediat**, nu după review. Editarea unei întrebări
   reface doar vectorul ei.
9. **Metadate minime.** Fără `recommended_for`, `tags` etc.: nu se potrivesc textelor
   filozofice și complică review-ul. Cuvintele-cheie le acoperă BM25.
10. **Întrebările de test** se extrag separat, într-o fază ulterioară (manual sau cu LLM),
    și nu se indexează. Nu fac parte din acest design.
11. `rag_simple` rămâne neatins, ca exemplu de referință.

## Modelul de date

Modele SQLModel în `app/knowledge_prep/models.py`, importate în `app/alembic/env.py`
alături de `app.models`, cu o migrare Alembic nouă.

### `document`

| Coloană | Tip | Note |
|---|---|---|
| `id` | UUID, PK | |
| `slug` | str, unic | din titlu, fără diacritice (`frontmatter.slugify`) |
| `title` | str | |
| `author` | str \| null | |
| `type` | enum | `carte \| regim \| nota \| insight \| transcriere \| conversatie` |
| `split` | bool | `false` = documentul întreg e un singur chunk |
| `stage` | enum | vezi „Stadii” |
| `original_path` | str \| null | absolut; null dacă documentul intră deja structurat |
| `markdown_path` | str | absolut; fișierul editat în IDE |
| `reviewed_hash` | str \| null | sha256 al markdown-ului la `review` |
| `indexed_hash` | str \| null | sha256 al markdown-ului la `index` |
| `last_error` | str \| null | eroarea ultimului pas eșuat; stadiul nu se schimbă |
| `created_at`, `updated_at` | datetime | |

Hash-ul se calculează pe corpul markdown-ului, fără front-matter.

### `chunk`

| Coloană | Tip | Note |
|---|---|---|
| `id` | UUID, PK | e și `doc_id` din metadatele vectorilor-copii |
| `document_id` | FK → `document`, `ON DELETE CASCADE` | |
| `position` | int | ordinea în document |
| `heading_path` | JSON (listă de str) | ex. `["Grâul încolțit", "Contraindicații ale curei cu grâu încolțit"]` |
| `text` | text | chunk-ul, cu headerele lui |
| `content_hash` | varchar(64), null | sha256 pe `embedding_text`; identifică chunk-ul la reindexare (2026-09-30) |
| `indexed_at` | timestamptz, null | pus după ce întrebările și vectorii chunk-ului sunt scriși; `null` = neterminat (2026-09-30) |

### `question`

| Coloană | Tip | Note |
|---|---|---|
| `id` | UUID, PK | e și id-ul vectorului-copil |
| `chunk_id` | FK → `chunk`, `ON DELETE CASCADE` | |
| `text` | str | |

### Tabelul de vectori

`PGVectorStore` pe un tabel nou, `knowledge_vectors` (nu `rag_simple_chunks`), cu același
model de embedding ca `rag_simple` (`openai:text-embedding-3-large`, 3072 dimensiuni).
Coloane de metadate filtrabile (`metadata_columns`): `doc_id` (chunk), `document_id`,
`kind` (`content | question`). Id-ul vectorului e `chunk.id` pentru `content` și
`question.id` pentru `question`, ca ștergerea și înlocuirea să se facă după id.

Textul embedded pentru `content` e `heading_path` unit cu ` > ` + rând gol + `chunk.text`.
Pentru `question` e doar textul întrebării.

## Stadii

```
uploaded ──prepare──> structured ──review──> reviewed ──index──> indexed ──validate──> validated
                           ▲
register --structured ─────┘
```

| Stadiu | Înseamnă | Se ajunge prin |
|---|---|---|
| `uploaded` | originalul e înregistrat, nu există markdown | `register <original>` |
| `structured` | markdown-ul există, nerevizuit | `prepare` sau `register --structured <md>` |
| `reviewed` | userul a validat markdown-ul; `reviewed_hash` salvat | `review` |
| `indexed` | chunk-uri, întrebări și vectori în DB; `indexed_hash` salvat | `index` |
| `validated` | userul a revizuit întrebările | `validate` |

Reguli:

- `index` cere `stage >= reviewed` și hash-ul curent egal cu `reviewed_hash`. Altfel
  refuză cu mesajul „markdown modificat după review; rulează din nou `review`”.
- `status` compară hash-ul curent cu `reviewed_hash` și `indexed_hash` și afișează
  „modificat după review” / „modificat după indexare”.
- `prepare` pe un document `reviewed` sau mai departe cere `--force` și îl readuce în
  `structured`. Chunk-urile existente rămân în DB până la următorul `index`.
- `prepare` pe un document fără original eșuează cu mesaj clar.
- Un pas eșuat scrie `last_error` și nu schimbă stadiul; un pas reușit golește `last_error`.

## Comenzi (CLI)

Un singur punct de intrare cu subcomenzi, rulat din `backend/`:

```
uv run python -m app.knowledge_prep <comandă> <țintă> [opțiuni]
```

`<țintă>` poate fi id-ul, slug-ul sau calea (a originalului ori a markdown-ului).
`register` și `prepare` acceptă și un folder, parcurs recursiv ca în `convert.py`, unde
un fișier eșuat nu oprește restul.

| Comandă | Ce face |
|---|---|
| `register <cale> [--type --title --author --no-split]` | înregistrează originalul → `uploaded` |
| `register --structured <md> [...]` | înregistrează un markdown deja structurat → `structured`; titlul, tipul și autorul se preiau din front-matter dacă există, altfel din opțiuni sau din numele fișierului |
| `prepare <țintă> [--restructure/--no-restructure] [--force]` | conversia (convertoarele existente) + restructurarea → `<folderul originalului>/converted/<slug>.md` → `structured` |
| `review <țintă>` | → `reviewed`, salvează `reviewed_hash` |
| `index <țintă> [--no-questions]` | chunking + întrebări + embeddings → `indexed` |
| `validate <țintă>` | → `validated` |
| `status [<țintă>]` | tabel cu documentele: slug, tip, stadiu, avertismente de hash, `last_error` |

`--type` implicit: `transcriere` pentru audio, altfel `carte`. `split` implicit: `false`
pentru `regim`, `nota`, `insight`, `true` pentru restul; `--split/--no-split` îl suprascrie.

Funcția `convert(...)` din `convert.py` se păstrează ca bibliotecă pentru `prepare`.
Regulile ei de protecție (`reviewed` nu se suprascrie, slug unic) se mută în registru, iar
CLI-ul `convert` e înlocuit de `prepare`. Fișierele deja convertite din `converted/` intră
în registru cu `register --structured`.

Registrul e vizibil și în Adminer (`compose.yml`, portul `8080`).

## Restructurarea

Modul nou `app/knowledge_prep/restructure.py`, apelat de `prepare`.

- **Când:** implicit pentru `.txt`, `.md` și audio; nu pentru PDF (promptul din `pdf.py`
  structurează deja). `--restructure/--no-restructure` suprascrie.
- **Model:** `GEMINI_MODEL` din `.env` (azi Gemini 2.5 Flash), prin
  `converters/gemini.py::generate` (fără thinking, retry cu `stop_after_attempt(5)`).
- **Promptul** e cel din spike-ul din 2026-09-25: corectează diacriticele, caracterele
  stricate și greșelile evidente de tipar; nu schimbă sensul, stilul, numerele, numele sau
  ordinea; structurează în `#`/`##`/`###`/`####` cu titluri care se înțeleg și singure;
  liste markdown; marcaje `<!-- VERIFICĂ: ... -->` pentru text mutat, trunchiat sau ilizibil.
- **Documente mari:** ferestre de ~8k tokeni, tăiate la granițe de paragraf. Fiecare apel
  primește calea de titluri de la finalul ferestrei anterioare, ca ierarhia să continue.
- **Verificare:** `pdf.missing_fraction` în ambele sensuri (lipsă față de original,
  adăugat față de original), prag 1% pe fiecare, per fereastră. La depășire se reîncearcă
  o dată; apoi se păstrează încercarea mai bună, precedată de
  `<!-- VERIFICĂ: fereastra N, lipsă X%, adăugat Y% -->`.

> **Implementat (2026-09-28)**, cu două precizări la verificare, venite din testul real:
> titlurile nu se numără ca text „adăugat” (regula 6 le cere), iar n-gramele se calculează pe
> fiecare paragraf separat, ca literele din jurul unui titlu scos din calcul să nu se lipească în
> n-grame false. Detalii, rezultate și limitări: `backend/app/knowledge_prep/README.md`.

## Indexarea

Modul nou `app/knowledge_prep/index.py`, apelat de `index`.

1. Citește markdown-ul, scoate front-matter-ul și verifică hash-ul (vezi „Stadii”).
2. **Chunking:**
   - `split=false`: un singur chunk cu tot corpul; `heading_path` = titlul documentului.
     Avertizează dacă depășește 8000 de tokeni (limita modelului de embedding e 8191).
   - `split=true`: `MarkdownHeaderTextSplitter` pe `#`-`####`, `strip_headers=False`, din
     care se construiește `heading_path`. Secțiunile sub `MIN_CHUNK_TOKENS` (80) se unesc
     cu următoarea secțiune din același părinte. Secțiunile peste `MAX_CHUNK_TOKENS` (1000)
     se împart cu `RecursiveCharacterTextSplitter` (overlap 150), păstrând `heading_path`.
     Marcajele `<!-- VERIFICĂ -->` se scot.
3. **Întrebări** (dacă nu e `--no-questions`): un apel Gemini per chunk, cu output
   structurat (listă de str), primind titlul documentului, `heading_path` și textul.
   Instrucțiuni: 1-5 întrebări în română, la care chunk-ul răspunde complet; formulate cum
   ar întreba un om; care se înțeleg fără context (numesc subiectul, ex. „grâul încolțit”,
   nu „acesta”); pentru `split=false`, întrebări care acoperă tot documentul. Retry ca la
   restructurare.
4. **Scriere**, într-o singură tranzacție pentru tabelele relaționale:
   - șterge vectorii documentului (`PGVectorStore.delete` cu id-urile chunk-urilor și
     întrebărilor existente), apoi chunk-urile (întrebările se șterg în cascadă);
   - inserează chunk-urile și întrebările noi;
   - adaugă vectorii-copii;
   - `stage=indexed`, `indexed_hash=reviewed_hash`.

   Dacă embedding-ul eșuează după commit-ul relațional, documentul rămâne cu stadiul
   anterior și `last_error`; o nouă rulare `index` curăță și reface totul.
5. Dacă documentul era `validated`, comanda cere `--force` și avertizează că întrebările
   editate manual se pierd.

> **Implementat (2026-09-28)**, cu precizări: parser propriu de titluri în loc de
> MarkdownHeaderTextSplitter (păstrează textul exact); embeddings înainte de scrierea
> relațională și ștergerea vectorilor după document_id; knowledge_vectors creat de PGEngine,
> nu de Alembic. Detalii: backend/app/knowledge_prep/README.md.

> **Revizuit (2026-09-30): indexare incrementală și reluabilă.** O rulare întreruptă (Ctrl+C în
> notebook, după 74 de apeluri Gemini) pierdea tot, pentru că scrierea se făcea abia la final.
> Acum:
> - chunk-urile noi se compară cu cele din DB după `content_hash`. Un chunk identic, cu
>   `indexed_at` pus și făcut în același mod (cu/fără întrebări), se păstrează, cu întrebările
>   lui. Restul chunk-urilor vechi se șterg, cu vectorii lor (`delete_chunks`, după `doc_id`);
> - chunk-urile de refăcut se procesează în loturi de `BATCH_SIZE` (10): întrebări, embeddings,
>   `chunk` + `question`, vectori, apoi `indexed_at`, cu commit per lot;
> - `stage=indexed` abia când toate chunk-urile au `indexed_at`. Un eșec sau o întrerupere
>   păstrează loturile terminate, iar rularea următoare reia de acolo;
> - `--force` reface toate chunk-urile. Regula pentru `validated` (cere `--force`) rămâne.
>
> Migrarea `3e8d5a1f4c27` completează coloanele pentru chunk-urile existente.

## Retrieval

Modul nou (numele se stabilește în planul fazei 4), fără modificări în `rag_simple`.

- `ChunkDocStore(BaseStore[str, Document])`: `mget` citește din tabelul `chunk` și întoarce
  `Document(page_content=text, metadata={chunk_id, document_id, slug, title, heading_path})`.
  `mset`/`mdelete` ridică `NotImplementedError`: scrierea se face doar prin indexare.
- `MultiVectorRetriever(vectorstore=knowledge_vectors, docstore=ChunkDocStore(), id_key="doc_id")`.
  `search_kwargs["k"]` pentru copii e mai mare decât numărul de chunk-uri dorit (mai multe
  întrebări ale aceluiași chunk se reduc la un singur părinte).
- **Baseline vs. multi-vector:** același retriever, cu `search_kwargs["filter"] = {"kind": "content"}`
  pentru baseline.
- **Hibrid:** BM25 peste `chunk.text`, combinat cu rezultatele semantice prin reciprocal
  rank fusion pe `chunk.id`. Rescrierea query-ului și reranking-ul se preiau din
  `rag_simple` doar dacă evaluarea arată că ajută. `PGVectorStore` are și căutare hibridă
  proprie (`hybrid_search_config`, full-text Postgres), dar ea lucrează pe copii, nu pe
  chunk-uri; alegerea între ea și BM25 pe `chunk` se face în planul fazei 4.
- **Evaluare:** metricile din `rag_simple/evaluation/retrieval_metrics.py` pe un set de
  întrebări de test, rulate pe baseline și pe multi-vector. Setul de test nou vine din
  faza separată de extragere a întrebărilor (decizia 10).

> **Implementat (2026-09-29) — doar retrieval-ul; evaluarea rămâne pentru după setul de
> întrebări de test.** Modulele: `docstore.py`, `lexical.py`, `retrieval.py` în
> `knowledge_prep`. Hibridul folosește BM25 cu `rank_bm25`, în memorie, peste `chunk.text`,
> fără diacritice (nu `hybrid_search_config`: acela lucrează pe copii, cu `ts_rank` și sumă
> ponderată). Detalii: `backend/app/knowledge_prep/README.md`.

> **Evaluarea, implementată (2026-09-30):** `evaluation.py` + comanda `evaluate`. Rulează cele
> patru moduri (baseline / multi-vector, cu și fără BM25) pe chunk-uri (`expand=False`, k=10),
> cu metricile din `retrieval_metrics.py` (copiate, pentru că importul lor încarcă modelele
> `rag_simple`) și potrivire fără diacritice. Setul implicit e `rag_simple/evaluation/tests.jsonl`
> până la setul dedicat (decizia 10), iar întrebările fără niciun cuvânt-cheie în chunk-uri se
> sar. Prima rulare arată diferențe mici între moduri; rezultatele sunt în README.

## REST (faza 5)

Rute în `app/api/routes/knowledge.py`, protejate ca rutele existente de superuser, care
cheamă aceleași funcții ca CLI-ul:

- `GET /knowledge/documents`, `GET /knowledge/documents/{id}`: registrul și avertismentele
  de hash;
- `POST /knowledge/documents` (register), `POST /knowledge/documents/{id}/prepare|review|index|validate`.
  `prepare` și `index` rulează ca `BackgroundTasks`; progresul se vede în `stage` și
  `last_error`;
- `GET /knowledge/documents/{id}/chunks` (cu întrebări), `PATCH /knowledge/chunks/{id}`;
- `POST /knowledge/chunks/{id}/questions`, `PATCH|DELETE /knowledge/questions/{id}`: fiecare
  reface doar vectorii afectați. Editarea textului unui chunk reface vectorul lui `content`.

## Faze de implementare

Fiecare fază are propriul plan și e utilizabilă singură:

1. **Registrul:** modele + migrare, `register` (inclusiv `--structured`), `status`,
   `review`, `validate`; `prepare` fără restructurare (convertoarele existente); stadiul
   scos din front-matter. ✅
2. **Restructurarea:** `restructure.py` și integrarea în `prepare`. ✅ 2026-09-28
3. **Indexarea:** chunking, întrebări, `knowledge_vectors`, `index` (cu `--no-questions`). ✅ 2026-09-28 (verificare reală în așteptare)
4. **Retrieval + evaluare:** `ChunkDocStore`, `MultiVectorRetriever`, BM25, comparația
   baseline vs. multi-vector. ✅ retrieval 2026-09-29, evaluare 2026-09-30
5. **REST.**

## Testare

- Teste pytest în `app/knowledge_prep/tests/`, în stilul celor existente: LLM-ul și
  embedding-urile se înlocuiesc cu fake-uri (ca `fake_converter` din `test_convert.py`).
- Tranzițiile de stadiu și regulile de hash: pe o bază de date de test.
- Chunking-ul: pe markdown sintetic (headere pe mai multe niveluri, secțiuni mici și mari,
  `split=false`).
- `ChunkDocStore` și ștergerea/refacerea vectorilor la reindexare: test de integrare pe
  Postgres cu pgvector.
- Verificare reală, manuală, pe `Graul_incoltit.md` (carte) și pe un regim (`split=false`).

## În afara acestui design

- Extragerea întrebărilor de test (fază separată).
- Un `diff` pe cuvinte între original și markdown-ul restructurat, fără diacritice, ca
  ajutor la review.
- Pagini de UI în frontend-ul React (Adminer și Swagger `/docs` sunt suficiente acum).
- Migrarea `rag_simple` pe noul index.
