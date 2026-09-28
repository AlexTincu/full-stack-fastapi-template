# `knowledge_prep` — pipeline-ul de cunoștințe

Transformă sursele corpusului „Calea de Autovindecare” (PDF, audio, `.txt`/`.md`) în
markdown curat și structurat, urmărit într-un registru Postgres, ca apoi să fie indexat pentru
RAG.

- **De ce și ce s-a decis:** spec-ul
  [`docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md`](../../../docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md).
  Are prioritate față de acest README la „ce trebuie să facă”.
- **Cum s-a implementat, pas cu pas:** planurile din
  [`docs/superpowers/plans/`](../../../docs/superpowers/plans/) (`2026-09-26-knowledge-registry-plan.md`
  pentru faza 1, `2026-09-28-knowledge-restructure-plan.md` pentru faza 2,
  `2026-09-28-knowledge-index-plan.md` pentru faza 3).
- **Acest README:** cum funcționează codul azi, de ce arată așa și ce trebuie știut înainte de
  a-l modifica.

## Starea fazelor

| Fază | Ce | Stare |
|---|---|---|
| 1 | Registrul: `register`, `prepare` (doar conversie), `review`, `validate`, `status` | ✅ |
| 2 | Restructurarea cu Gemini, în `prepare` | ✅ (2026-09-28) |
| 3 | Indexarea: chunking, întrebări, `knowledge_vectors`, `index` | ✅ (2026-09-28, verificare reală în așteptare) |
| 4 | Retrieval + evaluare | următoarea |
| 5 | REST | — |

## Fluxul

```
register ──> uploaded ──prepare──> structured ──review──> reviewed ──index──> indexed ──validate──> validated
                                        ▲
register --structured ──────────────────┘
```

`index` = **chunking** + **întrebări** (Gemini, opțional) + **embeddings** (OpenAI) →
`knowledge_vectors`. Detalii în secțiunea „Indexarea” de mai jos.

`prepare` = **conversie** (convertorul extensiei) + **restructurare** (Gemini) → scrie
`<folderul originalului>/converted/<slug>.md` cu front-matter. Omul revizuiește fișierul în IDE,
apoi rulează `review`.

**Stadiul stă doar în registru** (tabelul `document`, vizibil în Adminer pe `:8080`), nu în
front-matter. `review` salvează hash-ul corpului (fără front-matter). Dacă fișierul se modifică
după aceea, `status` afișează „modificat după review”.

## Comenzi

Din `full-stack-fastapi-template/backend/`:

```bash
uv run python -m app.knowledge_prep register "<fișier|folder>" [--structured] [--type regim] [--title "..."] [--author "..."] [--split/--no-split]
uv run python -m app.knowledge_prep prepare <id|slug|cale|folder> [--force] [--restructure/--no-restructure]
uv run python -m app.knowledge_prep review|validate <id|slug|cale>
uv run python -m app.knowledge_prep index <id|slug|cale> [--no-questions] [--force]
uv run python -m app.knowledge_prep status [<țintă>]

uv run pytest app/knowledge_prep                            # fără Postgres și fără Gemini
uv run ruff check app/knowledge_prep && uv run ty check app/knowledge_prep
```

Ai nevoie de `GEMINI_API_KEY` și `GEMINI_MODEL` în `full-stack-fastapi-template/.env`. `index`
are nevoie și de `OPENAI_API_KEY` (embeddings, `text-embedding-3-large`), pe lângă cheile Gemini
(întrebările, dacă nu rulezi cu `--no-questions`).

## Harta modulului

| Fișier | Rol |
|---|---|
| `__main__.py` | CLI subțire: parsează argumentele și cheamă `registry` |
| `registry.py` | **toate regulile de stadiu**: register, resolve, prepare, review, validate, avertismente de hash. Funcțiile primesc un `Session`, ca REST-ul din faza 5 să le cheme la fel |
| `models.py` | `Document`, `Chunk`, `Question` (SQLModel) + enum-urile `DocumentType`, `Stage` |
| `convert.py` | `convert(sursă) -> corp markdown`, prin registrul `converters/` |
| `converters/` | câte un convertor pe format: `pdf.py`, `audio.py`, `text.py`; plus `gemini.py` (client, thinking oprit, retry) și `diacritics.py` (`ş/ţ → ș/ț`) |
| `restructure.py` | corectura și structurarea cu Gemini (faza 2, detalii mai jos) |
| `frontmatter.py` | front-matter informativ, `slugify`, `split_frontmatter` |
| `chunking.py` | `chunk_markdown`: markdown revizuit → `ChunkDraft`-uri (faza 3, detalii mai jos) |
| `questions.py` | `generate_questions`: 1-5 întrebări Gemini per chunk, output structurat (faza 3) |
| `vectors.py` | `PgVectorIndex` / `VectorEntry`: tabelul `knowledge_vectors`, embeddings OpenAI (faza 3) |
| `index.py` | `index_document`: orchestrează chunking + întrebări + vectori într-un pas (faza 3) |
| `tests/index_fakes.py` | `FakeVectors`, `FakeQuestions` — fake-urile injectate în testele fazei 3 |

## Restructurarea (`restructure.py`)

### Ce rezolvă

Convertoarele scot text fidel, dar nu curat. `text.py` face doar curățare mecanică, iar
transcrierile audio sunt un bloc de vorbire. Fișiere ca `carti/Graul_incoltit` au:

- diacritice lipsă;
- glyph-uri de font în loc de buline;
- titluri lipite în text („Indicatii Graul incoltit este indicat in:”).

Chunking-ul din faza 3 taie după titluri `#`-`####`, deci fără structură nu are după ce tăia.

### Când rulează

| Sursă | Implicit | De ce |
|---|---|---|
| `.txt`, `.md`, audio | da | textul vine nestructurat |
| `.pdf` | nu | promptul din `pdf.py` structurează deja |

`prepare --restructure` sau `--no-restructure` schimbă comportamentul implicit. Decizia se ia
în `registry.prepare`, după extensia originalului.

### Cum funcționează

1. **Ferestre.** `split_windows` grupează paragrafe consecutive până la `WINDOW_CHARS = 24_000`
   de caractere (~8k tokeni). Taie doar între paragrafe. Un paragraf mai lung decât o fereastră
   (ex. o transcriere fără rânduri goale) devine o fereastră singură, nu se taie la mijloc.
   Fereastra e mică intenționat: la bucăți mari, modelul începe să rezume.
2. **Continuitatea ierarhiei.** Fiecare apel primește, în mesaj, a câta bucată e și titlul
   documentului. De la a doua bucată primește și calea de titluri de până atunci (`heading_path`:
   ultimul titlu de pe fiecare nivel din ce s-a produs deja, ex. `# A > ## B > ### C`). Prima
   bucată pune singurul `#`.
3. **Promptul** (`PROMPT`) e cel din spike-ul din 2026-09-25:
   - corectează diacriticele, caracterele stricate și greșelile evidente de tipar;
   - nu schimbă sensul, numerele sau numele și nu rezumă;
   - structurează în `#`-`####`, cu titluri care se înțeleg și singure;
   - face liste markdown;
   - semnalează textul suspect cu `<!-- VERIFICĂ: ... -->`.
4. **Curățare:** scoate gardurile ```` ``` ```` puse uneori în jurul răspunsului și
   normalizează `ş/ţ`.
5. **Verificare pe fiecare fereastră** (`_diff`, detalii mai jos). Dacă lipsa sau adăugarea
   depășește `MAX_DIFF = 1%`, fereastra se reîncearcă o dată. Apoi se păstrează încercarea mai
   bună, precedată de `<!-- VERIFICĂ: fereastra N, lipsă X%, adăugat Y% -->`.
6. **Răspuns gol de două ori:** fereastra intră nerestructurată, cu un marcaj `VERIFICĂ`.
   Textul nu se pierde, iar `prepare` nu cade.
7. **Erori API:** retry în `converters/gemini.py::generate` (5 încercări, doar pe
   `errors.APIError`). Dacă și ele eșuează, `prepare` scrie `last_error`, stadiul rămâne cel
   vechi și nu se scrie niciun fișier.

### Verificarea: cum se măsoară „lipsă” și „adăugat”

Folosește aceeași idee ca `pdf.py`. Textul se reduce la litere și cifre, fără diacritice și fără
spații. Apoi se compară **multiseturile de n-grame de 8 litere** (`pdf.shingles`), independent
de ordine. Corectarea diacriticelor, liniuțele de listă și mutarea titlurilor nu contează.
Contează doar literele care dispar sau apar.

- **lipsă** = n-gramele ferestrei originale care nu se regăsesc în răspuns;
- **adăugat** = n-gramele răspunsului care nu existau în fereastră.

Două detalii nu reies din spec. Au apărut la testul real pe `Graul_incoltit`. **Nu le scoate
fără să rulezi din nou testul real:**

1. **Titlurile nu se numără ca text adăugat.** Regula 6 din prompt îi cere modelului să
   adauge titluri descriptive noi. Pe o fereastră de 24k caractere, 10 titluri noi înseamnă
   ~1%, deci ar declanșa singure pragul. La „lipsă”, titlurile rămân în calcul, pentru că unele
   sunt text original mutat pe rând propriu.
2. **N-gramele se calculează pe fiecare paragraf separat** (`_paragraph_shingles`), nu pe tot
   textul lipit. Problema apare când un titlu preluat din text („Germenii de grâu…”) e scos din
   calcul (punctul 1): literele de dinainte și de după lui se lipesc și formează n-grame care nu
   există în original (`insomnie` + `germenii` → `nsomnieg`). Fără această corecție, un rezultat
   bun ieșea cu 1,6% „adăugat” fals. Costul e mic: când modelul desparte un paragraf, cele ~7
   n-grame de la tăietură apar ca „lipsă”.

### Rezultate reale (2026-09-28, `Graul_incoltit`, ~10k caractere, o fereastră)

- 13 s, lipsă 0,19%, adăugat 0%, niciun `VERIFICĂ` (spike-ul: 0,77% / 0,73%).
- Diacriticele, listele și titlurile lipite au fost recuperate corect.
- **Probleme de prompt observate** (nu de cod):
  - modelul a sărit o dată de la `#` direct la `###`, deși promptul interzice;
  - titlurile generate diferă de la o rulare la alta.

  Dacă apar des, corectează `PROMPT`, nu rezultatele de mână.

### Limitări cunoscute

- **Nu există cache pe ferestre** (spre deosebire de `pdf.py`). O carte mare întreruptă la
  mijloc se reface și se plătește de la zero. E acceptabil cât timp cărțile care trec prin
  restructurare sunt mici. Dacă devine o problemă, se poate adăuga un cache ca în `pdf.py`.
- **Promptul are două copii.** `PROMPT` din `restructure.py` a pornit de la
  `calea_de_autovindecare/.claude/skills/structurare-carte/prompt.md` (skill-ul manual cu
  `claude -p`). Diferențele: fără nota specifică unei singure cărți și cu prima frază
  generalizată pentru transcrieri. Cele două pot diverge. Pentru pipeline, sursa de adevăr e
  `restructure.py`.
- Tăietura dintre ferestre e artificială. Promptul îi cere modelului să nu completeze textul
  de la margini, dar un paragraf sau o listă pot rămâne rupte în două. Se corectează la review.

### Cum se reglează

| Vrei să… | Modifici |
|---|---|
| schimbi instrucțiunile modelului | `PROMPT` |
| ferestre mai mari/mici | `WINDOW_CHARS` (la valori mari crește riscul de rezumare) |
| prag mai strict/permisiv | `MAX_DIFF` |
| alt model | `GEMINI_MODEL` în `.env`; thinking-ul se alege în `converters/gemini.py::thinking_config` |
| refaci restructurarea unui document | `prepare <țintă> --force` (îl readuce în `structured`; reviewul se pierde) |

## Indexarea (`chunking.py`, `questions.py`, `vectors.py`, `index.py`)

### Ce rezolvă

Transformă un document `reviewed` în ce citește retrieval-ul (faza 4): `chunk`-uri (secțiuni
sau documentul întreg), 1-5 `question`-uri Gemini per chunk (pentru multi-vector retrieval —
o întrebare formulată ca a utilizatorului găsește chunk-ul chiar dacă textul lui e formulat
altfel) și vectorii lor în `knowledge_vectors`. `index <țintă>` face toți cei trei pași
dintr-o comandă; `--no-questions` sare peste întrebări (indexează doar vectorii `content`,
baseline-ul din spec).

### Ordinea pașilor, și de ce

`index_document` (`index.py`) face, în ordine:

1. **Tot ce cere un API extern, în memorie, înainte de orice scriere:** generarea
   întrebărilor (Gemini, dacă nu e `--no-questions`) și embeddings (OpenAI, pentru toate
   textele `content` + `question` deodată).
2. **`chunk` și `question` se înlocuiesc într-o singură tranzacție relațională:** se șterg
   `Question` (după `chunk_id`) și `Chunk` (după `document_id`) ale documentului, apoi se
   inserează cele noi. Commit.
3. **Vectorii se șterg după `document_id`**, apoi se adaugă cei noi (`vectors.replace_document`).
   Ștergerea după `document_id`, nu după id-urile chunk-urilor vechi, curăță și resturile
   unei rulări anterioare întrerupte la pasul 3.
4. Abia acum `doc.stage = indexed`, `doc.indexed_hash = doc.reviewed_hash`.

Motivul ordinii: API-urile (pașii 1 și, pentru vectori, embeddings din pasul 3) sunt partea
care poate eșua din motive externe (cotă, rețea), și un eșec acolo nu trebuie să lase nimic
pe jumătate scris. Ce rămâne în urmă, pe tip de eșec (toate scriu `last_error` prin
`record_step`, stadiul rămânând cel dinainte de `index`):

- **Eșec la întrebări sau la embeddings** (pasul 1): nimic nu s-a scris încă — `chunk` și
  `question` vechi rămân neatinse, vectorii vechi la fel.
- **Eșec la scrierea vectorilor** (pasul 3, după commit-ul relațional de la pasul 2): `chunk`
  și `question` sunt **deja** cele noi (commit-ul s-a făcut), dar `knowledge_vectors` a fost
  golit de `delete(filter={"document_id": ...})` înainte de eșecul la inserare — documentul
  are chunk-uri fără vectori. Dacă documentul pornea de la `reviewed` (primul `index`),
  `doc.stage` rămâne `reviewed`, deci pare „neindexat” deși are chunk-uri noi în DB. **Dacă
  documentul era deja `indexed` sau `validated` (reindexare)**, `doc.stage` nu se schimbă în
  acest pas eșuat — deci rămâne `indexed`/`validated` cu vectori inexistenți; singurul semnal
  e `last_error`, nu stadiul. În ambele cazuri, o nouă rulare `index` (fără `--force` dacă
  stadiul era `reviewed`, cu `--force` dacă era `validated`) reface totul corect: chunk-urile
  se recreează identic, iar vectorii se scriu din nou.

`__main__.py::_index` cheamă `check_indexable(doc, force=...)` **înainte** de a construi
`PgVectorIndex.from_settings()` (fix din 48ed5f3): o regulă încălcată (stadiu, hash modificat,
`validated` fără `--force`) se raportează fără nicio conexiune reală la Postgres sau OpenAI.
`index_document` verifică din nou regulile la început (`check_indexable`), ca apelul direct
din Python/teste să fie la fel de sigur.

### Chunking-ul (`chunking.py`)

- **`split=false`** (regimuri, note, insight-uri): un singur chunk, tot corpul,
  `heading_path = [titlul documentului]`. Avertizează dacă depășește `UNSPLIT_WARN_TOKENS`
  (8000; limita lui `text-embedding-3-large` e 8191) — vectorul `content` devine o medie pe
  bucăți, imprecisă, dar întrebările nu sunt afectate (tot corpul intră în promptul Gemini).
- **`split=true`:** o secțiune la fiecare titlu `#`-`####` din afara blocurilor de cod
  (headerele rămân în text), apoi:
  - secțiunile sub `MIN_CHUNK_TOKENS` (80) se lipesc de **următoarea** secțiune, dar doar
    dacă aceea e soră sau descendentă a aceluiași părinte (`heading_path[:-1]`); dacă
    următoarea secțiune sare într-o altă ramură, secțiunea mică rămâne mică, de sine
    stătătoare — nu sare granița de părinte. Calea rezultată e prefixul comun al celor două
    căi, **cu o excepție**: dacă secțiunea mică nu are decât linii de titlu (un `#` urmat
    direct de un subtitlu, fără text propriu între ele), chunk-ul unit ia calea **întreagă**
    a secțiunii următoare — titlul „gol” era doar structură, nu conținut care să justifice
    un prefix mai scurt;
  - secțiunile peste `MAX_CHUNK_TOKENS` (1000) se taie cu `RecursiveCharacterTextSplitter`
    (overlap `CHUNK_OVERLAP_TOKENS`, 150), păstrând `heading_path` neschimbat pe fiecare bucată;
  - marcajele `<!-- VERIFICĂ: ... -->` lăsate de restructurare se scot înainte de orice.
- **Parserul de titluri e propriu, nu `MarkdownHeaderTextSplitter`.** Acela scoate rândurile
  goale dintre linii și le lipește cu `"  \n"` (line break markdown), deci schimbă textul
  exact: `"# A\n\nText."` devine `"# A  \nText."` — rândul gol dintre titlu și text dispare.
  Pentru `chunk.text`, care trebuie să rămână identic cu markdown-ul revizuit (viitorul REST
  din faza 5 îl arată și îl editează așa cum e), asta nu e acceptabil. Mai grav, fără
  rândurile goale `RecursiveCharacterTextSplitter` nu mai găsește granițele de paragraf ca
  să taie secțiunile mari acolo. Parserul propriu (`_split_sections`) doar urmărește nivelul
  titlurilor, fără să atingă vreun caracter din text.

### `knowledge_vectors`

Un rând pe vector-copil: `content` (un rând per chunk, `id = chunk.id`) sau `question` (un
rând per întrebare, `id = question.id`). Coloane de metadate filtrabile: `doc_id` (chunk-ul
părinte — cheia lui `MultiVectorRetriever` din faza 4), `document_id`, `kind`. Textul embedded:
pentru `content`, `heading_path` unit cu ` > ` + rând gol + `chunk.text` (`embedding_text`,
în `chunking.py`); pentru `question`, doar textul întrebării.

Tabelul îl creează `PGEngine.init_vectorstore_table` la prima folosire (`PgVectorIndex.__init__`
verifică întâi dacă există, ca în `rag_simple`), **nu Alembic**: schema e a lui
`langchain_postgres`, iar `alembic/env.py` (`_ALEMBIC_EXCLUDE_PREFIXES`) o exclude explicit din
autogenerate. **Fără index ANN:** pgvector indexează coloana `vector` doar până la 2000 de
dimensiuni, iar `text-embedding-3-large` are 3072 — la câteva mii de rânduri, scanarea exactă
e suficient de rapidă.

### `validated` → `--force`

Reindexarea unui document `validated` cere `--force`: altfel, `check_indexable` refuză cu un
mesaj explicit că întrebările existente (eventual editate manual în faza 5 REST) s-ar pierde.
Independent de `--force`, dacă documentul are deja întrebări (orice reindexare, nu doar peste
`validated`), `index_document` avertizează `se înlocuiesc N întrebări existente (editările
manuale se pierd)` — e un avertisment, comanda continuă.

### Limitări cunoscute

- **Fără cache pe întrebări** (la fel ca `restructure.py`, care nu are nici el cache pe ferestre,
  dar e mai puțin costisitor). O carte cu multe secțiuni întreruptă la mijloc (eroare
  de rețea, cotă Gemini) se reindexează de la zero la următoarea rulare — niciun chunk nu-și
  ține întrebările generate deja.
- **Un apel Gemini per chunk, secvențial.** Nicio paralelizare: o carte cu 30 de secțiuni face
  30 de apeluri unul după altul. Acceptabil la dimensiunile actuale ale corpusului.
- **`tiktoken` descarcă encoderul `cl100k_base` la prima rulare** (cache local după aceea,
  vezi `tiktoken`), deci primul `index` (sau primul test care numără tokeni) are nevoie de
  acces la internet.

### Rezultate reale

_De completat după verificarea manuală (planul, taskul 7) — nu s-a rulat încă la data acestui
README._

### Cum se reglează

| Vrei să… | Modifici |
|---|---|
| chunk-uri mai mari/mici | `MIN_CHUNK_TOKENS`, `MAX_CHUNK_TOKENS` (`chunking.py`) |
| overlap la tăierea secțiunilor mari | `CHUNK_OVERLAP_TOKENS` (`chunking.py`) |
| pragul de avertisment pentru `split=false` | `UNSPLIT_WARN_TOKENS` (`chunking.py`) |
| instrucțiunile date modelului pentru întrebări | `PROMPT` (`questions.py`) |
| câte întrebări maxim per chunk | `MAX_QUESTIONS` (`questions.py`) |

## Testare

- Testele nu cheamă Gemini și nu au nevoie de Postgres:
  - registrul rulează pe SQLite in-memory (`tests/conftest.py`);
  - clientul Gemini e un `FakeClient` care întoarce răspunsuri prestabilite (`test_pdf.py`,
    `test_restructure.py`).
- **Convertoarele și restructuratorul se injectează.** `registry.prepare(...)` și
  `__main__.main(...)` primesc `converters=` și `restructurer=`. În teste, dă-le **pe amândouă**
  fake. Altfel, `prepare` pe un `.txt` cheamă Gemini pe bune. Helper-ele `prep()` din
  `test_registry.py` și `run()` din `test_cli.py` fac deja asta.
- `Restructurer = Callable[[str, str], str]`: primește corpul și titlul, întoarce corpul
  restructurat. Orice funcție cu forma asta poate înlocui Gemini.
- Verificarea reală se face manual, pe un document mic, fără să atingi corpusul: cheamă
  `restructure(corp, titlu)` din Python și scrie rezultatul într-un folder temporar.
- **Indexarea se testează la fel, cu fake-uri proprii** (`tests/index_fakes.py`):
  `FakeVectors` (embeddings deterministe după lungimea textului, `replace_document` în
  memorie, cu `fail="embed"` / `fail="write"` ca să simuleze un eșec la fiecare pas) și
  `FakeQuestions` (întrebări deterministe, înregistrează apelurile primite). `index_document(...)`
  primește `vectors=` și `questioner=` direct; `__main__.main(...)` primește și el `vectors=`
  și `questioner=` (implicit `PgVectorIndex.from_settings()`, respectiv `generate_questions`
  din `questions.py`) — dă-le fake în teste, la fel cum `converters=`/`restructurer=` se dau
  fake pentru `prepare`. Helper-ul `run()` din `test_cli.py` face deja asta pentru `index`.
- `test_vectors_pg.py` e integrare reală pe Postgres cu pgvector (ștergerea și refacerea
  vectorilor unui document, filtrarea după `kind`): se sare automat dacă `.env` nu are o
  bază care răspunde (`docker compose up -d db`), altfel creează un tabel propriu
  `knowledge_vectors_test_<hex>` per test și îl șterge la final — nu atinge `knowledge_vectors`.
