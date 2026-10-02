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
  `2026-09-28-knowledge-index-plan.md` pentru faza 3,
  `2026-09-29-knowledge-retrieval-plan.md` pentru faza 4 (doar retrieval, fără evaluare)).
- **Acest README:** cum funcționează codul azi, de ce arată așa și ce trebuie știut înainte de
  a-l modifica.

## Starea fazelor

| Fază | Ce | Stare |
|---|---|---|
| 1 | Registrul: `register`, `prepare` (doar conversie), `review`, `validate`, `status` | ✅ |
| 2 | Restructurarea cu Gemini, în `prepare` | ✅ (2026-09-28) |
| 3 | Indexarea: chunking, întrebări, `knowledge_vectors`, `index` | ✅ (2026-09-28, verificare reală în așteptare) |
| 4 | Retrieval + evaluare | ✅ retrieval (2026-09-29), evaluare (2026-09-30) |
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

**Reînregistrarea** aceluiași fișier (ca original sau ca markdown structurat, oricum ar fi fost
înregistrat prima dată) **actualizează rândul existent** (același id), ca să arate exact ca o
înregistrare nouă cu opțiunile date: tip, `split`, titlu/slug, autor, căi. Cu aceleași opțiuni se
sare (`AlreadyRegistered`). Dacă se schimbă markdown-ul (ex. original → `--structured`), stadiul și
hash-urile o iau de la capăt; altfel un document `indexed`/`validated` coboară la `reviewed`, ca să
fie reindexat cu noile setări (chunk-urile vechi rămân în DB până la `index`).

## Comenzi

Din `full-stack-fastapi-template/backend/`:

```bash
uv run python -m app.knowledge_prep register "<fișier|folder>" [--structured] [--type regim] [--title "..."] [--author "..."] [--split/--no-split]
uv run python -m app.knowledge_prep prepare <id|slug|cale|folder> [--force] [--restructure/--no-restructure]
uv run python -m app.knowledge_prep review|validate <id|slug|cale>
uv run python -m app.knowledge_prep index <id|slug|cale> [--no-questions] [--force]
uv run python -m app.knowledge_prep status [<țintă>]
uv run python -m app.knowledge_prep search "<întrebare>" [-k N] [--no-questions] [--no-hybrid] [--no-expand]
uv run python -m app.knowledge_prep evaluate [--tests <fișier.jsonl>] [--mode baseline ...] [--all] [--details] [--save "<etichetă>"]
uv run python -m app.knowledge_prep.explorer                # Gradio: comparația a două metode de retrieval

uv run pytest app/knowledge_prep                            # fără Postgres și fără Gemini
uv run ruff check app/knowledge_prep && uv run ty check app/knowledge_prep
```

Ai nevoie de `GEMINI_API_KEY` și `GEMINI_MODEL` în `full-stack-fastapi-template/.env`. `index`
are nevoie și de `OPENAI_API_KEY` (embeddings, `text-embedding-3-large`), pe lângă cheile Gemini
(întrebările, dacă nu rulezi cu `--no-questions`). `search` are nevoie doar de `OPENAI_API_KEY`
(embedding-ul întrebării), la fel și `evaluate`.

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
| `index.py` | `index_document`: chunking + întrebări + vectori, incremental și pe loturi (faza 3; reluabil din 2026-09-30) |
| `tests/index_fakes.py` | `FakeVectors` (cu `fail`/`writes_ok` pentru căderi la un anumit lot), `FakeQuestions` — fake-urile injectate în testele fazei 3 |
| `docstore.py` | `ChunkDocStore`: părintele `MultiVectorRetriever` peste tabelul `chunk`, doar-citire (faza 4) |
| `lexical.py` | `tokenize`, `ChunkBM25`: index BM25 în memorie peste titluri + `chunk.text`, cu stopwords și stemming (faza 4) |
| `parents.py` | `expand_to_parents`, `parent_of`: chunk-urile găsite → părinți mai mari (document / secțiune / chunk cu vecini) pentru LLM (faza 4) |
| `retrieval.py` | `build_retriever`, `KnowledgeRetriever`, `reciprocal_rank_fusion`: multi-vector, baseline, hibrid (faza 4) |
| `evaluation.py` | `evaluate`, `question_metrics`, `load_tests`, `split_answerable`: metricile retrieval-ului pe întrebările de test, pe fiecare mod (faza 4) |
| `explorer.py` | Gradio pentru comparația vizuală a retrieval-ului: două panouri, `rag_simple` sau oricare mod (faza 4, detalii mai jos) |
| `eval_results/` | instantanee JSON ale evaluărilor salvate cu `evaluate --save` (formatul `EvalSnapshot` din `rag_simple`) |
| `notebooks/` | câte un Jupyter notebook pe fază (`01_register.ipynb`, ...): aceleași funcții ca CLI-ul, cu explicații și opțiuni în variabile, pentru test și debug pe DB-ul real |
| `tests/retrieval_fakes.py` | `add_document`, `child`, `FakeChildStore` — fake-urile injectate în testele de retrieval |
| `tests/pg.py` | `PG_URL`, `requires_pg`, `has_table`, `drop_table` — conexiunea Postgres pentru testele de integrare (`test_vectors_pg.py`, `test_retrieval_pg.py`) |

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

### Ordinea pașilor, și de ce (indexare incrementală, 2026-09-30)

`index_document` (`index.py`) face, în ordine:

1. **Chunking** și comparația cu chunk-urile din DB după `content_hash` (`chunking.chunk_hash`,
   sha256 pe `embedding_text`). Un chunk vechi se **păstrează** dacă are același hash, are
   `indexed_at` pus și a fost făcut în același mod: are întrebări exact când rularea curentă
   le cere. Un chunk păstrat primește doar `position` nouă; întrebările lui, inclusiv cele
   editate manual, rămân. `--force` nu păstrează nimic.
2. **Chunk-urile vechi nepăstrate** (dispărute, schimbate, neterminate sau făcute în alt mod)
   se șterg: întâi vectorii lor (`vectors.delete_chunks`, după `doc_id`, deci și resturile unei
   rulări întrerupte), apoi `question` și `chunk`. Commit.
3. **Chunk-urile de refăcut, în loturi de `BATCH_SIZE` (10)**: întrebări Gemini, embeddings
   OpenAI, apoi `chunk` + `question` (commit, încă fără `indexed_at`), vectorii
   (`vectors.add`) și `indexed_at` (commit).
4. Abia după ultimul lot: `doc.stage = indexed`, `doc.indexed_hash = doc.reviewed_hash`.

Ce rămâne după un eșec sau o întrerupere (`record_step` scrie `last_error` și pentru
`KeyboardInterrupt`, ca „întrerupt”; stadiul rămâne cel dinainte):

- loturile terminate rămân în DB, cu `indexed_at`;
- lotul în curs lasă cel mult chunk-uri fără `indexed_at`. Rularea următoare le șterge (cu
  vectorii scriși parțial) și le reface;
- o nouă rulare `index` cere întrebări doar pentru chunk-urile rămase. Dacă totul fusese scris
  (întrerupere după ultimul lot), doar marchează documentul `indexed`, fără apeluri API.

**Atenție la întreruperea din notebook:** `langchain_postgres` scrie vectorii pe un thread
propriu, pe care Ctrl+C nu îl oprește, așa că un lot întrerupt la scriere se poate termina
totuși în fundal. Nu e o problemă de corectitudine: chunk-urile lui n-au `indexed_at`, deci
rularea următoare le reface.

Log-ul arată câte chunk-uri se păstrează și progresul pe loturi („lot 3/8: embeddings și
scriere ...”), ca o scriere lungă să nu pară blocată.

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
mesaj explicit. Cu `--force` se refac **toate** chunk-urile, deci se pierd și întrebările
editate manual. Fără `--force`, pe un document `reviewed`/`indexed` se pierd doar întrebările
chunk-urilor schimbate, iar `index_document` avertizează `se înlocuiesc N întrebări
existente (editările manuale se pierd)`. E doar un avertisment; comanda continuă.

### Limitări cunoscute

- **Un chunk cu 0 întrebări generate** (avertisment `nicio întrebare generată`) arată ca unul
  făcut cu `--no-questions`, deci se reface la următorul `index` cu întrebări.
- **Schimbarea parametrilor de chunking** schimbă textul chunk-urilor, deci hash-urile: se
  refac automat doar chunk-urile afectate.
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

## Retrieval (`docstore.py`, `lexical.py`, `parents.py`, `retrieval.py`)

### Ce rezolvă

Transformă indexul fazei 3 (`chunk`, `knowledge_vectors`) într-un retriever gata de folosit:
top `k` **părinți** pentru o întrebare (documentul întreg, o secțiune sau chunk-ul cu vecinii
lui — vezi „Extinderea la părinți”), ca `Document`-uri LangChain. Cu `expand=False` întoarce
chunk-urile găsite, cu metadatele din `ChunkDocStore` (pentru evaluarea căutării).
Patru combinații de căutare, prin cele două opțiuni ortogonale ale lui
`build_retriever(vectorstore, bind, *, k=DEFAULT_K, questions=True, hybrid=True, expand=True)`:

| `questions` | `hybrid` | Ce caută |
|---|---|---|
| `True` | `True` | multi-vector (`content` + întrebări) + BM25, combinate prin RRF |
| `True` | `False` | doar multi-vector |
| `False` | `True` | baseline (doar `content`) + BM25, combinate prin RRF |
| `False` | `False` | doar baseline |

`build_retriever` e singurul punct de intrare pentru cod (evaluarea din faza 4, REST-ul din
faza 5, un viitor agent) — nu construi `KnowledgeRetriever` direct.

### Cum funcționează

1. **Semantic:** `MultiVectorRetriever(vectorstore=..., docstore=ChunkDocStore(bind), id_key="doc_id")`
   caută în `knowledge_vectors` (copiii `content`/`question`) și reduce rezultatele la
   părinți (chunk-uri) după `doc_id`. `search_kwargs["k"] = k * CHILDREN_PER_CHUNK`, cu
   `CHILDREN_PER_CHUNK = MAX_QUESTIONS + 1 = 6`: un chunk poate avea un vector `content` și
   până la 5 întrebări, deci cerând de 6 ori mai mulți copii decât `k`, rămân cel puțin `k`
   părinți distincți chiar dacă toți copiii găsiți s-ar grupa în chunk-uri complete. Baseline =
   același retriever, cu `search_kwargs["filter"] = {"kind": "content"}` (sar peste întrebări).
2. **`ChunkDocStore(bind)`** e părintele `MultiVectorRetriever`: doar-citire peste tabelul
   `chunk` (`mget` face un join cu `document`; `mset`/`mdelete` ridică `NotImplementedError`,
   scrierea se face doar prin `index`). Acceptă chei `str` **și** `uuid.UUID`, pentru că
   `PGVectorStore` pune `doc_id` în metadatele copiilor ca `uuid.UUID` (coloana e `UUID`), iar
   `MultiVectorRetriever` i-l dă neschimbat lui `mget`; testele și CLI-ul folosesc `str`.
3. **Lexical (`ChunkBM25.from_db`):** un index `rank_bm25.BM25Okapi` construit o dată, în
   memorie, din **toate** chunk-urile din DB, pe același text ca vectorul `content`
   (`embedding_text`: calea de titluri + `chunk.text`). Titlurile contează: „Germenii de grâu
   combat constipația” e doar în `heading_path`, nu și în textul chunk-ului.
4. **Hibrid:** listele semantică și lexicală (id-uri de `chunk`) se combină prin
   `reciprocal_rank_fusion` pe `chunk.id`, cu `RRF_K = 60` (constanta din articolul RRF, ca în
   `rag_simple`); la egalitate câștigă ordinea din lista semantică.
5. **Extinderea la părinți** (implicit, `parents.py`): lista ordonată de chunk-uri devine o
   listă de părinți distincți, trunchiată la `k`. Cu `expand=False`, rezultatul se citește prin
   `ChunkDocStore.mget` și se trunchiază la `k` chunk-uri.

### Extinderea la părinți (decizia 2026-09-29)

Chunk-urile mici sunt bune pentru **găsire** (un vector pe o temă îngustă se potrivește precis),
dar prea mici pentru **răspuns**: la „am colesterolul mare”, LLM-ul trebuie să vadă și dozele, și
contraindicațiile curei, nu doar secțiunea „scade colesterolul”. De aceea căutarea rămâne pe
chunk-uri, iar fiecare chunk găsit se extinde, în ordinea rangului, la cea mai largă unitate care
încape în `PARENT_MAX_TOKENS = 4000`:

1. **documentul întreg** (`scope="document"`, `heading_path = [titlu]`);
2. altfel, **secțiunea strămoșă cea mai largă**: prefixele lui `heading_path`, de la rădăcină în
   jos; primul care încape (`scope="section"`). Un document cu un singur `#` (titlul) trece
   astfel la capitolele `##`;
3. altfel, **chunk-ul cu vecinii lui** (± `NEIGHBORS = 1`), doar cei cu același `heading_path`
   (`scope="chunk"`) — bucățile unei secțiuni mari tăiate de `_split_large`.

Părinții se deduplică după conținut: al doilea chunk găsit în același părinte nu ocupă un loc din
`k`, doar se adaugă la `metadata["matched_chunk_ids"]`. Metadatele unui părinte: `chunk_id` (cel
mai bun chunk găsit în el, ca la `expand=False`), `matched_chunk_ids`, `chunk_ids` (toți membrii,
în ordine), `document_id`, `slug`, `title`, `heading_path`, `scope`. Textul e `chunk.text`-urile
membrilor lipite cu un rând gol (headerele sunt deja în text). Cu `k=5`, contextul e de cel mult
~20k tokeni. Documentele se încarcă leneș, câte unul, și nu se mai încarcă altele după `k`
părinți.

### De ce BM25 cu `rank_bm25`, nu `hybrid_search_config`

`PGVectorStore` are propria căutare hibridă (`hybrid_search_config`), dar lucrează pe **copii**,
nu pe chunk-uri: full-text Postgres (`ts_rank`) pe coloana `content` a lui `knowledge_vectors`,
cu fuziune printr-o sumă ponderată de scoruri normalizate, **înainte** de reducerea la părinți —
un chunk cu 5 întrebări ar fi favorizat de 6 ori față de unul fără, iar spec-ul cere RRF pe
`chunk.id`. Ar cere și o coloană `tsvector` + index GIN pe un tabel creat de `PGEngine` (nu de
Alembic), plus o configurație românească fără diacritice (`unaccent`) — o migrare nouă pentru un
corpus care azi are câteva zeci de chunk-uri. `rank_bm25` în memorie e suficient de rapid la
această dimensiune și e deja dependență directă (folosit și de `rag_simple`).

**Tokenizarea** (`tokenize`, `lexical.py`): literele mici, NFKD, fără semnele combinante
(`ă→a`, `â`/`î→a`/`i`, `ș`/`ş→s`, `ț`/`ţ→t`), apoi `\w+`, fără `STOPWORDS` (cuvinte de legătură,
scrise fără diacritice), apoi stemmer-ul Snowball pentru română (`snowballstemmer`). Aceeași
funcție pe corpus și pe query. Diacriticele se scot **înainte** de stemming: utilizatorii scriu
des fără ele, unele texte vechi au sedilă (`ş`/`ţ`), iar așa „constipație”, „constipația”,
„constipatie” și „constipatia” dau toate `constipat` (cu diacritice, stemmer-ul dă `constip`
pentru unele și `constipat` pentru altele).

**De ce stopwords + stemming (2026-09-29):** fără ele, hibridul strica sistematic rezultatele
multi-vector. „ce”, „cine”, „pentru” primeau IDF mare pe corpusul mic (ex. `cine` = 2.44) și
aduceau chunk-uri care doar le conțineau, iar „constipație” din întrebare nu se potrivea cu
„constipația” din text. Măsurat pe 7 întrebări (vezi „Rezultate reale”): chunk-ul corect a
revenit pe #1 în hibrid la „constipație”, „contraindicații”, „obosit”, „părul”.

### Limitări cunoscute

- **Stemming parțial:** articolul hotărât după vocală rămâne (`grâul` → `graul`, `grâu` → `grau`);
  la fel, stemmer-ul nu unifică forme cu alternanțe (`nervos`/`nervoasă`). Partea semantică
  acoperă aceste cazuri.
- **Pe un corpus foarte mic, BM25Okapi dă IDF 0** unui cuvânt prezent în exact jumătate din
  chunk-uri (ex. 1 din 2), deci și scor 0 — ține cont de asta în teste.
- **Indexul BM25 nu vede chunk-uri noi până la un nou `build_retriever`:** se construiește o
  singură dată, din tot ce era în DB la acel moment. CLI-ul `search` îl reface la fiecare
  rulare, deci pentru el nu e o problemă; un proces de durată (server, agent) ar avea nevoie
  să-l reconstruiască periodic — decizie lăsată pentru faza 5 (întrebarea deschisă 2 din plan).
- **Cuvintele comune care nu sunt în `STOPWORDS` primesc scor BM25 mic, dar pozitiv.**
  `BM25Okapi` limitează IDF-ul negativ la `epsilon * average_idf` (epsilon implicit 0.25), iar
  media IDF e pozitivă pe un corpus real; un cuvânt prezent în peste jumătate din chunk-uri ia
  acest prag în loc de un IDF negativ. Lista de stopwords reduce efectul, nu-l elimină.
- **Părinții se repetă parțial la secțiunile foarte mari:** bucățile tăiate cu
  `CHUNK_OVERLAP_TOKENS` au text comun, iar textul lipit al părintelui îl repetă. Ferestrele
  `scope="chunk"` a două chunk-uri găsite apropiate se pot suprapune (sunt părinți diferiți).
- **Documentele readuse în `structured`** (`prepare --force`) rămân căutabile până la
  următorul `index`: retrieval-ul nu filtrează după stadiu, doar citește `chunk` și
  `knowledge_vectors` cum sunt (spec, „Stadii” — chunk-urile și vectorii unui document rămân
  neatinși până la o nouă indexare).
- **Fără filtrare după stadiul documentului**, în general: un document `reviewed` neindexat nu
  are chunk-uri, deci nu apare; unul `indexed`/`validated` apare mereu, indiferent de stadiu.

### Rezultate reale (2026-09-29)

Verificare pe corpusul real (Postgres + OpenAI), un singur document indexat la acel moment
(`orele-de-refacere-ale-organelor`, 36 chunk-uri, 178 vectori; `graul-incoltit-valeriu-popa` era
`reviewed`, neindexat). Toate cele 4 combinații rulează cap-coadă fără erori; căutarea fără
diacritice merge.

- **Q1** „La ce ora lucreaza cel mai intens ficatul?” (fără diacritice), `-k 3`: toate cele 4
  moduri pun pe #1 secțiunea corectă („Vezica biliară și ficatul”); modurile diferă doar la
  #2/#3. Hibridul (baseline+BM25, multi+BM25) aduce și chunk-uri cu `heading_path` egal cu
  titlul documentului (text dinaintea primului titlu / bucăți dintr-o secțiune mare) — un semn
  că BM25 pe `chunk.text` favorizează uneori chunk-uri mari, generice.
- **Q2** „Când e bine să mergi la toaletă dimineața?”, `-k 3`: baseline și baseline+BM25 pun pe
  #1 secțiunea cea mai relevantă (intestinul gros, ORELE 05-07); la multi-vector, aceeași
  secțiune iese din top 3; la multi+BM25 apare pe #2. Pe acest document diferențele dintre
  moduri sunt mici și nu arată încă un câștig clar al multi-vector.
- Fără document care să conțină „grâul încolțit” în corpusul indexat la acel moment: întrebarea
  respectivă din plan a întors doar chunk-uri irelevante din singurul document indexat —
  comportament corect (nu există prag de relevanță, retriever-ul întoarce mereu top `k`), dar
  nu verifică nimic.
- Concluziile despre baseline vs. multi-vector vin doar din evaluare, cu un set de întrebări de
  test și un corpus mai mare (inclusiv cărți indexate). Zgomot: fiecare rulare CLI afișează pe
  stderr avertismentul `from pydantic.v1 import BaseModel` (dintr-o bibliotecă, preexistent).

### Rezultate reale, după schimbarea întrebărilor, BM25 și extindere (2026-09-29, a doua rundă)

Două documente indexate: `graul-incoltit-valeriu-popa` (19 chunk-uri, 3.409 tokeni) și
`orele-de-refacere-ale-organelor` (7.836 tokeni). Întrebări „cum întreabă omul”, `--no-expand`,
locul chunk-ului corect:

| Întrebare | multi-vector, promptul vechi | + BM25 vechi | multi-vector, promptul nou | + BM25 nou |
|---|---|---|---|---|
| ce recomanzi pentru sistemul nervos? | #2 | lipsă din top 5 | **#1** | #2 |
| am colesterolul mare, ce pot face? | #1 | #1 | #1 | #1 |
| ce să mănânc pentru constipație? | #1 | lipsă din top 5 | #1 | **#1** |
| cine nu are voie să mănânce grâu încolțit? | #1 | #5 | #1 | **#1** |
| sunt mereu obosit, ce pot face? | — | — | #1 | #1 |
| îmi cade părul | — | — | #1 | #1 |

- **Promptul întrebărilor** (`questions.py`) cere acum, în ordine: 1-2 întrebări pornind de la
  problemă, fără să numească remediul („Ce pot face pentru colesterolul mare?”), una de
  ansamblu cu subiectul numit și 0-2 de detaliu, fără variante ale aceleiași idei (vârste, doze).
  Motivul: omul nu știe ce e în corpus și întreabă de la problema lui. Promptul vechi cerea
  întrebări la care fragmentul „răspunde complet”, deci doar întrebări înguste, pornind de la remediu.
- **Cu extindere** (implicit), „am colesterolul mare” întoarce pe #1 tot documentul cu grâul
  (`document`, 10/19 chunk-uri găsite), apoi secțiuni din „Orele de refacere” (peste prag).
- Corpusul are doar 2 documente: măsurătorile sunt orientative până la un set de evaluare pe un
  corpus mai mare.

### Cum se reglează

| Vrei să… | Modifici |
|---|---|
| câte rezultate implicit | `DEFAULT_K` (`retrieval.py`) |
| cât de mare poate fi un părinte | `PARENT_MAX_TOKENS` (`parents.py`) |
| câți vecini are un chunk dintr-o secțiune prea mare | `NEIGHBORS` (`parents.py`) |
| câți copii candidați per chunk cerut | `CHILDREN_PER_CHUNK` (`retrieval.py`; legat de `MAX_QUESTIONS` din `questions.py`) |
| cât de repede scade greutatea unui rang mai slab la RRF | `RRF_K` (`retrieval.py`) |
| tokenizarea BM25 (stemming, stopwords) | `tokenize` (`lexical.py`) |

## Evaluarea (`evaluation.py`, comanda `evaluate`)

### Ce face

Trece fiecare întrebare de test prin cele patru moduri ale lui `build_retriever`
(`baseline`, `baseline+bm25`, `multi-vector`, `multi-vector+bm25`), cu `k=10` și **fără
extindere la părinți**. Părinții (document/secțiune) ar conține aproape orice cuvânt-cheie și ar
umfla metricile. Afișează media pe toate întrebările, apoi pe fiecare categorie (`direct_fact`,
`reasoning` etc.). Cu `--details` afișează și MRR-ul fiecărei întrebări, iar cu `--save` scrie
mediile în `eval_results/`.

Pentru analiză, `notebooks/05_evaluation.ipynb` face aceeași evaluare, dar păstrează rezultatele în
memorie. Arată întrebările unde două moduri diferă, pe cele ratate în toate modurile și, pentru o
întrebare, top 10 din fiecare mod plus chunk-urile din DB care conțin cuvintele-cheie. Așa se vede
dacă a greșit retrieval-ul sau cuvântul-cheie. Secțiunea 7 compară ultimele două instantanee salvate.

- **Setul de test:** implicit `rag_simple/evaluation/tests.jsonl`, în formatul `TestQuestion`
  (`question`, `keywords`, `reference_answer`, `category`); alt fișier se dă cu `--tests`. Setul
  dedicat noului corpus vine din faza separată de extragere a întrebărilor (decizia 10 din spec)
  și va folosi același format.
- **Relevanța** e aproximată ca în `rag_simple`: un chunk e relevant pentru un cuvânt-cheie dacă
  îl conține (subșir). Metricile (MRR, nDCG@10, recall/precision@5/@10, hit rate@10) au aceleași
  definiții ca în `rag_simple/evaluation/retrieval_metrics.py`.

### De ce arată așa

- **Metricile sunt copiate, nu importate.** Importul lui `retrieval_metrics.py` trage după el
  `rag_simple.retrieval.pipeline`, care la import creează modelele de chat și citește
  `chunks.json`. Asta durează ~10 s și cere cheile API chiar și în teste. `rag_simple` rămâne
  neatins (decizia 11), deci definițiile sunt copiate în `question_metrics`.
- **Potrivirea ignoră diacriticele și spațiile multiple** (`fold`), spre deosebire de
  `rag_simple`. Cuvintele-cheie au fost scrise pe textele vechi, adesea fără diacritice („2-5
  saptamani”, „grãsimi”), iar markdown-ul restructurat le are.
- **Întrebările fără răspuns în corpus se sar** (`split_answerable`): o întrebare e sărită dacă
  niciun cuvânt-cheie de-al ei nu apare în niciun chunk din DB, adică documentul ei nu e indexat.
  Altfel ar da 0 în toate modurile și ar tasa diferențele. Opțiunea `--all` le include.

### Rezultate reale (2026-09-30)

Rulat pe `tests.jsonl` (43 de întrebări, dintre care 36 evaluate și 7 sărite), cu chunk-urile din
DB de la acea dată: `graul-incoltit` (19 chunk-uri, 62 de întrebări), `orele-de-refacere` (36,
142) și `seminar-macrobiotica` (74, 251; e `reviewed`, dar are chunk-urile de la o indexare
anterioară).

| mod | MRR | nDCG | recall@5 | recall@10 | hit@10 |
|---|---|---|---|---|---|
| baseline | 0.713 | 0.719 | 0.756 | 0.804 | 0.861 |
| baseline+bm25 | 0.744 | 0.756 | 0.831 | 0.852 | 0.917 |
| multi-vector | 0.734 | 0.736 | 0.784 | 0.798 | 0.861 |
| multi-vector+bm25 | 0.726 | 0.735 | 0.800 | 0.824 | 0.889 |

Ce se vede:

- **BM25 ajută peste baseline** (+0.03 MRR, +0.075 recall@5), mai ales la `reasoning` (MRR
  0.71 → 0.91). **Multi-vector ajută puțin singur** (+0.02 MRR), dar **combinat cu BM25 nu
  depășește `baseline+bm25`**.
- Multi-vector câștigă clar la întrebările despre ore („21:00-23:00”, „trezirea între 23 și 1”:
  MRR 0.17-0.62 → 1.00). Pierde la „Spune-mi despre grâul încolțit” și la „carențe minerale”
  (1.00 → 0.33): întrebările generate trag chunk-uri de detaliu înaintea celui de ansamblu.
- BM25 strică la enumerări lungi din seminar („organele cerealelor”, „5 dificultăți”: 1.00 →
  0.50-0.69), unde cuvintele comune ridică alte chunk-uri.
- Ratate în toate modurile: întrebările „după Ohsawa” (cartea lui Ohsawa nu e indexată, dar nu
  sunt sărite, pentru că termeni ca „diabetul” apar în seminar) și cele scrise ca un utilizator
  („Ce recomanzi pentru sistemul nervos”, „Ce să mănânc pentru constipație?”). Pe acestea
  multi-vector ar trebui să le ajute, dar cuvântul-cheie „grâu încolțit” nu apare în chunk-urile
  despre efecte, care spun doar „grâul” sau „germenii”. E o limită a aproximării prin
  cuvinte-cheie.
- **Concluzie provizorie:** diferențele sunt de ordinul 1-2 întrebări din 36. Nu se poate alege
  între moduri până la setul de test dedicat, pe mai multe documente.

### Limitări

- Relevanța prin subșir favorizează chunk-urile lungi și penalizează parafrazele. Un chunk
  corect care spune „germenii” în loc de „grâu încolțit” contează ca irelevant.
- `split_answerable` e grosier: o întrebare se evaluează dacă *oricare* cuvânt-cheie apare
  oriunde, chiar dacă documentul ei țintă lipsește.
- Fiecare mod face propriul embedding al întrebării, deci 4 × N apeluri OpenAI (ieftine, câteva
  zeci de secunde).

## Comparația vizuală (`explorer.py`)

`evaluate` dă cifrele; `explorer` arată *de ce* ies așa. Aceeași întrebare merge la două
panouri (A | B), fiecare cu metoda lui: `rag_simple` (rescriere + semantic + BM25 + rerank, peste
`rag_simple_chunks`) sau unul din cele 4 moduri din `evaluation.MODES`, cu chunk-uri sau cu
părinți. Fiecare rezultat e marcat față de celălalt panou: ★ același chunk (doar între moduri
`knowledge_prep`, chunking-ul `rag_simple` e altul), ◆ același document (cheia comună e numele
fișierului markdown), ○ doar aici.

- **Acoperirea** (accordion-ul de jos) pune alături documentele fiecărei metode: chunk-urile din
  `rag_simple_chunks` și stadiul + chunk-urile din registru. **Compară metodele doar pe
  documentele pe care le au amândouă**; altfel diferența vine din corpus, nu din metodă.
- `rag_simple` se importă leneș (modelele și BM25-ul lui durează câteva secunde) și costă un
  apel de rescriere + un rerank per întrebare; modurile `knowledge_prep` costă un embedding.
- Prima rulare a unei metode e secvențială: importurile leneșe din `openai` rulate în paralel pe
  două fire dau `_DeadlockError`. După ce fiecare metodă a mers o dată, panourile rulează în
  paralel.
- Dacă portul 7860 e ocupat (ex. de `rag_simple/gradio_app.py`), Gradio alege 7861.

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
  `FakeVectors` (embeddings deterministe după lungimea textului, `delete_chunks`/`add` în
  memorie, cu `fail="embed"` / `fail="write"` ca să simuleze un eșec la fiecare pas;
  `writes_ok=N` lasă primele N loturi să treacă, pentru testele de reluare) și
  `FakeQuestions` (întrebări deterministe, înregistrează apelurile primite). `index_document(...)`
  primește `vectors=` și `questioner=` direct; `__main__.main(...)` primește și el `vectors=`
  și `questioner=` (implicit `PgVectorIndex.from_settings()`, respectiv `generate_questions`
  din `questions.py`) — dă-le fake în teste, la fel cum `converters=`/`restructurer=` se dau
  fake pentru `prepare`. Helper-ul `run()` din `test_cli.py` face deja asta pentru `index`.
- **Evaluarea** (`test_evaluation.py`) folosește retriever-e fixe (`FixedRetriever`), fără DB;
  testele CLI pentru `evaluate` (`test_cli.py`) rulează toate modurile pe `FakeChildStore` și un
  `tests.jsonl` temporar.
- `test_vectors_pg.py` e integrare reală pe Postgres cu pgvector (ștergerea și refacerea
  vectorilor unui document, filtrarea după `kind`): se sare automat dacă `.env` nu are o
  bază care răspunde (`docker compose up -d db`), altfel creează un tabel propriu
  `knowledge_vectors_test_<hex>` per test și îl șterge la final — nu atinge `knowledge_vectors`.
  Helper-ul de conexiune (`PG_URL`, `requires_pg`, `has_table`, `drop_table`) e în `tests/pg.py`,
  comun cu `test_retrieval_pg.py`.
- **Retrieval-ul se testează cu fake-uri proprii** (`tests/retrieval_fakes.py`): `add_document`
  scrie un document `indexed` cu chunk-urile date direct în tabelul `chunk` (SQLite in-memory);
  `FakeChildStore` simulează `PGVectorStore` — întoarce copiii într-o ordine fixă (primul = cel
  mai apropiat, indiferent de query) și aplică filtrul (`{"kind": "content"}`) prin egalitate
  simplă pe metadate, ca `PGVectorStore`; înregistrează fiecare apel (`query`, `k`, `filter`) în
  `store.calls`, pentru teste care verifică ce a cerut `MultiVectorRetriever`.
- **`__main__.main(...)` primește și `vectorstore=`** (implicit `None` → `PgVectorIndex.from_settings().store`
  la rulare reală): dă-l fake în teste (`FakeChildStore`), la fel cum `vectors=`/`questioner=` se
  dau fake pentru `index`. Helper-ul `run()` din `test_cli.py` face deja asta.
- `test_retrieval_pg.py` e integrare reală pe Postgres: scrie un document `retrieval-test-<hex>`
  direct în tabelele reale `document`/`chunk` (prin `add_document`) și îl șterge la final
  (`ON DELETE CASCADE` pentru chunk-uri), plus un tabel propriu `knowledge_vectors_test_<hex>`
  pentru vectori (șters la final) — nu atinge `knowledge_vectors`. Acoperă `ChunkDocStore` pe
  rânduri Postgres reale, `ChunkBM25.from_db`, și `build_retriever` cap-coadă (multi-vector,
  baseline, hibrid) cu `DeterministicFakeEmbedding` în loc de OpenAI.
