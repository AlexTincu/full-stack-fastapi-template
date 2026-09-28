# `knowledge_prep` — pipeline-ul de cunoștințe

Transformă sursele corpusului „Calea de Autovindecare” (PDF, audio, `.txt`/`.md`) în
markdown curat și structurat, urmărit într-un registru Postgres, ca apoi să fie indexat pentru
RAG.

- **De ce și ce s-a decis:** spec-ul
  [`docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md`](../../../docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md).
  Are prioritate față de acest README la „ce trebuie să facă”.
- **Cum s-a implementat, pas cu pas:** planurile din
  [`docs/superpowers/plans/`](../../../docs/superpowers/plans/) (`2026-09-26-knowledge-registry-plan.md`
  pentru faza 1, `2026-09-28-knowledge-restructure-plan.md` pentru faza 2).
- **Acest README:** cum funcționează codul azi, de ce arată așa și ce trebuie știut înainte de
  a-l modifica.

## Starea fazelor

| Fază | Ce | Stare |
|---|---|---|
| 1 | Registrul: `register`, `prepare` (doar conversie), `review`, `validate`, `status` | ✅ |
| 2 | Restructurarea cu Gemini, în `prepare` | ✅ (2026-09-28) |
| 3 | Indexarea: chunking, întrebări, `knowledge_vectors`, `index` | următoarea |
| 4 | Retrieval + evaluare | — |
| 5 | REST | — |

## Fluxul

```
register ──> uploaded ──prepare──> structured ──review──> reviewed ──index──> indexed ──validate──> validated
                                        ▲
register --structured ──────────────────┘
```

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
uv run python -m app.knowledge_prep status [<țintă>]

uv run pytest app/knowledge_prep                            # fără Postgres și fără Gemini
uv run ruff check app/knowledge_prep && uv run ty check app/knowledge_prep
```

Ai nevoie de `GEMINI_API_KEY` și `GEMINI_MODEL` în `full-stack-fastapi-template/.env`.

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
