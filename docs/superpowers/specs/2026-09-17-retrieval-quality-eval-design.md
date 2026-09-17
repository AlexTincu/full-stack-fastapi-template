# Monitorizare cuantificată a calității retrieval-ului — design

Data: 2026-09-17
Status: aprobat pentru implementare (fără dependința `pandas`)

## Context și motivație

`backend/app/rag_simple/` are deja un pipeline de retrieval relativ sofisticat
(`retrieval/pipeline.py`): query rewrite → hybrid search (semantic + BM25) →
Reciprocal Rank Fusion → cross-encoder rerank → top-`FINAL_K`. Fiecare etapă e
deja expusă explicit prin `RetrievalTrace` (`semantic_hits`, `lexical_hits`,
`fused`, `reranked`, `final`), construit special pentru debugging/notebook use.

Există și o infrastructură de evaluare parțială:
- `evaluation/tests.jsonl` — 10 întrebări de test, fiecare cu `keywords` (ground
  truth prin prezența unui cuvânt-cheie în textul chunk-ului recuperat),
  `reference_answer`, `category`.
- `evaluation/eval.py` — `calculate_mrr`, `calculate_ndcg` (calculate pe baza
  keyword-in-content), plus evaluare LLM-as-judge a răspunsului final
  (`AnswerEval`).
- `retrieval/retrieval_playground.ipynb` — notebook de inspecție **manuală**,
  fără nicio metrică: printează chunk-urile de la fiecare etapă ca să le
  citești cu ochiul liber.

Ce lipsește, și e obiectul acestui design: un mod **cuantificat și repetabil**
de a răspunde la întrebarea "schimbarea pe care tocmai am făcut-o (sau vreau
s-o fac) în retrieval chiar ajută, sau doar pare că ajută?" — înainte de a
aduce îmbunătățiri suplimentare (multi-query decomposition, buclă de
suficiență stil CRAG/Self-RAG, activarea contextualize.py etc., discutate
separat), utilizatorul vrea un baseline măsurat și comparabil.

Chunking-ul a fost discutat și exclus explicit din acest design: nu există o
metodologie stabilă/ieftină de măsurat calitatea lui izolat de efectul asupra
retrieval-ului, așa că rămâne evaluat indirect (dacă un keyword nu e găsit,
chunking-ul e una dintre cauzele posibile, diagnosticabilă manual din
breakdown-ul per întrebare de mai jos), nu printr-un notebook dedicat.

## Ce se reutilizează, nemodificat

- `retrieval/pipeline.py::retrieve_with_trace` — sursa unică pentru cele 4
  etape de retrieval.
- `evaluation/eval.py::calculate_mrr`, `calculate_ndcg` — reimportate, nu
  duplicate.
- `evaluation/test.py::load_tests`, `evaluation/tests.jsonl` — setul de 10
  întrebări rămâne neschimbat pentru această primă trecere.
- `config.py` — citit doar (RETRIEVAL_K, FINAL_K etc.), pentru a eticheta
  snapshot-urile cu configurația activă la momentul rulării.

## Componente noi

### `evaluation/retrieval_metrics.py`

Funcții noi, toate operând pe `list[Document]` + `list[str]` (keywords), în
același stil proxy ca `eval.py` (keyword prezent în `doc.page_content`):

```python
def keyword_hit_ranks(keyword: str, docs: list[Document]) -> int | None:
    """Rangul (1-based) primului doc care conține keyword-ul, sau None."""

def recall_at_k(keywords: list[str], docs: list[Document], k: int) -> float:
    """Fracția de keywords găsite în primele k documente."""

def precision_at_k(keywords: list[str], docs: list[Document], k: int) -> float:
    """Media, peste keywords, a fracției din top-k care conțin acel keyword."""

def hit_rate(keywords: list[str], docs: list[Document], k: int) -> float:
    """Fracția de keywords cu cel puțin un hit în top-k (binar per keyword)."""

@dataclass
class StageMetrics:
    stage: str  # "semantic" | "lexical" | "fused" | "reranked"
    mrr: float
    ndcg: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    hit_rate_at_10: float

def evaluate_stage(stage_name: str, docs: list[Document], keywords: list[str]) -> StageMetrics:
    ...

def evaluate_trace(trace: RetrievalTrace, test: TestQuestion) -> dict[str, StageMetrics]:
    """Rulează evaluate_stage pe cele 4 etape ale unui RetrievalTrace."""
```

Import direct din `eval.py` pentru `calculate_mrr`/`calculate_ndcg` — nu se
reimplementează.

### `evaluation/results.py`

```python
@dataclass
class EvalSnapshot:
    label: str                      # ex. "baseline", "contextualize-on"
    timestamp: str                  # ISO 8601
    config: dict                    # RETRIEVAL_K, FINAL_K, CHAT_MODEL, EMBEDDING_MODEL
    stage_metrics: dict[str, dict]  # stage_name -> metrici agregate (medie pe toate întrebările)
    notes: str = ""

    def save(self, results_dir: Path) -> Path: ...

    @classmethod
    def load(cls, path: Path) -> "EvalSnapshot": ...

def list_snapshots(results_dir: Path) -> list[EvalSnapshot]: ...

def diff_snapshots(a: EvalSnapshot, b: EvalSnapshot) -> dict[str, dict]:
    """Delta per stage/metrică între două snapshot-uri, pentru comparație before/after."""
```

Snapshot-urile se salvează ca JSON în `evaluation/results/` (director nou,
comis în git — fișiere mici, fără secrete, utile ca istoric de benchmark).
Numele fișierului: `<timestamp>_<label-slugificat>.json`.

### `evaluation/retrieval_quality.ipynb`

Notebook nou, lângă `eval.py`/`test.py` (nu în `retrieval/`, unde rămâne
`retrieval_playground.ipynb` pentru inspecție liberă — cele două servesc
scopuri diferite: unul e cuantificat și repetabil, celălalt e explorator).

Structură:
1. Setup: `load_dotenv`, import din `retrieval.pipeline`, `evaluation.test`,
   `evaluation.retrieval_metrics`, `evaluation.results`.
2. Rulare: pentru fiecare din cele 10 întrebări din `tests.jsonl`, apel
   `retrieve_with_trace()`, apoi `evaluate_trace()`.
3. **Tabel agregat etapă × metrică** (semantic / lexical / fused / reranked),
   construit cu print/f-strings aliniate manual (fără `pandas`) — arată cât
   adaugă fiecare etapă deja construită.
4. **Breakdown per întrebare**: pentru fiecare (întrebare, keyword) care nu e
   găsit în etapa finală, sau care își schimbă rangul semnificativ între
   etape, se listează sursa + heading-ul chunk-ului relevant (dacă a fost
   găsit la vreo etapă) — reutilizează stilul `show_chunk` din
   `retrieval_playground.ipynb`, ca diagnostic direct al eșecurilor.
5. Celulă de salvare: creează un `EvalSnapshot` cu un `label` editabil de
   utilizator și îl salvează în `evaluation/results/`.
6. Celulă de comparație: încarcă două snapshot-uri (implicit: ultimele două
   salvate) și afișează delta per etapă/metrică.

## Ce nu se schimbă / e explicit în afara scopului

- Niciun fișier existent din `rag_simple/` nu se modifică.
- Nicio dependință nouă în `pyproject.toml` (renunțat la `pandas` — tabelele
  se construiesc cu print/f-strings, ca în `eval.py` actual).
- Evaluarea calității răspunsului (LLM-as-judge din `eval.py`) nu e integrată
  în acest notebook.
- Fără notebook dedicat calității chunking-ului (decizie explicită, vezi
  Context).
- Fără extinderea `tests.jsonl` la mai multe întrebări — rămâne la cele 10
  existente pentru această primă trecere; extinderea e un pas ulterior,
  condiționat de ce lipsuri arată prima rulare.

## Validare

Nu există teste automate clasice pentru un notebook — validarea e: notebook-ul
rulează cap-coadă fără erori pe corpusul curent (96 chunks / 5 documente),
produce un tabel agregat plauzibil (fused ≥ semantic-only și lexical-only pe
majoritatea metricilor, altfel fuziunea nu ajută și asta e deja un semnal
util), și un snapshot JSON valid salvat pe disc, reîncărcabil.
