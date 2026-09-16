# Plan de implementare RAG

Document de continuitate: ce am decis, de ce, și ce urmează — ca să putem relua lucrul într-o sesiune nouă fără să reluăm discuția de la zero.

## Context

Acesta e Milestone 1 din viziunea mai largă descrisă în [`../plan.md`](../plan.md) (proiect de AI + medicină naturistă/spiritualitate). RAG-ul e prima piesă tractabilă: răspunsuri pe bază de cunoștințe verificate, nu cunoaștere generalistă.

Înlocuiește baseline-ul non-LangChain din `../week5/pro_implementation/` (exercițiu de curs, conținut exemplu "Insurellm"), păstrat doar ca referință/inspirație — `week5/` nu se mai dezvoltă.

Ne bazăm pe tehnica **Contextual Retrieval** din [articolul Anthropic](https://www.anthropic.com/engineering/contextual-retrieval), combinată cu hybrid search (semantic + BM25), reranking și query rewriting.

**Definiție corectă Contextual Retrieval** (contează, pentru că e ușor de înțeles greșit): NU înseamnă că lași LLM-ul să inventeze granițele chunk-urilor. Rețeta e:
1. Chunking determinist (pe structură/headere, apoi pe dimensiune fixă).
2. Pentru fiecare chunk, dai LLM-ului *documentul întreg* + chunk-ul, ceri doar 1-2 propoziții care situează chunk-ul în document.
3. Prepend-uiești acel context generat la chunk, **înainte** de embedding și de indexarea BM25.
4. Combini BM25 + semantic (rank fusion), adaugi reranking, iei top-K (articolul recomandă K=20).

## Decizii stabilite (2026-09-05)

- **Locație**: proiect nou, de sine stătător, în `rag/` la rădăcina repo-ului (nu `week5/langchain_rag/`) — `week5/` e doar inspirație de curs.
- **Sursa de conținut**: `knowledge/ready_for_ingestion/*.md` — folder dedicat, curatoriat manual; orice fișier nou pus acolo devine automat parte din baza de cunoștințe la următorul `ingest.py`. V1 conține cele 5 fișiere despre seminarul de macrobiotică (`01`-`05`), deja împărțite pe secțiuni tematice.
- **Provider LLM/embeddings**: OpenAI pentru moment (cheie deja disponibilă), dar arhitectura trebuie să rămână ușor de schimbat. De aceea `config.py` ține modelele ca string-uri `"provider:model"` folosite prin `init_chat_model` / `init_embeddings` din LangChain — schimbarea providerului = o linie în `config.py` (plus pachetul `langchain-<provider>` instalat).
- **Reranker**: LLM-based (un LLM ordonează chunk-urile după relevanță), ca în `pro_implementation/answer.py`. Ales pentru simplitate — fără cheie API nouă. De revizuit dacă calitatea retrieval-ului o cere (alternative: Cohere Rerank, cross-encoder local).
- **Vector store**: Chroma (persistent, local) — suficient pentru volumul actual de conținut, fără nevoie de infra suplimentară.
- **BM25**: nu ține index persistat separat; se reconstruiește din `db/chunks.json` la nevoie (corpus mic, ~200 chunk-uri, reconstrucția e instant).

## Status pe milestone-uri

### ✅ Milestone 1 — Ingestion (complet, verificat)

Fișiere: `ingestion/load.py`, `ingestion/chunk.py`, `ingestion/contextualize.py`, `ingestion/store.py`, orchestrat de `ingest.py`.

Pipeline:
1. `load.py` — citește toate `.md` din `knowledge/ready_for_ingestion/`.
2. `chunk.py` — `MarkdownHeaderTextSplitter` pe headere `###`/`####` (nivelurile variază între fișiere) → `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (500 tokeni, overlap 75) pentru secțiunile prea lungi.
3. `contextualize.py` — pentru fiecare chunk, LLM primește documentul întreg + chunk-ul, generează 1-2 propoziții de context, le prepend-uie la text. Chunk-ul original rămâne în `metadata["original_text"]`.
4. `store.py` — embeddings (OpenAI `text-embedding-3-large` via `init_embeddings`) → Chroma persistat în `db/` + salvare `db/chunks.json` (text + metadata, sursă pentru BM25 la Milestone 2).

Rezultat ultimei rulări: 5 documente → 188 chunk-uri. Testat manual: contextul generat e corect și relevant (verificat pe eșantion), iar o interogare semantică de test ("cum se tratează cancerul cu macrobiotica") a scos exact secțiunile relevante, inclusiv documentul dedicat.

Rulare: `uv run python ingest.py` (regenerează tot `db/` de la zero — ignorat de git).

### ✅ Milestone 2 — Retrieval (complet, verificat)

Fișiere: `retrieval/query_rewrite.py`, `retrieval/semantic_search.py`, `retrieval/bm25_index.py`, `retrieval/fusion.py`, `retrieval/rerank.py`, orchestrate de `retrieval/pipeline.py`.

Pipeline (`pipeline.retrieve_with_trace` / `pipeline.retrieve`):
1. `query_rewrite.py` — condensează întrebarea + istoricul conversației într-o interogare de căutare specifică (portat din `pro_implementation/answer.py::rewrite_query`, adaptat la `init_chat_model`).
2. Hybrid search, rulat atât pe întrebarea originală cât și pe cea rescrisă:
   - `semantic_search.py` — similarity search pe Chroma.
   - `bm25_index.py` — BM25 (rank-bm25), index reconstruit la import din `db/chunks.json` (pe același text contextualizat folosit la embeddings, tokenizare unicode-aware pentru diacritice).
3. `fusion.py` — reciprocal rank fusion (RRF, k=60): întâi orig+rescris pe fiecare index, apoi semantic+lexical între ele. Deduplicare pe `page_content`.
4. `rerank.py` — LLM-based (structured output cu `with_structured_output`, portat din `pro_implementation/answer.py::rerank`), robust la id-uri omise/duplicate de LLM.
5. Top-K final: `FINAL_K=20` (crescut de la 10 inițial, aliniat cu recomandarea din articolul Anthropic — vezi decizia de mai jos).

`RetrievalTrace` (dataclass din `pipeline.py`) expune toate etapele intermediare (`semantic_hits`, `lexical_hits`, `fused`, `reranked`, `final`) pentru debugging.

**Notebook de testare**: [`retrieval_playground.ipynb`](retrieval_playground.ipynb) — listă de întrebări ad-hoc, rulează pipeline-ul complet și afișează rezultatele pe fiecare etapă (semantic, lexical, final reranked). Rulat și verificat manual pe 3 întrebări din domeniul macrobioticii — rezultatele sunt relevante (secțiunile corecte, documentele corecte).

**Decizie nouă (2026-09-05)**: `FINAL_K` crescut de la 10 la 20 — utilizatorul a cerut explicit fluxul "merge/fusion -> reranker -> select Top-20", aliniat cu recomandarea K=20 din articolul Anthropic despre Contextual Retrieval.

### ✅ Milestone 3 — Generation (complet, verificat)

Fișiere: `generation/answer.py` (`answer_question`), `app.py` (Gradio UI la rădăcina `rag/`).

- `generation/answer.py::answer_question(question, history) -> (answer, chunks)` — apelează `retrieval.pipeline.retrieve` pentru context, construiește promptul (system prompt + context + istoric + întrebare) și cheamă `CHAT_MODEL`. Portat din `pro_implementation/answer.py::answer_question`, adaptat la `init_chat_model` și la domeniul macrobioticii (system prompt în română, specific bazei de cunoștințe curente).
- `app.py` — UI Gradio (chat + panou context regăsit), portat din `week5/app.py`, doar retitlat/tradus; rulare: `uv run python app.py`.

Testat manual (headless, fără UI): întrebarea despre tratamentul cancerului a produs un răspuns corect, complet, bazat exclusiv pe cele 2 documente sursă relevante (`01 ... cosmos.md`, `04 ... tratamentul cancerului.md`).

Notă: `implementation`-ul original avea un CLI REPL propus ca alternativă; s-a ales direct Gradio pentru că userul are deja `week5/app.py` funcțional ca referință și vrea să-l refolosească.

### ⬜ Milestone 4 — Evaluare (opțional, propus de Claude, nu cerut explicit)

Set mic de întrebări cu răspunsuri/chunk-uri așteptate, ca să măsurăm obiectiv impactul schimbărilor la chunking/hybrid/rerank — altfel orice optimizare ulterioară e "din ochi". De decis dacă se face acum sau mai târziu.

### ⬜ Milestone 5 — LangGraph server + agent-chat-ui

Abia după ce Milestone 2-3 funcționează bine în CLI. Expunem pipeline-ul ca agent LangGraph, conectăm `agent-chat-ui` ca frontend.

## Note pentru sesiuni viitoare

- Nu reluați deciziile de mai sus de la zero — au fost stabilite deliberat pentru viteză în v1. Redeschideți-le doar dacă utilizatorul cere explicit sau dacă un milestone (ex. adăugarea unui reranker real) e atins.
- Când se adaugă conținut nou în `knowledge/ready_for_ingestion/`, doar rulați din nou `ingest.py` — nu necesită cod nou.
- Acest fișier trebuie actualizat pe măsură ce milestone-urile avansează sau deciziile se schimbă (marcați ✅/🚧/⬜ și adăugați note noi de decizie cu dată).
