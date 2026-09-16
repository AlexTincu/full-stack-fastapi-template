# RAG — calea de autovindecare

Pipeline RAG (LangChain) peste cunoștințele din folderul indicat de `KNOWLEDGE_DIR` (`.env`, în mod curent `calea_de_autovindecare/knowledge/ready_for_ingestion/`). Prima piesă din viziunea mai largă din [`../plan.md`](../plan.md).

Planul complet, deciziile luate și ce urmează: vezi [`implement_rag.md`](implement_rag.md). Acest README e doar "cum rulez".

## Setup

```bash
cd rag
uv sync
```

`.env` conține `OPENAI_API_KEY` (copiat din `week5/.env`; nu e commit-uit).

## Rulare

```bash
uv run python ingest.py
```

Citește fișierele `.md` din `KNOWLEDGE_DIR`, le împarte în chunk-uri, generează context per chunk (Contextual Retrieval), creează embeddings și populează vectorstore-ul Chroma în `rag/db/` (ignorat de git — se regenerează oricând rulând `ingest.py`).

```python
from retrieval.pipeline import retrieve

chunks = retrieve("cum se tratează cancerul cu macrobiotica?")
```

Rescrie interogarea, caută hibrid (semantic + BM25), combină prin fusion, reranchează cu LLM, întoarce top-`FINAL_K` chunk-uri (`langchain_core.documents.Document`).

Pentru testare interactivă pe întrebări ad-hoc, cu vizibilitate pe fiecare etapă (semantic/lexical/fused/reranked), vezi notebook-ul [`retrieval_playground.ipynb`](retrieval_playground.ipynb):

```bash
uv run jupyter notebook retrieval_playground.ipynb
```

Generation + UI de chat:

```bash
uv run python app.py
```

Deschide un Gradio cu chat + panoul de context regăsit pentru fiecare răspuns.

## Structură

```
config.py          setări centralizate (model, path-uri, K) — schimbi providerul într-un loc
ingestion/          load -> chunk -> contextualize -> store
retrieval/          query_rewrite -> hybrid search (semantic + BM25) -> fusion -> rerank -> top-K
retrieval_playground.ipynb   notebook de testare ad-hoc pentru retrieval
generation/         answer_question — RAG prompt + apel LLM peste context
app.py              UI Gradio (chat + context regăsit)
db/                 Chroma persistat + chunks.json (regenerabil, ignorat de git)
```
