"""Retrieval peste indexul fazei 3 (spec 2026-09-25, „Retrieval”).

- **multi-vector:** `MultiVectorRetriever` peste `knowledge_vectors` (copiii `content` + `question`),
  cu părinții din `ChunkDocStore`;
- **baseline:** același retriever, doar cu copiii `content` (`filter={"kind": "content"}`);
- **hibrid:** oricare dintre cele două + BM25 pe `chunk.text` (`lexical.py`), combinate prin
  reciprocal rank fusion pe `chunk.id`.

Evaluarea, rescrierea query-ului și reranking-ul nu sunt aici (vin după evaluare).
"""

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
