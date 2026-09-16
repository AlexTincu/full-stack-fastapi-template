from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag_simple.config import FINAL_K, RETRIEVAL_K
from app.rag_simple.retrieval.bm25_index import lexical_search
from app.rag_simple.retrieval.fusion import reciprocal_rank_fusion
from app.rag_simple.retrieval.query_rewrite import rewrite_query
from app.rag_simple.retrieval.rerank import rerank
from app.rag_simple.retrieval.semantic_search import semantic_search


@dataclass
class RetrievalTrace:
    """Every intermediate stage of a retrieval call, for debugging/notebook use."""

    question: str
    rewritten_question: str
    semantic_hits: list[Document]
    lexical_hits: list[Document]
    fused: list[Document]
    reranked: list[Document]

    @property
    def final(self) -> list[Document]:
        return self.reranked[:FINAL_K]


def retrieve_with_trace(question: str, history: list[dict] | None = None) -> RetrievalTrace:
    rewritten = rewrite_query(question, history)

    # Query rewrite + hybrid search: search both the original and rewritten
    # query on each index, fusing each pair, then fuse semantic with lexical.
    semantic_hits = reciprocal_rank_fusion(
        [semantic_search(question, RETRIEVAL_K), semantic_search(rewritten, RETRIEVAL_K)]
    )
    lexical_hits = reciprocal_rank_fusion(
        [lexical_search(question, RETRIEVAL_K), lexical_search(rewritten, RETRIEVAL_K)]
    )
    fused = reciprocal_rank_fusion([semantic_hits, lexical_hits])
    reranked = rerank(question, fused)

    return RetrievalTrace(
        question=question,
        rewritten_question=rewritten,
        semantic_hits=semantic_hits,
        lexical_hits=lexical_hits,
        fused=fused,
        reranked=reranked,
    )


def retrieve(question: str, history: list[dict] | None = None) -> list[Document]:
    """Rewrite -> hybrid search (semantic + BM25) -> fuse -> rerank -> top FINAL_K."""
    return retrieve_with_trace(question, history).final
