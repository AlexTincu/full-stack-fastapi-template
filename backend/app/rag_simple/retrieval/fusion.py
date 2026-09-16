from langchain_core.documents import Document

RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Merge several ranked result lists into one, by reciprocal rank fusion.

    Documents are deduplicated by page_content (identical across the semantic
    and lexical indexes, since both are built from the same contextualized
    chunk text).
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_by_key[key] = doc
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ranked_keys]
