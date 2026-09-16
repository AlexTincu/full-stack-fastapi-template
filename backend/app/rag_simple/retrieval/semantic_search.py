from langchain_core.documents import Document

from app.rag_simple.vectorstore import get_vectorstore


def semantic_search(query: str, k: int) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k)
