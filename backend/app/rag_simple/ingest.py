import logging

from app.rag_simple.ingestion.chunk import split_documents
from app.rag_simple.ingestion.contextualize import add_context
from app.rag_simple.ingestion.load import load_documents
from app.rag_simple.ingestion.store import build_vectorstore

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    documents = load_documents()
    chunks = split_documents(documents)
    contextualized_chunks = add_context(chunks, documents)
    build_vectorstore(contextualized_chunks)
    logger.info("Ingestion complete")
