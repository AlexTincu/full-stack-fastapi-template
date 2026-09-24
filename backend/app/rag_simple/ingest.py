import logging

from app.rag_simple.ingestion.archive import check_not_ingested, move_to_ingested
from app.rag_simple.ingestion.chunk import split_documents
from app.rag_simple.ingestion.contextualize import add_context
from app.rag_simple.ingestion.load import load_documents
from app.rag_simple.ingestion.store import build_vectorstore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ingest() -> None:    
    documents = load_documents()
    if not documents:
        logger.info("Nothing to ingest")
        raise SystemExit(0)
    check_not_ingested(documents)
    chunks = split_documents(documents)
    # chunks = add_context(chunks, documents)
    build_vectorstore(chunks)
    move_to_ingested(documents)
    logger.info("Ingestion complete")

if __name__ == "__main__":
    ingest()
