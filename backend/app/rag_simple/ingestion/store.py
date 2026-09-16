import json
import logging

from langchain_core.documents import Document

from app.rag_simple.config import CHUNKS_PATH, DB_DIR, PGVECTOR_TABLE_NAME
from app.rag_simple.vectorstore import get_vectorstore, rebuild_table

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: list[Document]) -> None:
    rebuild_table()
    get_vectorstore().add_documents(chunks)
    logger.info("Vectorstore created with %d chunks in Postgres table '%s'", len(chunks), PGVECTOR_TABLE_NAME)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    
    logger.info("Chunks saved to %s (for BM25 index)", CHUNKS_PATH)
