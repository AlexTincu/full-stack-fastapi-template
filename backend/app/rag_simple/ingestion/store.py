import json
import logging

from langchain_core.documents import Document

from app.rag_simple.config import CHUNKS_PATH, DB_DIR, PGVECTOR_TABLE_NAME
from app.rag_simple.vectorstore import ensure_table, get_vectorstore

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: list[Document]) -> None:
    # Incremental: append to the existing table instead of rebuild_table().
    ensure_table()
    get_vectorstore().add_documents(chunks)
    logger.info("Added %d chunks to Postgres table '%s'", len(chunks), PGVECTOR_TABLE_NAME)

    # chunks.json must mirror the whole table (it's the BM25 corpus), so append too.
    DB_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(CHUNKS_PATH.read_text(encoding="utf-8")) if CHUNKS_PATH.exists() else []
    existing.extend({"page_content": c.page_content, "metadata": c.metadata} for c in chunks)
    CHUNKS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Chunks saved to %s (for BM25 index, %d total)", CHUNKS_PATH, len(existing))
