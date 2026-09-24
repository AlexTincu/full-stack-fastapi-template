from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import create_engine, inspect

from app.rag_simple.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PGVECTOR_CONNECTION,
    PGVECTOR_TABLE_NAME,
)


@lru_cache
def get_engine() -> PGEngine:
    return PGEngine.from_connection_string(url=PGVECTOR_CONNECTION)


@lru_cache
def get_vectorstore() -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=get_engine(),
        table_name=PGVECTOR_TABLE_NAME,
        embedding_service=init_embeddings(EMBEDDING_MODEL),
    )


def rebuild_table() -> None:
    """Drop and recreate the vectorstore table, mirroring the old rmtree-and-rebuild.

    Requires the Postgres server to have the pgvector extension available
    (e.g. the `pgvector/pgvector` image); this call runs
    `CREATE EXTENSION IF NOT EXISTS vector` itself.
    """
    get_engine().init_vectorstore_table(
        table_name=PGVECTOR_TABLE_NAME,
        vector_size=EMBEDDING_DIMENSIONS,
        overwrite_existing=True,
    )
    get_vectorstore.cache_clear()


def ensure_table() -> None:
    """Create the vectorstore table only if it doesn't exist yet (incremental ingest).

    `init_vectorstore_table` issues a plain CREATE TABLE, so it can't be called
    unconditionally once the table holds data.
    """
    sync_engine = create_engine(PGVECTOR_CONNECTION)
    try:
        exists = inspect(sync_engine).has_table(PGVECTOR_TABLE_NAME)
    finally:
        sync_engine.dispose()
    if not exists:
        get_engine().init_vectorstore_table(
            table_name=PGVECTOR_TABLE_NAME,
            vector_size=EMBEDDING_DIMENSIONS,
        )
        get_vectorstore.cache_clear()
