from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_postgres import PGEngine, PGVectorStore

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
