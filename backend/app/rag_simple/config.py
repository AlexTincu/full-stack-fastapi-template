import os
from pathlib import Path

from app.core.config import settings

# langchain / litellm read the API key from the environment, not from our
# Settings object directly (pydantic-settings parses .env into its own
# fields only). Bridge it once here, since every rag_simple module ends up
# importing this file before it needs to call a model or embedding.
os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)

ROOT = Path(__file__).parent

# Model ids use the "provider:model" form accepted by init_chat_model /
# init_embeddings, so swapping providers later is a one-line change here.
CHAT_MODEL = "openai:gpt-4.1-mini"
EMBEDDING_MODEL = "openai:text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # must match EMBEDDING_MODEL's output size

KNOWLEDGE_DIR = settings.KNOWLEDGE_DIR
DB_DIR = ROOT / "db"
CHUNKS_PATH = DB_DIR / "chunks.json"

PGVECTOR_CONNECTION = str(settings.DATABASE_URL)
PGVECTOR_TABLE_NAME = "rag_simple_chunks"

RETRIEVAL_K = 20
FINAL_K = 20
