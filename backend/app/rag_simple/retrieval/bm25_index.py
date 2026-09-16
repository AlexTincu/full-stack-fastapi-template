import json
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.rag_simple.config import CHUNKS_PATH

# \w with re.UNICODE matches Romanian diacritics (ă, â, î, ș, ț) as word characters.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_chunks() -> list[Document]:
    raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in raw]


# Corpus is small (~200 chunks) and not persisted separately: rebuilding on
# import from db/chunks.json (the same text that was embedded) is instant.
_chunks = _load_chunks()
_bm25 = BM25Okapi([_tokenize(chunk.page_content) for chunk in _chunks])


def lexical_search(query: str, k: int) -> list[Document]:
    """BM25 search over the same contextualized chunk text used for embeddings."""
    scores = _bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_chunks[i] for i in ranked_indices[:k] if scores[i] > 0]
