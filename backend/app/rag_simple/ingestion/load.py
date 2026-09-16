import logging
from dataclasses import dataclass

from app.rag_simple.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    source: str
    text: str


def load_documents() -> list[SourceDocument]:
    documents = []
    for file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        documents.append(SourceDocument(source=file.name, text=text))
    logger.info("Loaded %d documents from %s", len(documents), KNOWLEDGE_DIR)
    return documents
