import logging
import shutil

from app.rag_simple.config import INGESTED_DIR, KNOWLEDGE_DIR
from app.rag_simple.ingestion.load import SourceDocument

logger = logging.getLogger(__name__)


def check_not_ingested(documents: list[SourceDocument]) -> None:
    """Fail before embedding if a file with the same name was already ingested.

    Re-ingesting it would duplicate its chunks in the vectorstore and in chunks.json.
    """
    duplicates = [d.source for d in documents if (INGESTED_DIR / d.source).exists()]
    if duplicates:
        raise FileExistsError(f"Already ingested (found in {INGESTED_DIR}): {duplicates}")


def move_to_ingested(documents: list[SourceDocument]) -> None:
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        shutil.move(KNOWLEDGE_DIR / doc.source, INGESTED_DIR / doc.source)
    logger.info("Moved %d documents to %s", len(documents), INGESTED_DIR)
