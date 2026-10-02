import argparse
import logging
from collections import Counter
from pathlib import Path

from app.rag_simple.config import EMBEDDING_MODEL, KNOWLEDGE_DIR
from app.rag_simple.ingestion.chunk import split_documents
from app.rag_simple.ingestion.contextualize import add_context
from app.rag_simple.ingestion.load import load_documents
from app.rag_simple.ingestion.store import build_vectorstore
from app.rag_simple.ingestion.tracking import (
    adopt_untracked,
    chunk_counts,
    ensure_documents_table,
    plan_ingest,
    record_ingested,
    tracked_documents,
)
from app.rag_simple.vectorstore import get_sql_engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ingest(folder: Path = KNOWLEDGE_DIR) -> None:
    """Ingest the *.md files in `folder` not ingested yet; files stay where they are."""
    engine = get_sql_engine()
    ensure_documents_table(engine)
    for source in adopt_untracked(engine, chunk_counts(engine)):
        logger.info("Recorded %s (ingested before tracking, no content hash)", source)

    plan = plan_ingest(load_documents(folder), tracked_documents(engine))
    for source in plan.unchanged:
        logger.info("Skipped %s: already ingested", source)
    for source in plan.changed:
        logger.warning(
            "Skipped %s: already ingested with other (or unknown) content; "
            "re-ingesting would duplicate its chunks", source
        )
    if not plan.to_ingest:
        logger.info("Nothing to ingest")
        return

    chunks = split_documents(plan.to_ingest)
    # chunks = add_context(chunks, plan.to_ingest)
    build_vectorstore(chunks)
    record_ingested(
        engine,
        plan.to_ingest,
        Counter(c.metadata["source"] for c in chunks),
        EMBEDDING_MODEL,
    )
    logger.info("Ingestion complete: %d documents", len(plan.to_ingest))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m app.rag_simple.ingest")
    parser.add_argument(
        "folder", nargs="?", type=Path, default=KNOWLEDGE_DIR,
        help=f"folder with *.md files (default: KNOWLEDGE_DIR = {KNOWLEDGE_DIR})",
    )
    ingest(parser.parse_args().folder)
