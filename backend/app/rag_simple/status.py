"""List the documents ingested by rag_simple: `uv run python -m app.rag_simple.status`."""

import sys

from app.rag_simple.ingestion.tracking import (
    adopt_untracked,
    chunk_counts,
    ensure_documents_table,
    tracked_documents,
)
from app.rag_simple.vectorstore import get_sql_engine


def status_rows() -> list[tuple[str, ...]]:
    engine = get_sql_engine()
    ensure_documents_table(engine)
    counts = chunk_counts(engine)
    adopt_untracked(engine, counts)
    rows = [("source", "chunks", "ingested at", "hash", "warning")]
    for doc in tracked_documents(engine).values():
        stored = counts.get(doc.source, 0)
        warning = "" if stored == doc.chunks else f"{stored} chunks in the vectorstore"
        rows.append((
            doc.source,
            str(doc.chunks),
            doc.ingested_at.strftime("%Y-%m-%d %H:%M") if doc.ingested_at else "before tracking",
            doc.content_hash[:12] if doc.content_hash else "-",
            warning,
        ))
    return rows


def main() -> None:
    rows = status_rows()
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        line = "  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True))
        sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.write(f"\n{len(rows) - 1} documents\n")


if __name__ == "__main__":
    main()
