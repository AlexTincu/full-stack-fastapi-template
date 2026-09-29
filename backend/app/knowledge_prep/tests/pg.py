"""Baza Postgres din `.env` pentru testele de integrare; se sar dacă nu răspunde (`docker compose up -d db`)."""

import pytest
from sqlalchemy import create_engine, inspect, text


def _pg_url() -> str | None:
    try:
        from app.core.config import settings

        url = str(settings.DATABASE_URL)
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
        return url
    except Exception:
        return None


PG_URL = _pg_url()
requires_pg = pytest.mark.skipif(
    PG_URL is None, reason="Postgres cu pgvector indisponibil (docker compose up -d db)"
)


def has_table(name: str) -> bool:
    if PG_URL is None:
        return False
    engine = create_engine(PG_URL)
    try:
        return inspect(engine).has_table(name)
    finally:
        engine.dispose()


def drop_table(name: str) -> None:
    assert PG_URL is not None
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
    engine.dispose()
