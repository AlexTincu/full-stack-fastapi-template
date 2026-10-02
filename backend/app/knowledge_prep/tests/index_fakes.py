"""Fake-uri pentru indexare: vectori în memorie și întrebări deterministe (fără OpenAI, Gemini, pgvector)."""

import uuid
from collections.abc import Sequence

from app.knowledge_prep.vectors import VectorEntry


class FakeVectors:
    """`fail="embed"`: cade la embeddings; `fail="write"`: cade la `add`, după `writes_ok` scrieri reușite."""

    def __init__(self, fail: str | None = None, writes_ok: int = 0):
        self.rows: dict[uuid.UUID, VectorEntry] = {}
        self.fail = fail
        self.writes_ok = writes_ok
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail == "embed":
            raise RuntimeError("OpenAI căzut")
        self.embedded += texts
        return [[float(len(t))] for t in texts]

    def delete_chunks(self, chunk_ids: Sequence[uuid.UUID]) -> None:
        gone = set(chunk_ids)
        self.rows = {k: e for k, e in self.rows.items() if e.doc_id not in gone}

    def add(
        self, entries: Sequence[VectorEntry], embeddings: Sequence[list[float]]
    ) -> None:
        assert len(entries) == len(embeddings)
        if self.fail == "write":
            if self.writes_ok == 0:
                raise RuntimeError("pgvector căzut")
            self.writes_ok -= 1
        self.rows.update({e.id: e for e in entries})

    def of(self, document_id: uuid.UUID, kind: str | None = None) -> list[VectorEntry]:
        return [
            e
            for e in self.rows.values()
            if e.document_id == document_id and (kind is None or e.kind == kind)
        ]


class FakeQuestions:
    def __init__(self, per_chunk: int = 2):
        self.per_chunk = per_chunk
        self.calls: list[tuple[str, list[str], str, bool]] = []

    def __call__(
        self, title: str, heading_path: list[str], text: str, whole_document: bool
    ) -> list[str]:
        self.calls.append((title, list(heading_path), text, whole_document))
        return [
            f"Întrebarea {i} despre {heading_path[-1]}?"
            for i in range(1, self.per_chunk + 1)
        ]
