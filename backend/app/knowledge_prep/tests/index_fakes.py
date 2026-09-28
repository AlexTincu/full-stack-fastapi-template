"""Fake-uri pentru indexare: vectori în memorie și întrebări deterministe (fără OpenAI, Gemini, pgvector)."""

import uuid
from collections.abc import Sequence

from app.knowledge_prep.vectors import VectorEntry


class FakeVectors:
    """`fail="embed"`: cade la embeddings; `fail="write"`: șterge vectorii vechi, apoi cade."""

    def __init__(self, fail: str | None = None):
        self.rows: dict[uuid.UUID, VectorEntry] = {}
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail == "embed":
            raise RuntimeError("OpenAI căzut")
        return [[float(len(t))] for t in texts]

    def replace_document(
        self,
        document_id: uuid.UUID,
        entries: Sequence[VectorEntry],
        embeddings: Sequence[list[float]],
    ) -> None:
        assert len(entries) == len(embeddings)
        self.rows = {k: e for k, e in self.rows.items() if e.document_id != document_id}
        if self.fail == "write":
            raise RuntimeError("pgvector căzut")
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
