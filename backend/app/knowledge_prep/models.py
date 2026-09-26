"""Registrul pipeline-ului de cunoștințe: documente, chunk-uri, întrebări.

Spec: docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md („Modelul de date”).
"""

import uuid
from datetime import datetime
from enum import StrEnum, auto

import sqlalchemy as sa
from sqlmodel import Field, SQLModel

from app.models import get_datetime_utc


class DocumentType(StrEnum):
    carte = auto()
    regim = auto()
    nota = auto()
    insight = auto()
    transcriere = auto()
    conversatie = auto()


class Stage(StrEnum):
    uploaded = auto()
    structured = auto()
    reviewed = auto()
    indexed = auto()
    validated = auto()

    @property
    def rank(self) -> int:
        return list(Stage).index(self)


def _enum_column(enum: type[StrEnum]) -> sa.Column:
    # VARCHAR, nu tip enum Postgres: un stadiu/tip nou nu cere ALTER TYPE
    return sa.Column(sa.Enum(enum, native_enum=False, length=20), nullable=False)


class Document(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    slug: str = Field(unique=True, index=True, max_length=255)
    title: str = Field(max_length=500)
    author: str | None = Field(default=None, max_length=255)
    type: DocumentType = Field(sa_column=_enum_column(DocumentType))
    split: bool  # false = documentul întreg e un singur chunk
    stage: Stage = Field(sa_column=_enum_column(Stage))
    original_path: str | None = Field(default=None, unique=True)  # absolut
    markdown_path: str = Field(unique=True)  # absolut; fișierul editat în IDE
    reviewed_hash: str | None = Field(default=None, max_length=64)
    indexed_hash: str | None = Field(default=None, max_length=64)
    last_error: str | None = Field(default=None, sa_type=sa.Text)
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
    )
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
    )


class Chunk(SQLModel, table=True):
    # id-ul e și `doc_id` din metadatele vectorilor-copii
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        foreign_key="document.id", ondelete="CASCADE", index=True
    )
    position: int
    heading_path: list[str] = Field(sa_type=sa.JSON)
    text: str = Field(sa_type=sa.Text)


class Question(SQLModel, table=True):
    # id-ul e și id-ul vectorului-copil
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    chunk_id: uuid.UUID = Field(foreign_key="chunk.id", ondelete="CASCADE", index=True)
    text: str = Field(sa_type=sa.Text)
