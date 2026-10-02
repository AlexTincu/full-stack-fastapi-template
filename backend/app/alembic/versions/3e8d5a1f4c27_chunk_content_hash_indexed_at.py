"""Chunk: content_hash + indexed_at (indexare incrementală, reluabilă)

Revision ID: 3e8d5a1f4c27
Revises: 7b3f1c2a9d10
Create Date: 2026-09-30 18:00:00.000000

Completează coloanele pentru chunk-urile existente: `content_hash` din textul lor, iar
`indexed_at` acolo unde vectorul `content` există deja în `knowledge_vectors` (vechiul `index`
scria toți vectorii unui document dintr-o dată). Un document rămas `reviewed` după o rulare
întreruptă, dar cu vectorii scriși, devine `indexed` la următorul `index`, fără apeluri API.
"""
import hashlib

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op

# revision identifiers, used by Alembic.
revision = '3e8d5a1f4c27'
down_revision = '7b3f1c2a9d10'
branch_labels = None
depends_on = None


def _chunk_hash(heading_path: list[str], text: str) -> str:
    # copia lui app.knowledge_prep.chunking.chunk_hash (sha256 pe embedding_text)
    return hashlib.sha256(f"{' > '.join(heading_path)}\n\n{text}".encode()).hexdigest()


def upgrade():
    op.add_column('chunk', sa.Column('content_hash', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True))
    op.add_column('chunk', sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True))

    conn = op.get_bind()
    chunk = sa.table(
        'chunk',
        sa.column('id', sa.Uuid()),
        sa.column('heading_path', sa.JSON()),
        sa.column('text', sa.Text()),
        sa.column('content_hash', sa.String()),
    )
    rows = conn.execute(sa.select(chunk.c.id, chunk.c.heading_path, chunk.c.text)).all()
    if rows:
        conn.execute(
            chunk.update().where(chunk.c.id == sa.bindparam('cid')).values(content_hash=sa.bindparam('h')),
            [{'cid': r.id, 'h': _chunk_hash(r.heading_path, r.text)} for r in rows],
        )
    if sa.inspect(conn).has_table('knowledge_vectors'):
        conn.execute(sa.text(
            "UPDATE chunk SET indexed_at = now() WHERE EXISTS ("
            " SELECT 1 FROM knowledge_vectors v WHERE v.doc_id = chunk.id AND v.kind = 'content')"
        ))


def downgrade():
    op.drop_column('chunk', 'indexed_at')
    op.drop_column('chunk', 'content_hash')
