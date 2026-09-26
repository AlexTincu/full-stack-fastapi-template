"""Add knowledge registry: document, chunk, question

Revision ID: 7b3f1c2a9d10
Revises: fe56fa70289e
Create Date: 2026-09-26 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '7b3f1c2a9d10'
down_revision = 'fe56fa70289e'
branch_labels = None
depends_on = None

DOCUMENT_TYPES = ('carte', 'regim', 'nota', 'insight', 'transcriere', 'conversatie')
STAGES = ('uploaded', 'structured', 'reviewed', 'indexed', 'validated')


def upgrade():
    op.create_table(
        'document',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('slug', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('title', sqlmodel.sql.sqltypes.AutoString(length=500), nullable=False),
        sa.Column('author', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('type', sa.Enum(*DOCUMENT_TYPES, name='documenttype', native_enum=False, length=20), nullable=False),
        sa.Column('split', sa.Boolean(), nullable=False),
        sa.Column('stage', sa.Enum(*STAGES, name='stage', native_enum=False, length=20), nullable=False),
        sa.Column('original_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('markdown_path', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('reviewed_hash', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),
        sa.Column('indexed_hash', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('original_path'),
        sa.UniqueConstraint('markdown_path'),
    )
    op.create_index(op.f('ix_document_slug'), 'document', ['slug'], unique=True)
    op.create_table(
        'chunk',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('heading_path', sa.JSON(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['document.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chunk_document_id'), 'chunk', ['document_id'], unique=False)
    op.create_table(
        'question',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('chunk_id', sa.Uuid(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunk.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_question_chunk_id'), 'question', ['chunk_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_question_chunk_id'), table_name='question')
    op.drop_table('question')
    op.drop_index(op.f('ix_chunk_document_id'), table_name='chunk')
    op.drop_table('chunk')
    op.drop_index(op.f('ix_document_slug'), table_name='document')
    op.drop_table('document')
