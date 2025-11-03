"""Add userprofile and memoryevent tables

Revision ID: a1b2c3d4
Revises: 
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create userprofile table (if not exists)
    op.execute("""
        CREATE TABLE IF NOT EXISTS userprofile (
            user_id TEXT NOT NULL,
            name TEXT,
            timezone TEXT,
            preferred_units TEXT,
            vehicle_vin TEXT,
            vehicle_type TEXT,
            notes JSONB,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            PRIMARY KEY (user_id)
        )
    """)
    
    # Create memoryevent table (if not exists)
    op.execute("""
        CREATE TABLE IF NOT EXISTS memoryevent (
            id UUID NOT NULL,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding_id TEXT,
            score DOUBLE PRECISION,
            source_message_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            PRIMARY KEY (id)
        )
    """)
    
    # Create index on memoryevent.user_id (if not exists)
    op.execute("CREATE INDEX IF NOT EXISTS idx_memoryevent_user_id ON memoryevent(user_id)")


def downgrade() -> None:
    # Drop index
    op.drop_index('idx_memoryevent_user_id', table_name='memoryevent')
    
    # Drop tables
    op.drop_table('memoryevent')
    op.drop_table('userprofile')

