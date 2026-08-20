"""add needs_input run status

Revision ID: 0003
Revises: 0002
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A run that finished but needs a detail it was never given: not COMPLETED,
    # because the question is unanswered; not FAILED, because nothing broke.
    op.execute("ALTER TYPE run_status ADD VALUE IF NOT EXISTS 'needs_input'")


def downgrade() -> None:
    # Postgres cannot drop a value from an enum without rebuilding the type, and
    # rebuilding it would rewrite agent_runs. Left in place deliberately.
    pass
