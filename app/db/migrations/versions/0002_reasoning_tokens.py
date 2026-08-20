"""track reasoning tokens per run

Revision ID: 0002
Revises: 0001
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Reasoning models bill thinking separately from visible output, and
    # `tokens_used` alone hides it. server_default backfills existing rows.
    op.add_column(
        "agent_runs",
        sa.Column("reasoning_tokens", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "reasoning_tokens")
