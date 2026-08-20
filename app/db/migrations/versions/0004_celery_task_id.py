"""store the celery task id on the run

Revision ID: 0004
Revises: 0003
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Revocation targets a task id, so DELETE /runs/{id} needs the mapping from
    # run to task. 155 chars is Celery's own task id column width.
    op.add_column("agent_runs", sa.Column("celery_task_id", sa.String(155), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "celery_task_id")
