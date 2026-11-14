"""one live reel per video

Beta: a video has at most one reel that isn't deleted (partial unique index on
reel.video_id WHERE deleted_at IS NULL). Videos that already have several keep the
most recently updated one; the others are soft-deleted (deleted_at set, rows kept),
so nothing is lost and they could be restored by hand.

Downgrade drops the index only: the extra reels stay soft-deleted.

Revision ID: 9bd7ddc4eed1
Revises: 6acbb05a9b0b
Create Date: 2026-11-12 19:57:14.207394

"""
import uuid
from collections.abc import Iterable
from datetime import datetime

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '9bd7ddc4eed1'
down_revision = '6acbb05a9b0b'
branch_labels = None
depends_on = None

reel = sa.table(
    "reel",
    sa.column("id", sa.Uuid()),
    sa.column("video_id", sa.Uuid()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
    sa.column("deleted_at", sa.DateTime(timezone=True)),
)


def reels_to_retire(
    live: Iterable[tuple[uuid.UUID, uuid.UUID, datetime, datetime]],
) -> list[uuid.UUID]:
    """(id, video_id, updated_at, created_at) of live reels -> ids of all but the
    one to keep per video: the most recently updated (then the newest, then the
    highest id, so the choice is always the same)."""
    keep: dict[uuid.UUID, tuple[datetime, datetime, uuid.UUID]] = {}
    for reel_id, video_id, updated_at, created_at in live:
        candidate = (updated_at, created_at, reel_id)
        if video_id not in keep or candidate > keep[video_id]:
            keep[video_id] = candidate
    kept = {key[2] for key in keep.values()}
    return [row[0] for row in live if row[0] not in kept]


def upgrade():
    bind = op.get_bind()
    live = bind.execute(
        sa.select(reel.c.id, reel.c.video_id, reel.c.updated_at, reel.c.created_at)
        .where(reel.c.deleted_at.is_(None))
    ).all()
    retire = reels_to_retire([tuple(row) for row in live])
    if retire:
        bind.execute(
            sa.update(reel).where(reel.c.id.in_(retire)).values(deleted_at=sa.func.now())
        )
    op.create_index('uq_reel_video_id_live', 'reel', ['video_id'], unique=True, postgresql_where=sa.text('deleted_at IS NULL'))


def downgrade():
    op.drop_index('uq_reel_video_id_live', table_name='reel', postgresql_where=sa.text('deleted_at IS NULL'))
