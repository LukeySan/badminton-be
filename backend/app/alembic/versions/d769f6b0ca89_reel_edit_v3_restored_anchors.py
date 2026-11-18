"""reel edit v3: restored breaks are anchors

reel.edit v2 -> v3: `restored_gaps` [{start, end}] becomes `restored_anchors`
[time]: one time inside each restored break, the midpoint of the old range. A break
is then "the hidden footage between neighbouring pieces that contains an anchor", so
it follows its pieces when rally fixes or lead changes move them (a stored range
didn't). Anchors that end up inside a piece are ignored, and dropped on the next save.

Downgrade: each anchor becomes the range it restores now (worked out from the pieces
and the video's duration, the v3 rule); anchors that restore nothing are dropped.

Data only, no schema change. The conversions are written out here, not imported from
the app: a migration must keep doing exactly this after the app's code changes.

Revision ID: d769f6b0ca89
Revises: bb84033178f8
Create Date: 2026-11-17 03:40:00.000000

"""
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd769f6b0ca89'
down_revision = 'bb84033178f8'
branch_labels = None
depends_on = None

MIN_SPAN = 0.05  # the app's: shorter stretches count as nothing

reel = sa.table(
    "reel",
    sa.column("id", sa.Uuid()),
    sa.column("video_id", sa.Uuid()),
    sa.column("edit", postgresql.JSONB()),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
video = sa.table(
    "video",
    sa.column("id", sa.Uuid()),
    sa.column("duration", sa.Float()),
)


def edit_v2_to_v3(edit: dict[str, Any]) -> dict[str, Any]:
    if edit.get("version") != 2:
        return edit
    new = {k: v for k, v in edit.items() if k != "restored_gaps"}
    new["version"] = 3
    new["restored_anchors"] = sorted(
        round((gap["start"] + gap["end"]) / 2, 3)
        for gap in edit.get("restored_gaps", [])
    )
    return new


def _hidden_regions(
    edit: dict[str, Any], duration: float | None
) -> list[tuple[float, float]]:
    limit = float("inf") if duration is None else duration
    pieces = sorted(
        (max(0.0, p["start"]), min(p["end"], limit))
        for clip in edit.get("clips", [])
        for p in clip.get("pieces", [])
    )
    covered: list[tuple[float, float]] = []
    for start, end in pieces:
        if end - start < MIN_SPAN:
            continue
        if covered and start <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], end))
        else:
            covered.append((start, end))
    regions, cursor = [], 0.0
    for start, end in covered:
        if start - cursor >= MIN_SPAN:
            regions.append((cursor, start))
        cursor = end
    if limit != float("inf") and limit - cursor >= MIN_SPAN:
        regions.append((cursor, limit))
    return regions


def edit_v3_to_v2(edit: dict[str, Any], duration: float | None) -> dict[str, Any]:
    if edit.get("version") != 3:
        return edit
    anchors = edit.get("restored_anchors", [])
    new = {k: v for k, v in edit.items() if k != "restored_anchors"}
    new["version"] = 2
    new["restored_gaps"] = [
        {"start": round(start, 3), "end": round(end, 3)}
        for start, end in _hidden_regions(edit, duration)
        if any(start < a < end for a in anchors)
    ]
    return new


def _convert_reels(convert: Any) -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(reel.c.id, reel.c.edit, video.c.duration).join(
            video, video.c.id == reel.c.video_id
        )
    ).all()
    for reel_id, edit, duration in rows:
        new = convert(edit, duration)
        if new != edit:
            # A new updated_at: an editor holding the old version reloads (409)
            bind.execute(
                sa.update(reel)
                .where(reel.c.id == reel_id)
                .values(edit=new, updated_at=sa.func.now())
            )


def upgrade():
    _convert_reels(lambda edit, _duration: edit_v2_to_v3(edit))


def downgrade():
    _convert_reels(edit_v3_to_v2)
