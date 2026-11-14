"""reel edit v2 (pieces), rally detected bounds, correction log

- rally.detected_start/detected_end: the detector's bounds, kept through user
  corrections. Existing rallies are all the detector's, so they're copied from
  start/end.
- correction: operation_id, user_id, result_rally_id (the table was unused so far,
  so operation_id can be NOT NULL right away).
- reel.edit v1 -> v2: each clip's single trim becomes one piece; lead_in/lead_out
  are set so the old trims still fit (default 1 s, at most 3 s; anything further out
  is trimmed). Clips without a rally (v1 allowed rally_id null) are dropped.

The conversions are written out here, not imported from the app: a migration must
keep doing exactly this even after the app's code changes.

Revision ID: 6acbb05a9b0b
Revises: b4da243b2693
Create Date: 2026-11-11 00:20:13.207465

"""
import math
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '6acbb05a9b0b'
down_revision = 'b4da243b2693'
branch_labels = None
depends_on = None

MAX_LEAD = 3.0
DEFAULT_LEAD = 1.0
DEFAULT_OVERLAYS = {"trail": False, "captions": []}

reel = sa.table(
    "reel",
    sa.column("id", sa.Uuid()),
    sa.column("video_id", sa.Uuid()),
    sa.column("edit", postgresql.JSONB()),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
rally = sa.table(
    "rally",
    sa.column("id", sa.Integer()),
    sa.column("video_id", sa.Uuid()),
    sa.column("start", sa.Float()),
    sa.column("end", sa.Float()),
    sa.column("detected_start", sa.Float()),
    sa.column("detected_end", sa.Float()),
)


def _lead(needed: float) -> float:
    """Lead-in/out that fits the old trims: at least the default, at most 3 s,
    rounded up to 0.01 s."""
    return min(MAX_LEAD, max(DEFAULT_LEAD, math.ceil(round(needed * 100, 6)) / 100))


def edit_v1_to_v2(
    edit: dict[str, Any], rallies: dict[int, tuple[float, float]]
) -> dict[str, Any]:
    if edit.get("version") != 1:
        return edit
    clips = [
        (clip, rallies[clip["rally_id"]])
        for clip in edit.get("clips", [])
        if clip.get("rally_id") in rallies
    ]
    lead_in = _lead(max((r[0] - c["trim_start"] for c, r in clips), default=0.0))
    lead_out = _lead(max((c["trim_end"] - r[1] for c, r in clips), default=0.0))
    new_clips = []
    for clip, (start, end) in clips:
        piece_start = max(clip["trim_start"], start - lead_in, 0.0)
        piece_end = min(clip["trim_end"], end + lead_out)
        if piece_end - piece_start > 1e-6:
            new_clips.append({
                "rally_id": clip["rally_id"],
                "pieces": [{"start": piece_start, "end": piece_end,
                            "slow_mo": bool(clip.get("slow_mo", False))}],
            })  # fmt: skip
    return {
        "version": 2,
        "view": edit.get("view", "cut"),
        "format": edit.get("format", "16:9"),
        "lead_in": lead_in,
        "lead_out": lead_out,
        "clips": new_clips,
        "restored_gaps": edit.get("restored_gaps", []),
        "overlays": edit.get("overlays", DEFAULT_OVERLAYS),
    }


def edit_v2_to_v1(edit: dict[str, Any]) -> dict[str, Any]:
    """Downgrade: a clip's pieces become one trim from the first piece's start to
    the last one's end (gaps between pieces are lost); slow-mo if any piece was."""
    if edit.get("version") != 2:
        return edit
    clips = [
        {
            "rally_id": clip["rally_id"],
            "trim_start": clip["pieces"][0]["start"],
            "trim_end": clip["pieces"][-1]["end"],
            "slow_mo": any(p.get("slow_mo", False) for p in clip["pieces"]),
        }
        for clip in edit.get("clips", [])
        if clip.get("pieces")
    ]
    return {
        "version": 1,
        "view": edit.get("view", "cut"),
        "format": edit.get("format", "16:9"),
        "clips": clips,
        "restored_gaps": edit.get("restored_gaps", []),
        "overlays": edit.get("overlays", DEFAULT_OVERLAYS),
    }


def _convert_reels(convert: Any) -> None:
    bind = op.get_bind()
    bounds: dict[Any, dict[int, tuple[float, float]]] = {}
    for reel_id, video_id, edit in bind.execute(
        sa.select(reel.c.id, reel.c.video_id, reel.c.edit)
    ).all():
        if video_id not in bounds:
            bounds[video_id] = {
                r.id: (r.start, r.end)
                for r in bind.execute(
                    sa.select(rally.c.id, rally.c.start, rally.c.end).where(
                        rally.c.video_id == video_id
                    )
                )
            }
        new = convert(edit, bounds[video_id])
        if new != edit:
            # A new updated_at: an editor holding the old version reloads (409)
            bind.execute(
                sa.update(reel)
                .where(reel.c.id == reel_id)
                .values(edit=new, updated_at=sa.func.now())
            )


def upgrade():
    op.add_column('correction', sa.Column('operation_id', sa.Uuid(), nullable=False))
    op.add_column('correction', sa.Column('user_id', sa.Uuid(), nullable=True))
    op.add_column('correction', sa.Column('result_rally_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_correction_operation_id'), 'correction', ['operation_id'], unique=False)
    op.create_foreign_key('correction_user_id_fkey', 'correction', 'user', ['user_id'], ['id'], ondelete='SET NULL')
    op.add_column('rally', sa.Column('detected_start', sa.Float(), nullable=True))
    op.add_column('rally', sa.Column('detected_end', sa.Float(), nullable=True))
    op.execute(
        sa.update(rally).values(
            detected_start=rally.c.start, detected_end=rally.c.end
        )
    )
    _convert_reels(edit_v1_to_v2)


def downgrade():
    _convert_reels(lambda edit, _bounds: edit_v2_to_v1(edit))
    op.drop_column('rally', 'detected_end')
    op.drop_column('rally', 'detected_start')
    op.drop_constraint('correction_user_id_fkey', 'correction', type_='foreignkey')
    op.drop_index(op.f('ix_correction_operation_id'), table_name='correction')
    op.drop_column('correction', 'result_rally_id')
    op.drop_column('correction', 'user_id')
    op.drop_column('correction', 'operation_id')
