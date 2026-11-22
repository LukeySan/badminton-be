"""reel edit v4: clips record deleted footage as cuts

reel.edit v3 -> v4: every clip gets `cuts`, the footage the user deleted, which never
plays and bounds restored breaks. v3 didn't record deletions, so they are
reconstructed: a clip's cuts are the parts of its rally's window that no piece covers.
The window is the editor's (frontend reelEdit.ts rallyWindows): the rally widened by
lead_in / lead_out, inside the video, two close rallies sharing the gap between them
in proportion to the leads; then widened to the clip's own pieces (clipWindow). A
rally used by several clips (a replay) gives its cuts to the first one only.

Downgrade: the cuts are dropped (v3 can't express them).

Data only. The conversions are written out here, not imported from the app: a
migration must keep doing exactly this after the app's code changes.

Revision ID: ec4778601b52
Revises: d769f6b0ca89
Create Date: 2026-11-17 05:10:00.000000

"""
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'ec4778601b52'
down_revision = 'd769f6b0ca89'
branch_labels = None
depends_on = None

MIN_SPAN = 0.05  # the app's: shorter stretches count as nothing
DEFAULT_LEAD = 1.0

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
rally = sa.table(
    "rally",
    sa.column("id", sa.Integer()),
    sa.column("video_id", sa.Uuid()),
    sa.column("start", sa.Float()),
    sa.column("end", sa.Float()),
)

Rallies = dict[int, tuple[float, float]]


def rally_windows(
    rallies: Rallies, lead_in: float, lead_out: float, duration: float | None
) -> dict[int, tuple[float, float]]:
    """The frontend's rallyWindows, step for step."""
    ordered = sorted(rallies.items(), key=lambda item: item[1][0])
    total = lead_in + lead_out

    def boundary(a: tuple[float, float], b: tuple[float, float]) -> float:
        gap = b[0] - a[1]
        if gap <= 0 or total <= 0:
            return (a[1] + b[0]) / 2
        return a[1] + gap * (lead_out / total)

    windows = {}
    for i, (rally_id, bounds) in enumerate(ordered):
        start = max(0.0, bounds[0] - lead_in)
        end = bounds[1] + lead_out
        if duration:
            end = min(end, duration)
        if i > 0:
            start = max(start, boundary(ordered[i - 1][1], bounds))
        if i + 1 < len(ordered):
            end = min(end, boundary(bounds, ordered[i + 1][1]))
        windows[rally_id] = (round(start, 3), round(end, 3))
    return windows


def _free(span: tuple[float, float], covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of span that no covered range overlaps."""
    parts = [span]
    for c_start, c_end in covered:
        parts = [
            part
            for start, end in parts
            for part in ((start, min(end, c_start)), (max(start, c_end), end))
            if part[1] - part[0] > 0
        ]
    return parts


def edit_v3_to_v4(
    edit: dict[str, Any], rallies: Rallies, duration: float | None
) -> dict[str, Any]:
    if edit.get("version") != 3:
        return edit
    windows = rally_windows(
        rallies,
        edit.get("lead_in", DEFAULT_LEAD),
        edit.get("lead_out", DEFAULT_LEAD),
        duration,
    )
    covered = [
        (p["start"], p["end"])
        for clip in edit.get("clips", [])
        for p in clip.get("pieces", [])
    ]
    seen: set[int] = set()
    clips = []
    for clip in edit.get("clips", []):
        pieces = clip.get("pieces", [])
        window = windows.get(clip["rally_id"])
        cuts = []
        if window is not None and pieces and clip["rally_id"] not in seen:
            span = (
                min(window[0], pieces[0]["start"]),
                max(window[1], pieces[-1]["end"]),
            )
            cuts = [
                {"start": round(start, 3), "end": round(end, 3)}
                for start, end in _free(span, covered)
                if end - start >= MIN_SPAN
            ]
        seen.add(clip["rally_id"])
        clips.append({**clip, "cuts": cuts})
    return {**edit, "version": 4, "clips": clips}


def edit_v4_to_v3(edit: dict[str, Any]) -> dict[str, Any]:
    if edit.get("version") != 4:
        return edit
    clips = [
        {k: v for k, v in clip.items() if k != "cuts"}
        for clip in edit.get("clips", [])
    ]
    return {**edit, "version": 3, "clips": clips}


def _convert_reels(convert: Any) -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(reel.c.id, reel.c.video_id, reel.c.edit, video.c.duration).join(
            video, video.c.id == reel.c.video_id
        )
    ).all()
    for reel_id, video_id, edit, duration in rows:
        rallies = {
            r.id: (r.start, r.end)
            for r in bind.execute(
                sa.select(rally.c.id, rally.c.start, rally.c.end).where(
                    rally.c.video_id == video_id
                )
            )
        }
        new = convert(edit, rallies, duration)
        if new != edit:
            # A new updated_at: an editor holding the old version reloads (409)
            bind.execute(
                sa.update(reel)
                .where(reel.c.id == reel_id)
                .values(edit=new, updated_at=sa.func.now())
            )


def upgrade():
    _convert_reels(edit_v3_to_v4)


def downgrade():
    _convert_reels(lambda edit, _rallies, _duration: edit_v4_to_v3(edit))
