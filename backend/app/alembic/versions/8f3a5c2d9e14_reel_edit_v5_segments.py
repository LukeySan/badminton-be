"""reel edit v5: a flat list of segments (Editor v2)

reel.edit v4 -> v5: each reel becomes exactly the segments its v4 "Rallies only"
playback played, in the same order, so it plays the same:

- the clips' pieces in reel order (their times already include the lead), cut to
  the video, labelled with their rally (null if the rally no longer exists);
- the restored stretches at normal speed, labelled null: the footage no piece covers,
  split by the clips' cuts (deleted footage, minus what any piece shows), that
  contains a restored anchor; each placed before the first piece (reel order) that
  starts where it ends or later;
- captions move from overlays.captions to captions; view, lead_in/lead_out, cuts,
  restored_anchors and the shuttle trail flag are dropped.

v5 stores times to 0.1 s and needs segments of at least 0.3 s: times are rounded to
0.1 s (at most 0.05 s off), and a segment that comes out shorter is merged into a
touching neighbour of the same speed (the one before it, else after), or dropped.
Segment ids: "s1", "s2", ... in playback order.

Downgrade (where possible): segments of a rally that still exists become its clip's
pieces (consecutive ones in increasing time join one clip; a replay starts another),
with lead_in/lead_out set to fit them (1-3 s; anything further out is cut off);
unlabelled segments become restored anchors at their middle (they play again only
if they're still a hidden stretch between pieces). No cuts.

Data only. The conversions are written out here, not imported from the app: a
migration must keep doing exactly this after the app's code changes.

Revision ID: 8f3a5c2d9e14
Revises: ec4778601b52
Create Date: 2026-11-21 10:00:00.000000

"""
import math
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '8f3a5c2d9e14'
down_revision = 'ec4778601b52'
branch_labels = None
depends_on = None

MIN_SPAN = 0.05  # v4's: shorter stretches counted as nothing
MIN_SEGMENT = 0.3  # v5's shortest segment
MIN_LEAD, MAX_LEAD = 1.0, 3.0  # v4's default and largest lead_in/lead_out

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

Span = tuple[float, float]
Rallies = dict[int, Span]


def _union(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(span: Span, holes: list[Span]) -> list[Span]:
    parts = [span]
    for h_start, h_end in holes:
        parts = [
            part
            for start, end in parts
            for part in ((start, min(end, h_start)), (max(start, h_end), end))
            if part[1] - part[0] > 0
        ]
    return parts


def _gaps(covered: list[Span], limit: float) -> list[Span]:
    gaps, cursor = [], 0.0
    for start, end in covered:
        if start - cursor >= MIN_SPAN:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if limit - cursor >= MIN_SPAN:
        gaps.append((cursor, limit))
    return gaps


def v4_playback(
    edit: dict[str, Any], duration: float | None
) -> list[tuple[float, float, bool, int | None]]:
    """What a v4 reel's Rallies-only playback played: (start, end, slow_mo,
    rally_id or None for a restored stretch), in order. The v4 rule, step for step
    (app/reel_ops.py and app/export_snapshot.py as of edit list v4)."""
    limit = float("inf") if duration is None else duration
    pieces = []
    for clip in edit.get("clips", []):
        for p in clip.get("pieces", []):
            start, end = max(0.0, p["start"]), min(p["end"], limit)
            if end - start >= MIN_SPAN:
                pieces.append((start, end, bool(p.get("slow_mo")), clip["rally_id"]))
    covered = _union([(s, e) for s, e, _, _ in pieces])
    cuts = [
        part
        for clip in edit.get("clips", [])
        for c in clip.get("cuts", [])
        for part in _subtract((max(0.0, c["start"]), min(c["end"], limit)), covered)
        if part[1] - part[0] >= MIN_SPAN
    ]
    anchors = edit.get("restored_anchors", [])
    gaps = [
        (s, e, False, None)
        for s, e in _gaps(_union(covered + cuts), limit)
        if e != float("inf") and any(s < a < e for a in anchors)
    ]
    played: list[tuple[float, float, bool, int | None]] = []
    for piece in pieces:
        while gaps and gaps[0][1] <= piece[0]:
            played.append(gaps.pop(0))
        played.append(piece)
    return played + gaps


def _to_v5_segments(
    played: list[tuple[float, float, bool, int | None]], rallies: Rallies
) -> list[dict[str, Any]]:
    rounded: list[list[Any]] = [
        [round(s, 1), round(e, 1), slow, r if r in rallies else None]
        for s, e, slow, r in played
    ]
    kept: list[list[Any]] = []
    for i, seg in enumerate(rounded):
        if seg[1] - seg[0] >= MIN_SEGMENT - 1e-9:
            kept.append(seg)
            continue
        before = kept[-1] if kept else None
        after = rounded[i + 1] if i + 1 < len(rounded) else None
        if before and before[1] == seg[0] and before[2] == seg[2]:
            before[1] = seg[1]  # no footage lost: it joins the one before
        elif after and after[0] == seg[1] and after[2] == seg[2]:
            after[0] = seg[0]  # ...or the one after
        # else: too short to keep on its own (v4 allowed 0.05 s)
    return [
        {"id": f"s{n}", "start": s, "end": e, "slow_mo": slow, "rally_id": r}
        for n, (s, e, slow, r) in enumerate(kept, 1)
    ]


def edit_v4_to_v5(
    edit: dict[str, Any], rallies: Rallies, duration: float | None
) -> dict[str, Any]:
    if edit.get("version") != 4:
        return edit
    return {
        "version": 5,
        "format": edit.get("format", "16:9"),
        "segments": _to_v5_segments(v4_playback(edit, duration), rallies),
        "captions": edit.get("overlays", {}).get("captions", []),
    }


def _lead(needed: float) -> float:
    return min(MAX_LEAD, max(MIN_LEAD, math.ceil(round(needed * 100, 6)) / 100))


def edit_v5_to_v4(edit: dict[str, Any], rallies: Rallies) -> dict[str, Any]:
    if edit.get("version") != 5:
        return edit
    segments = edit.get("segments", [])
    labelled = [s for s in segments if s.get("rally_id") in rallies]
    lead_in = _lead(max((rallies[s["rally_id"]][0] - s["start"] for s in labelled), default=0.0))
    lead_out = _lead(max((s["end"] - rallies[s["rally_id"]][1] for s in labelled), default=0.0))
    clips: list[dict[str, Any]] = []
    anchors = []
    for s in segments:
        bounds = rallies.get(s.get("rally_id"))  # type: ignore[arg-type]
        if bounds is None:  # footage outside the rallies: a restored stretch
            anchors.append(round((s["start"] + s["end"]) / 2, 3))
            continue
        start = max(s["start"], bounds[0] - lead_in, 0.0)
        end = min(s["end"], bounds[1] + lead_out)
        if end - start <= 1e-6:
            continue
        piece = {"start": start, "end": end, "slow_mo": bool(s.get("slow_mo"))}
        last = clips[-1] if clips else None
        if last and last["rally_id"] == s["rally_id"] and start >= last["pieces"][-1]["end"] - 1e-9:
            last["pieces"].append(piece)
        else:
            clips.append({"rally_id": s["rally_id"], "pieces": [piece], "cuts": []})
    return {
        "version": 4,
        "view": "cut",
        "format": edit.get("format", "16:9"),
        "lead_in": lead_in,
        "lead_out": lead_out,
        "clips": clips,
        "restored_anchors": anchors,
        "overlays": {"trail": False, "captions": edit.get("captions", [])},
    }


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
    _convert_reels(edit_v4_to_v5)


def downgrade():
    _convert_reels(lambda edit, rallies, _duration: edit_v5_to_v4(edit, rallies))
