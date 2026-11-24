"""reel edit v6: overlays on the reel's timeline (captions become text overlays)

reel.edit v5 -> v6: segments stay as they are; `captions` (source time, shown in
every segment showing that time, the first in the list winning where two overlap)
become `overlays` of type "text" on the REEL's timeline, exactly where v5's export
showed them:

- each caption maps into every segment that shows part of it (a replay shows it
  again; slow-mo counts double); in each segment, time already taken by an earlier
  caption is left out; the same text continuing across touching segments is one
  overlay;
- times rounded to 0.1 s; pieces shorter than 0.3 s (v6's minimum) are dropped;
- in time order, at most 20 (v6's limit; the first 20); ids "o1", "o2", ...;
- centred near the bottom (x 0.5, y 0.85) at the default size (0.06).

export.snapshot v1 -> v2: `captions` (already on the output timeline) become
`overlays` the same way (text, start, end; x 0.5, y 0.85, size 0.06), so exports
queued or retried after the upgrade render with the new worker.

Downgrade: each text overlay goes back to a caption per segment it covers (its
source time there), in overlay order; positions and sizes are lost. Snapshots:
overlays back to captions (in time order).

Data only. The conversions are written out here, not imported from the app: a
migration must keep doing exactly this after the app's code changes.

Revision ID: 4c1e7a9b2d60
Revises: 8f3a5c2d9e14
Create Date: 2026-11-23 10:00:00.000000

"""
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4c1e7a9b2d60'
down_revision = '8f3a5c2d9e14'
branch_labels = None
depends_on = None

SLOW_MO = 2.0
MIN_OVERLAY = 0.3  # v6's shortest overlay
MAX_OVERLAYS = 20  # v6's limit
MAX_CAPTIONS = 500  # v5's limit
UNDER_A_FRAME = 0.01  # v5's export skipped caption pieces shorter than this
DEFAULT_PLACE = {"x": 0.5, "y": 0.85, "size": 0.06}

reel = sa.table(
    "reel",
    sa.column("id", sa.Uuid()),
    sa.column("edit", postgresql.JSONB()),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
export = sa.table(
    "export",
    sa.column("id", sa.Uuid()),
    sa.column("snapshot", postgresql.JSONB()),
)

Span = tuple[float, float]


def _subtract(span: Span, holes: list[Span]) -> list[Span]:
    parts = [span]
    for h_start, h_end in holes:
        next_parts = []
        for start, end in parts:
            if h_end <= start or end <= h_start:
                next_parts.append((start, end))
                continue
            if start < h_start:
                next_parts.append((start, h_start))
            if h_end < end:
                next_parts.append((h_end, end))
        parts = next_parts
    return parts


def _timeline(segments: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """(source start, source end, reel start, speed factor) per segment."""
    out = []
    position = 0.0
    for s in segments:
        factor = SLOW_MO if s.get("slow_mo") else 1.0
        out.append((float(s["start"]), float(s["end"]), position, factor))
        position += (float(s["end"]) - float(s["start"])) * factor
    return out


def captions_on_reel_timeline(edit: dict[str, Any]) -> list[dict[str, Any]]:
    """v5's rule (its export's map_captions), on the reel's timeline:
    [{text, start, end}] in time order, at full precision."""
    out: list[dict[str, Any]] = []
    for seg_start, seg_end, position, factor in _timeline(edit.get("segments", [])):
        taken: list[Span] = []
        for caption in edit.get("captions", []):
            span = (max(caption["start"], seg_start), min(caption["end"], seg_end))
            if span[1] <= span[0]:
                continue
            for start, end in _subtract(span, taken):
                out_start = round(position + (start - seg_start) * factor, 3)
                out_end = round(position + (end - seg_start) * factor, 3)
                if out_end - out_start < UNDER_A_FRAME:
                    continue
                last = out[-1] if out else None
                if last and last["text"] == caption["text"] and last["end"] == out_start:
                    last["end"] = out_end
                else:
                    out.append(
                        {"text": caption["text"], "start": out_start, "end": out_end}
                    )
            taken.append(span)
    return sorted(out, key=lambda c: c["start"])


def to_text_overlays(captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    for caption in captions:
        start, end = round(caption["start"], 1), round(caption["end"], 1)
        if end - start < MIN_OVERLAY - 1e-9:
            continue
        overlays.append(
            {"type": "text", "text": caption["text"], "start": start, "end": end,
             **DEFAULT_PLACE}
        )
    overlays = overlays[:MAX_OVERLAYS]
    return [{"id": f"o{i}", **o} for i, o in enumerate(overlays, start=1)]


def edit_v5_to_v6(edit: dict[str, Any]) -> dict[str, Any]:
    if edit.get("version") != 5:
        return edit
    return {
        "version": 6,
        "format": edit.get("format", "16:9"),
        "segments": edit.get("segments", []),
        "overlays": to_text_overlays(captions_on_reel_timeline(edit)),
    }


def edit_v6_to_v5(edit: dict[str, Any]) -> dict[str, Any]:
    if edit.get("version") != 6:
        return edit
    timeline = _timeline(edit.get("segments", []))
    captions = []
    for overlay in edit.get("overlays", []):
        if overlay.get("type") != "text":
            continue
        for seg_start, seg_end, position, factor in timeline:
            reel_end = position + (seg_end - seg_start) * factor
            start, end = max(overlay["start"], position), min(overlay["end"], reel_end)
            # Float noise in the summed positions (0.7 + 0.6 = 1.2999999999999998)
            # mustn't leave a zero-length caption in the next segment
            if end - start < UNDER_A_FRAME:
                continue
            captions.append({
                "text": overlay["text"],
                "start": round(seg_start + (start - position) / factor, 3),
                "end": round(seg_start + (end - position) / factor, 3),
            })  # fmt: skip
    return {
        "version": 5,
        "format": edit.get("format", "16:9"),
        "segments": edit.get("segments", []),
        "captions": captions[:MAX_CAPTIONS],
    }


def snapshot_v1_to_v2(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot.get("version", 1) != 1:
        return snapshot
    new = {k: v for k, v in snapshot.items() if k != "captions"}
    new["version"] = 2
    new["overlays"] = [
        {"type": "text", "text": c["text"], "start": c["start"], "end": c["end"],
         **DEFAULT_PLACE}
        for c in snapshot.get("captions", [])
    ]  # fmt: skip
    return new


def snapshot_v2_to_v1(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot.get("version") != 2:
        return snapshot
    new = {k: v for k, v in snapshot.items() if k != "overlays"}
    new["version"] = 1
    texts = [o for o in snapshot.get("overlays", []) if o.get("type") == "text"]
    new["captions"] = sorted(
        ({"text": o["text"], "start": o["start"], "end": o["end"]} for o in texts),
        key=lambda c: c["start"],
    )
    return new


def _convert(convert_edit: Any, convert_snapshot: Any) -> None:
    bind = op.get_bind()
    for reel_id, edit in bind.execute(sa.select(reel.c.id, reel.c.edit)).all():
        new = convert_edit(edit)
        if new != edit:
            # A new updated_at: an editor holding the old version reloads (409)
            bind.execute(
                sa.update(reel)
                .where(reel.c.id == reel_id)
                .values(edit=new, updated_at=sa.func.now())
            )
    # export.updated_at is the queue time / heartbeat: left alone
    for export_id, snapshot in bind.execute(
        sa.select(export.c.id, export.c.snapshot)
    ).all():
        new = convert_snapshot(snapshot)
        if new != snapshot:
            bind.execute(
                sa.update(export).where(export.c.id == export_id).values(snapshot=new)
            )


def upgrade():
    _convert(edit_v5_to_v6, snapshot_v1_to_v2)


def downgrade():
    _convert(edit_v6_to_v5, snapshot_v2_to_v1)
