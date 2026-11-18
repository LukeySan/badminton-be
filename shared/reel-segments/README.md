# Reel → export segments: shared fixtures

What an exported reel plays, given its edit list. Both repos test against these
files, so the app's preview and the exported MP4 can't silently drift apart:

- backend: `backend/tests/test_export_snapshot.py` runs each file through
  `app/export_snapshot.py` (the snapshot the worker renders);
- frontend: its timeline test builds the "Rallies only" timeline
  (`src/components/Reels/timeline.ts`) from the same edit and compares the stretches
  it plays. The frontend keeps a copy (`bun run sync-fixtures` refreshes it).

Each file:

```jsonc
{
  "description": "what the case is about",
  "duration": 60,                       // the video's, seconds
  "rallies": [{"id": 1, "start": 10, "end": 20}],   // the video's current rallies
  "edit": { /* a ReelEdit (v4), as stored */ },
  "expected": {
    "segments": [{"kind": "piece" | "gap", "start": 9, "end": 21, "slow_mo": false}],
    "captions": [{"text": "...", "start": 3, "end": 9}],  // OUTPUT seconds (backend only)
    "duration": 24                      // output length; slow-mo counts double
  }
}
```

Rules (see `backend/app/export_snapshot.py` and `backend/app/reel_ops.py`): always the
Rallies-only result, even in Full video view; pieces in reel order as stored (lead
already included). A clip's `cuts` (footage the user deleted) never play. The other
footage no piece covers is split into stretches by the pieces and cuts around it (and
the start / end of the video); a stretch plays, at normal speed, if one of
`restored_anchors` is strictly inside it, placed before the first piece (in reel
order) that starts where it ends or later. So a restored break never crosses or
includes a cut. Anchors inside a piece or a cut do nothing. Captions show where the
source time is inside them; the first one wins on overlap.

Add a file here for every new rule or bug, then copy it to the frontend.
