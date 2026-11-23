# Edit list v4 → v5 migration cases

Reels as edit list v4 described them (rally clips cut into pieces, lead-in/out,
deleted cuts, restored-break anchors) with what their "Rallies only" playback
played. Until v5 these were shared with the frontend as the contract between its
preview and the export; now they check migration 8f3a5c2d9e14
(`backend/tests/test_migrations.py`): each v4 reel must become v5 segments that play
exactly `expected.segments` (a `gap` becomes a segment with `rally_id: null`), and
its export snapshot must have the same `expected.captions` and `expected.duration`.

Each file:

```jsonc
{
  "description": "what the case is about",
  "duration": 60,                       // the video's, seconds
  "rallies": [{"id": 1, "start": 10, "end": 20}],   // the video's rallies
  "edit": { /* a ReelEdit v4, as stored */ },
  "expected": {
    "segments": [{"kind": "piece" | "gap", "start": 9, "end": 21, "slow_mo": false}],
    "captions": [{"text": "...", "start": 3, "end": 9}],  // OUTPUT seconds
    "duration": 24                      // output length; slow-mo counts double
  }
}
```

Frozen: they describe v4, so they don't change when the app does.
