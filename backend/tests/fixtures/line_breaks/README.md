# Shared fixture: Thai line breaks

`thai.json` is a copy of the frontend's `tests/fixtures/line-breaks/thai.json`. The
frontend owns it: its test `src/components/Reels/lineBreaks.fixtures.test.ts`
computes the cases with the JavaScript runtime's `Intl.Segmenter` (the ICU
dictionary breaks browsers use for Thai in CSS) and writes the file with
`UPDATE_LINE_BREAKS=1 bun test src/components/Reels/lineBreaks`.

Here, `tests/worker/test_line_breaks.py` checks the same cases against the export's
wrapping (`worker/caption_image.py`, PyICU), so an exported reel's text overlays
wrap where the app's preview does.

To add a case: add it to `CASES` in the frontend test, regenerate there, copy the
file here unchanged, and run both tests. Both sides use a font-free width (characters
that aren't combining marks), so the lines only depend on where breaks may go and on
the shared greedy rule; real line widths come from the fonts.
