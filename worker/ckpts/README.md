# Model weights (not committed)

Put `TrackNet_best.pt` here. It comes from the TrackNetV3 release
(https://github.com/qaz812345/TrackNetV3, MIT license — see the README there for the
download link), or copy it from the research folder:

```bash
cp ../badminton-test/TrackNetV3/ckpts/TrackNet_best.pt worker/ckpts/
```

InpaintNet is not used (it is broken in the fast `nonoverlap` mode we run).
