"""
Rally-detector regression check: run detect_rallies on the labeled matches' saved
tracking output and score it against the hand labels.

    uv run --package worker python worker/benchmark.py                 # v1 and v1.1
    uv run --package worker python worker/benchmark.py --detector v1.1
    uv run --package worker python worker/benchmark.py --save-baseline # after an intended change

Inputs, in worker/benchmark/ (committed, ~1 MB, no videos or GPU needed):
  matches.json        per match: fps, analysis-copy size, court corners (fractions)
  <match>/ball.csv    the worker's own tracking of that match (TrackNet fast mode on the
                      analysis copy), made with modal_worker_tracking.py in research
  <match>/labels.csv  hand labels, copied from ../badminton-test/TrackNetV3/results/
                      (or read them from there with --labels-dir after relabeling)

Not an accuracy estimate: v1.1's models were trained on these same matches, so its
numbers here are optimistic. It answers "did my change make things better or
worse?". Honest (leave-one-match-out) numbers: research EXPERIMENTS.md.

Scoring is score_rallies.py's: a detected rally matches the labeled rally it overlaps
most (one-to-one); missed = labels without a match; fake = shown rallies without one
("maybe" rallies count as shown); accurate = start and end both within 1 s (and 2 s).
"""

import argparse
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import DETECTORS, detect_rallies  # noqa: E402
from pipeline import write_court_json  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
BASELINE = os.path.join(BENCH_DIR, "baseline.json")
KEYS = ("labeled", "shown", "missed", "fake", "confident_fake", "maybe", "maybe_real",
        "accurate_1s", "accurate_2s")


def match(detected, truth):
    """score_rallies.match: greedy one-to-one matching by largest overlap."""
    pairs = []
    for i, d in enumerate(detected):
        for j, t in enumerate(truth):
            ov = max(0.0, min(d[1], t[1]) - max(d[0], t[0]))
            if ov > 0:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    used_d, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_t:
            used_d.add(i); used_t.add(j); out.append((i, j))
    return out


def read_labels(path):
    with open(path) as fh:
        return sorted((float(r["start"]), float(r["end"])) for r in csv.DictReader(fh) if r.get("end"))


def score(rallies, truth):
    shown = [(r["start"], r["end"]) for r in rallies]
    pairs = match(shown, truth)
    matched = {i for i, _ in pairs}
    within = lambda tol: sum(1 for i, j in pairs if abs(shown[i][0] - truth[j][0]) <= tol
                             and abs(shown[i][1] - truth[j][1]) <= tol)
    maybe = [i for i, r in enumerate(rallies) if r["is_maybe"]]
    return {
        "labeled": len(truth), "shown": len(shown), "missed": len(truth) - len(pairs),
        "fake": len(shown) - len(pairs),
        "confident_fake": sum(1 for i, r in enumerate(rallies) if not r["is_maybe"] and i not in matched),
        "maybe": len(maybe), "maybe_real": sum(1 for i in maybe if i in matched),
        "accurate_1s": within(1.0), "accurate_2s": within(2.0),
    }


def run(detector, matches, labels_dir):
    per_match = {}
    with tempfile.TemporaryDirectory() as tmp:
        for m in matches:
            d = os.path.join(BENCH_DIR, m["name"])
            court = write_court_json(m["corners"], m["width"], m["height"],
                                     os.path.join(tmp, f"{m['name']}_court.json"))
            result = detect_rallies(os.path.join(d, "ball.csv"), None, court, detector=detector, fps=m["fps"])
            labels = os.path.join(labels_dir, m["name"], "labels.csv") if labels_dir else os.path.join(d, "labels.csv")
            per_match[m["name"]] = score(result["rallies"], read_labels(labels))
    total = {k: sum(s[k] for s in per_match.values()) for k in KEYS}
    return {"version": DETECTORS[detector], "matches": per_match, "total": total}


def show(result, baseline, saving=False):
    t = result["total"]
    base = (baseline or {}).get(result["version"], {}).get("total")
    delta = lambda k: f" ({t[k] - base[k]:+d})" if base and t[k] != base[k] else ""
    print(f"\n{result['version']}: {t['labeled']} labeled rallies, {t['shown']} shown")
    print(f"  missed {t['missed']}{delta('missed')}   fake {t['fake']}{delta('fake')} "
          f"(confident {t['confident_fake']}{delta('confident_fake')})   "
          f"maybes {t['maybe']}{delta('maybe')} ({t['maybe_real']} real)   "
          f"accurate {t['accurate_1s']}{delta('accurate_1s')} within 1 s, "
          f"{t['accurate_2s']}{delta('accurate_2s')} within 2 s")
    for name, s in result["matches"].items():
        print(f"    {name:<8} missed {s['missed']:>3}  fake {s['fake']:>3}  maybes {s['maybe']:>3}  "
              f"accurate {s['accurate_1s']:>3}/{s['labeled']:<3} (2 s: {s['accurate_2s']})")
    if base is None and not saving:
        print("  (no baseline for this version yet: --save-baseline)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", action="append", choices=sorted(DETECTORS),
                   help="repeatable; default: all")
    p.add_argument("--labels-dir", default=None,
                   help="read <dir>/<match>/labels.csv instead of the copies in worker/benchmark/")
    p.add_argument("--save-baseline", action="store_true",
                   help="store these results as the numbers to compare against")
    args = p.parse_args()

    with open(os.path.join(BENCH_DIR, "matches.json")) as fh:
        matches = json.load(fh)
    baseline = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    results = [run(d, matches, args.labels_dir) for d in (args.detector or sorted(DETECTORS))]
    for r in results:
        show(r, baseline, saving=args.save_baseline)
    if args.save_baseline:
        baseline.update({r["version"]: r for r in results})
        with open(BASELINE, "w") as fh:
            json.dump(baseline, fh, indent=2)
        print(f"\nSaved {BASELINE}")


if __name__ == "__main__":
    main()
