#!/usr/bin/env python3
"""Compare training runs from their metric CSVs.

    scripts/summarize_runs.py bot/build/metrics/general-baseline.csv \
                              bot/build/metrics/general-selfplay.csv

Prints, for each run, the metrics that actually tell you whether it worked --
averaged over the last quarter of the run so a single noisy iteration does not
decide the comparison.

    scripts/summarize_runs.py --trend bot/build/metrics/main-p1-validate.csv

`--trend` instead splits ONE run into equal buckets and averages each, which is
how you see a plateau. The tail-mean view above cannot: a run that learned for
40M steps and then stalled for 80M has a perfectly healthy last quarter. It also
prints PPO health first, because a frozen policy produces flat behavioural
metrics and entirely plausible reward shares, and the two look identical from
the reward side. See docs/metrics.md.

Only stdlib; no pandas.
"""

import csv
import sys
from pathlib import Path

# Metrics worth comparing, in the order they are printed.
# Reward is first because it is what everyone looks at, and last in usefulness.
INTERESTING = [
    ("Average Step Reward", "reward", "{:.4f}"),
    ("Policy Entropy", "entropy", "{:.4f}"),
    ("Player/Touch Height", "touch height", "{:.1f}"),
    ("Player/Ball Touch Ratio", "touch ratio", "{:.4f}"),
    ("Player/In Air Ratio", "air ratio", "{:.4f}"),
    ("Player/Speed", "speed", "{:.1f}"),
    ("Player/Boost", "boost", "{:.1f}"),
    ("Game/Goal Speed", "goal speed", "{:.1f}"),
    ("Overall Steps/Second", "steps/sec", "{:,.0f}"),
]

PHASE_PREFIX = "Phase/"
RATING_PREFIX = "Rating/"


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if parsed:
                rows.append(parsed)
    return rows


def tail_mean(rows, key, fraction=0.25):
    """Mean of `key` over the final `fraction` of rows that have it."""
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    n = max(1, int(len(vals) * fraction))
    tail = vals[-n:]
    return sum(tail) / len(tail)


def final(rows, key):
    for r in reversed(rows):
        if key in r:
            return r[key]
    return None


def summarize(path):
    rows = load(path)
    if not rows:
        print(f"{path}: no data")
        return None

    name = Path(path).stem
    print(f"\n{'=' * 62}")
    print(f"{name}   ({len(rows)} iterations)")
    print("=" * 62)

    steps = final(rows, "Total Timesteps")
    if steps:
        print(f"  {'total timesteps':<22} {steps:>14,.0f}")

    print(f"\n  {'metric':<22} {'mean (last 25%)':>14}")
    print(f"  {'-' * 22} {'-' * 14}")
    for key, label, fmt in INTERESTING:
        v = tail_mean(rows, key)
        if v is not None:
            print(f"  {label:<22} {fmt.format(v):>14}")

    phases = sorted({k for r in rows for k in r if k.startswith(PHASE_PREFIX)})
    if phases:
        print(f"\n  {'phase':<22} {'share':>14}")
        print(f"  {'-' * 22} {'-' * 14}")
        for k in phases:
            v = tail_mean(rows, k)
            if v is not None:
                print(f"  {k[len(PHASE_PREFIX):]:<22} {v * 100:>13.1f}%")

    ratings = sorted({k for r in rows for k in r if k.startswith(RATING_PREFIX)})
    if ratings:
        print(f"\n  {'skill rating':<22} {'final':>14}")
        print(f"  {'-' * 22} {'-' * 14}")
        for k in ratings:
            v = final(rows, k)
            if v is not None:
                print(f"  {k[len(RATING_PREFIX):]:<22} {v:>14.1f}")
    else:
        print("\n  (no skill ratings -- run with --track-skill, and note that")
        print("   ratings only appear once the version pool is non-empty)")

    return rows


def compare(paths, all_rows):
    """Side-by-side delta table, using the first run as the reference."""
    runs = [(Path(p).stem, r) for p, r in zip(paths, all_rows) if r]
    if len(runs) < 2:
        return

    base_name, base_rows = runs[0]
    print(f"\n{'=' * 62}")
    print(f"vs {base_name}")
    print("=" * 62)

    # Ratings are the point of a self-play comparison, so they go in the delta
    # table too -- as an absolute difference, since ELO-style ratings start at
    # zero and a percentage change against ~0 is meaningless.
    rating_keys = sorted(
        {k for _, r in runs for row in r for k in row if k.startswith(RATING_PREFIX)}
    )

    for other_name, other_rows in runs[1:]:
        print(f"\n  {other_name}")
        print(f"  {'metric':<22} {'base':>11} {'this':>11} {'delta':>11}")
        print(f"  {'-' * 22} {'-' * 11} {'-' * 11} {'-' * 11}")
        for key, label, _ in INTERESTING:
            a, b = tail_mean(base_rows, key), tail_mean(other_rows, key)
            if a is None or b is None:
                continue
            delta = ((b - a) / abs(a) * 100) if a else 0.0
            print(f"  {label:<22} {a:>11.3f} {b:>11.3f} {delta:>10.1f}%")

        for key in rating_keys:
            a, b = final(base_rows, key), final(other_rows, key)
            if a is None or b is None:
                continue
            label = "rating " + key[len(RATING_PREFIX):]
            print(f"  {label:<22} {a:>11.1f} {b:>11.1f} {b - a:>+11.1f}")


# PPO health goes first: these say whether the policy moved at all, and nothing
# below them means anything until they look right.
TREND_PPO = [
    ("Policy Entropy", "entropy", "{:.4f}"),
    ("Mean KL Divergence", "KL", "{:.5f}"),
    ("SB3 Clip Fraction", "clip frac", "{:.5f}"),
    ("Policy Update Magnitude", "update mag", "{:.4f}"),
    ("GAE/Avg Advantage", "advantage", "{:.4f}"),
    ("GAE/Returns STD", "returns std", "{:.2f}"),
]

TREND_BEHAV = [
    ("Player/Ball Touch Ratio", "touch ratio", "{:.5f}"),
    ("Player/In Air Ratio", "air ratio", "{:.4f}"),
    ("Player/Touch Height", "touch height", "{:.1f}"),
    ("Game/Goal Speed", "goal speed", "{:.0f}"),
    ("Average Step Reward", "reward", "{:.4f}"),
]


def trend(path, buckets=8):
    """Bucket-averaged view of one run, to expose plateaus."""
    rows = load(path)
    if not rows:
        print(f"{path}: no data")
        return

    n = min(buckets, len(rows))
    edges = [round(i * len(rows) / n) for i in range(n + 1)]
    groups = [rows[edges[i]:edges[i + 1]] for i in range(n) if edges[i + 1] > edges[i]]

    def mean(group, key):
        vals = [r[key] for r in group if key in r]
        return sum(vals) / len(vals) if vals else None

    print(f"\n{'=' * 62}")
    print(f"{Path(path).stem}   ({len(rows)} iterations, bucket means)")
    print("=" * 62)

    header = "".join(
        f"{(mean(g, 'Total Timesteps') or 0) / 1e6:>10.0f}M" for g in groups
    )
    print(f"\n  {'metric':<16}{header}")
    print(f"  {'-' * (16 + 11 * len(groups))}")

    shares = sorted({k for r in rows for k in r if k.startswith("RewardShare/")})
    sections = [
        ("PPO health", TREND_PPO),
        ("behaviour", TREND_BEHAV),
        ("reward shares", [(k, k[len("RewardShare/"):], "{:.3f}") for k in shares]),
    ]
    for title, keys in sections:
        if not keys:
            continue
        print(f"  {title}")
        for key, label, fmt in keys:
            cells = []
            for g in groups:
                v = mean(g, key)
                cells.append((fmt.format(v) if v is not None else "-").rjust(11))
            if any(c.strip() != "-" for c in cells):
                print(f"  {label:<16}" + "".join(cells))

    # Ratings are sparse (one point per skillUpdateInterval iterations), so a
    # bucket mean would hide how few there are. List them.
    for key in sorted({k for r in rows for k in r if k.startswith(RATING_PREFIX)}):
        pts = [(r["Total Timesteps"] / 1e6, r[key]) for r in rows if key in r]
        print(f"\n  {key}: " + "  ".join(f"{s:.0f}M={v:.3f}" for s, v in pts))
    print()


def main():
    args = sys.argv[1:]
    if "--trend" in args:
        args = [a for a in args if a != "--trend"]
        if not args:
            print(__doc__)
            return 1
        for p in args:
            trend(p)
        return 0

    paths = args
    if not paths:
        print(__doc__)
        return 1

    all_rows = [summarize(p) for p in paths]
    compare(paths, all_rows)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
