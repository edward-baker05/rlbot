#!/usr/bin/env python3
"""Play a policy against frozen ancestors and report an ANCHORED score.

    scripts/ladder.py --challenger <ckpt> [--rung label=<ckpt> ...] [--games N] [--seconds S]

`Rating/1v1` cannot answer "did this run make the bot better". It is an Elo over
a rolling 32-version pool whose new entrants inherit the current rating, so it is
an unanchored random walk: measured on p18entropy, 41 samples at per-sample sigma
6.77 give a random-walk sigma of 42.8 against an observed drift of 29.8. A
saturated policy and an improving one produce the same trace.

This does the opposite. The opponents are FROZEN checkpoints that cannot move to
meet the challenger, so goal share has a real null of 0.5 and two runs measured a
year apart are comparable.

Every pairing is played twice with the sides swapped. Blue and orange are not
symmetric here -- `ResetToRandomKickoff` and the goal geometry are mirrored but
the policy is not -- so a one-sided result confounds side bias with skill.
"""

import argparse
import concurrent.futures
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "bot" / "build"
SUMMARY = re.compile(
    r"blue (\d+) wins, (\d+) wins.*?goals blue (\d+) - (\d+) orange"
    r"|blue (\d+) wins, orange (\d+) wins, (\d+) draws \| goals blue (\d+) - (\d+) orange")

DEFAULT_RUNGS = [
    ("p12goal",   "checkpoints/main-p12goal/250006016"),
    ("p13strike", "checkpoints/main-p13strike/350001792"),
    ("p14aerial", "checkpoints/main-p14aerial/424733056"),
    ("p15manual", "checkpoints/main-p15manual/1044744064"),
    ("p16",       "checkpoints/main-p16/2281639168"),
    ("p17",       "checkpoints/main-p17/2579043968"),
]


def run_eval(blue, orange, games, seconds, seed, cpu, random_spawn=False):
    """One eval subprocess. Returns (blueWins, orangeWins, draws, blueGoals, orangeGoals)."""
    cmd = [str(BUILD / "HivemindBot"), "eval",
           "--blue", str(blue), "--orange", str(orange),
           "--games", str(games), "--seconds", str(seconds), "--seed", str(seed)]
    if cpu:
        cmd.append("--cpu")
    if random_spawn:
        cmd.append("--random-spawn")
    p = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("Eval summary:"):
            n = [int(x) for x in re.findall(r"\d+", line)]
            # blueWins, orangeWins, draws, blueGoals, orangeGoals
            return n[0], n[1], n[2], n[3], n[4]
    sys.exit(f"eval failed for {blue} vs {orange}:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")


def wilson(k, n, z=1.96):
    """Wilson interval: the normal approximation is wrong near 0 and 1, and rungs get lopsided."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--rung", action="append", default=[], metavar="label=path")
    ap.add_argument("--games", type=int, default=25, help="games per side, so 2x this per rung")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--random-spawn", action="store_true", help="use the TRAINING state distribution instead of kickoffs")
    ap.add_argument("--workers", type=int, default=6, help="concurrent eval processes")
    ap.add_argument("--chunks", type=int, default=6, help="chunks per side, for load balance")
    args = ap.parse_args()

    rungs = [tuple(r.split("=", 1)) for r in args.rung] if args.rung else DEFAULT_RUNGS
    chall = args.challenger if Path(args.challenger).is_absolute() else str(REPO / args.challenger)

    print(f"challenger: {chall}")
    sim_h = len(rungs) * 2 * args.games * args.seconds / 3600
    print(f"{args.games} games per side x 2 sides x {args.seconds:.0f}s, seed {args.seed}")
    print(f"{sim_h:.1f} h of simulation, ~{args.workers} at a time\n")
    print(f"{'rung':12}{'GF':>6}{'GA':>6}{'goal share':>12}{'95% CI':>16}{'W-D-L':>12}")
    print("-" * 64)

    # A goal arrives about once per 600 s of simulation, so a rung needs tens of
    # thousands of sim-seconds to be worth reading. One arena runs ~60x realtime,
    # so the work is split into chunks and run concurrently; each chunk gets its
    # own kickoff seed so the chunks are not replays of each other.
    jobs = []
    for label, path in rungs:
        p = path if Path(path).is_absolute() else str(REPO / path)
        per_chunk = max(1, args.games // args.chunks)
        for c in range(args.chunks):
            seed = args.seed + 1000 * c
            jobs.append((label, "home", chall, p, per_chunk, seed))
            jobs.append((label, "away", p, chall, per_chunk, seed + 500))

    tally = {label: [0, 0, 0, 0, 0] for label, _ in rungs}  # gf, ga, w, d, l
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_eval, b, o, g, args.seconds, sd, args.cpu, args.random_spawn): (lab, side)
                for lab, side, b, o, g, sd in jobs}
        for fut in concurrent.futures.as_completed(futs):
            lab, side = futs[fut]
            bw, ow, dr, bg, og = fut.result()
            t = tally[lab]
            if side == "home":
                t[0] += bg; t[1] += og; t[2] += bw; t[3] += dr; t[4] += ow
            else:
                t[0] += og; t[1] += bg; t[2] += ow; t[3] += dr; t[4] += bw
            done += 1
            print(f"  [{done}/{len(jobs)}] {lab} {side}", end="\r", flush=True)

    print(" " * 40, end="\r")
    results = []
    for label, _ in rungs:
        gf, ga, w, d, l = tally[label]
        n = gf + ga
        share = gf / n if n else float("nan")
        lo, hi = wilson(gf, n)
        results.append((label, gf, ga, share, lo, hi))
        print(f"{label:12}{gf:6d}{ga:6d}{share:11.1%} "
              f"{f'[{lo:.1%}, {hi:.1%}]':>16}{f'{w}-{d}-{l}':>12}")

    print("-" * 64)
    print("null = 50.0% goal share; a CI excluding 50% is a real difference")
    beaten = [r for r in results if r[4] > 0.5]
    level = [r for r in results if r[4] <= 0.5 <= r[5]]
    print(f"\nbeats {len(beaten)}/{len(results)} rungs significantly; "
          f"level with {len(level)}: {', '.join(r[0] for r in level) if level else 'none'}")


if __name__ == "__main__":
    main()
