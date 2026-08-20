#!/usr/bin/env python3
"""Solve reward budgets from a TARGET REWARD SHARE vector.

Why this exists
---------------
Three consecutive runs shipped the term they were built around at under 4% of
realized reward mass, while the budget field looked generous:

    p10touch  StrongTouch  RewardShare 0.037
    p11boost  SaveBoost    RewardShare 0.016
    p12goal   AirTouch     RewardShare 0.008

A budget converts a MAXIMAL payout into a weight. The policy responds to
REALIZED mass, which is `budget x mean realized value x rate`, and nothing in
the budget framework ever estimated the last two. So a term whose maximum is
rare (a ceiling-height air touch after 1.75 s aloft) is inert at a budget that
reads large, and nobody notices until the post-mortem.

This script closes that gap. Give it the shares you WANT and the shares a short
calibration probe MEASURED, and it returns the budgets that realize them.

The maths
---------
`RewardShare/X` in Train.cpp is `mean|r_X * w_X|` normalized across terms, so
with m_i the per-step mean of |r_i| under the probe policy:

    share_i = m_i w_i / sum_j m_j w_j

Rescaling w_i -> r_i w_i and holding the policy fixed (which is what a short
probe buys) leaves m_i unchanged, so with s_i the probe's measured shares:

    share_i^new = s_i r_i / sum_j s_j r_j = t_i

Let D = sum_j s_j r_j. Then r_i = t_i D / s_i, and D is fixed by whichever
weight is FROZEN. Freezing one is not optional: the system is otherwise scale
degenerate, since multiplying every weight by a constant leaves all shares
unchanged. p13strike anchors on Goal, whose weight is the one number carried
over from p12 untouched:

    r_anchor = 1  =>  D = s_anchor / t_anchor

The solve is exact in one pass, because m does not depend on w. It calibrates
the STARTING share, not the equilibrium: if a term works, the behaviour it pays
for gets more frequent and its share grows. That is intended, and it is why
every target here is paired with a share CEILING in the run's kill criteria --
a term that becomes the run's argmax is a farm.

THE PROBE POLICY MUST BE THE RUN'S STARTING POLICY
--------------------------------------------------
This is not a detail, and getting it wrong killed p13strike's second attempt at
48.7M steps.

EVENT-term share scales with event RATE, and the event rate is exactly what a
run is trying to change. So there is no single static share vector that is
correct at both ends of a run. Solving against a probe of a CONVERGED policy and
then starting FRESH gives budgets calibrated for the end state: under a random
policy the event terms have nothing to pay for and collapse, while the rate
terms absorb everything they vacated. Measured, on that attempt: TouchGoalAccel
landed at 0.05x its target and TouchEdge at 0.02x, while Air hit 19.5x and
SaveBoost 6.6x -- 60% of reward mass on terms collectable without ever touching
the ball, which is a do-nothing attractor and the bot duly did nothing.

So: calibrate on the policy the run will START from. That makes share targeting
the natural instrument for a RESUMED run. For a cold start, probe from a cold
init and understand that only the early shares are being controlled.

`--probe-checkpoint` and `--run-checkpoint` are asserted equal below for exactly
this reason.

Usage
-----
    scripts/solve_budgets.py --csv bot/build/metrics/main-p13cal.csv \\
                             --targets p13strike \\
                             --probe-checkpoint p12goal/250006016 \\
                             --run-checkpoint  p12goal/250006016
"""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Target reward-share vectors, by run label. Each must sum to 1.0 and name the
# frozen anchor term.
#
# p13strike: the proximity block (SpeedToBall + FaceBall + TouchEdge) goes
# 0.688 -> 0.330 and the ball block (TouchGoalAccel + Goal) goes 0.247 -> 0.550.
# That inversion is the whole run. Justification per term is in bot/src/Config.h.
TARGETS = {
    "p13strike": {
        "anchor": "Goal",
        "shares": {
            "TouchGoalAccel": 0.364,
            "SpeedToBall": 0.220,
            "Goal": 0.200,
            "TouchEdge": 0.065,
            "SaveBoost": 0.050,
            "FaceBall": 0.045,
            "AirTouch": 0.030,
            "PickupBoost": 0.020,
            # HELD at p12's measured share rather than given a target: this is
            # the float attractor's own term and the lever already found wrong
            # twice. It gets no promotion.
            "Air": 0.006,
        },
    },
    # p14aerial: AirTouch changes SHAPE (height goes convex), so only its
    # budget is re-solved. Everything else is HELD at p13's value.
    #
    # Deliberately not a full re-solve. Re-solving the whole vector every run
    # would claw back exactly the terms that WORKED -- p13's TouchGoalAccel
    # drifted 0.364 -> 0.449 because the behaviour it pays for got better, and
    # resetting that to target would undo the run's own result. That is the
    # treadmill this file already rejects for adaptive normalizers, and it
    # applies between runs too. Solve what changed shape; freeze the rest.
    "p14aerial": {
        "anchor": None,
        "hold": ["TouchGoalAccel", "SpeedToBall", "Goal", "TouchEdge",
                 "SaveBoost", "FaceBall", "PickupBoost", "Air"],
        "shares": {"AirTouch": 0.060},
    },
}

# Terms whose budget is a per-EPISODE integral (RateWeight divides by
# REFERENCE_EPISODE_STEPS); everything else is a per-event budget used directly.
# The solve works on WEIGHTS, so rate terms need the conversion undone to print
# a budget you can paste into Config.h.
RATE_TERMS = {"SpeedToBall", "FaceBall", "SaveBoost", "Air"}


def tail_mean(rows, key, n):
    vals = []
    for r in rows[-n:]:
        v = r.get(key)
        if v not in (None, "", "nan"):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        raise SystemExit(f"no usable values for {key!r} in the last {n} rows")
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="metrics CSV from the calibration probe")
    ap.add_argument("--targets", default="p13strike", choices=sorted(TARGETS))
    ap.add_argument("--tail", type=int, default=10,
                    help="iterations to average the probe over (default 10)")
    ap.add_argument("--reference-steps", type=float, default=390.0,
                    help="REFERENCE_EPISODE_STEPS in Config.h (default 390 = 26.0 s)")
    ap.add_argument("--show-ledger", action="store_true",
                    help="also print the resulting per-episode touch-unit ledger")
    ap.add_argument("--probe-checkpoint", required=True,
                    help="checkpoint the probe was RESUMED from ('fresh' for a cold init)")
    ap.add_argument("--run-checkpoint", required=True,
                    help="checkpoint the solved run will START from ('fresh' for a cold init)")
    args = ap.parse_args()

    # See "THE PROBE POLICY MUST BE THE RUN'S STARTING POLICY" above. A solve
    # across a policy boundary is not merely imprecise, it is calibrated for a
    # behaviour distribution the run does not begin in, and it fails silently:
    # every share is wrong from iteration one and the run looks healthy on every
    # PPO metric while optimizing a do-nothing attractor.
    if args.probe_checkpoint != args.run_checkpoint:
        raise SystemExit(
            f"probe policy ({args.probe_checkpoint}) is not the run's starting "
            f"policy ({args.run_checkpoint}).\n"
            f"Re-probe from {args.run_checkpoint} before solving. This killed "
            f"p13strike attempt 2 at 48.7M steps; see runs/RUNLOG.md.")

    spec = TARGETS[args.targets]
    targets, anchor = spec["shares"], spec["anchor"]
    held = spec.get("hold", [])

    # Two modes. FULL: every term has a target, the shares must sum to 1, and
    # one frozen anchor breaks the scale degeneracy. PARTIAL: only the terms
    # whose SHAPE changed are solved and the rest are held, which is what you
    # want between runs -- a full re-solve claws back the terms that worked.
    if held:
        if anchor is not None:
            raise SystemExit("a partial solve holds weights; it must not also name an anchor")
        overlap = set(held) & set(targets)
        if overlap:
            raise SystemExit(f"terms cannot be both held and targeted: {sorted(overlap)}")
    else:
        total = sum(targets.values())
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(f"target shares sum to {total:.6f}, not 1.0")

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit(f"{args.csv} is empty")

    measured = {}
    for term in list(targets) + held:
        col = f"RewardShare/{term}"
        if col not in rows[0]:
            raise SystemExit(
                f"{col} is not in {args.csv}. The probe must run the SAME reward "
                f"stack you are solving for, or the shares mean nothing.")
        measured[term] = tail_mean(rows, col, args.tail)

    for term, s in measured.items():
        if s <= 0:
            raise SystemExit(
                f"RewardShare/{term} measured {s} -- a term with zero realized "
                f"mass cannot be rescaled into a target share. Fix the term, "
                f"not the budget.")

    if held:
        # Held terms keep r = 1. For a single free term A:
        #   share_A = s_A r_A / (sum_held s_j + s_A r_A) = t_A
        #   =>  r_A = t_A * sum_held s_j / (s_A (1 - t_A))
        # With several free terms this holds per-term only when their combined
        # target is what is being aimed at, so solve them as one block.
        heldSum = sum(measured[t] for t in held)
        freeTotal = sum(targets.values())
        if freeTotal >= 1.0:
            raise SystemExit("free-term targets must sum to less than 1.0 in a partial solve")
        blockScale = freeTotal * heldSum / (1.0 - freeTotal)
        freeMeasured = sum(measured[t] for t in targets)
        ratios = {t: (targets[t] / freeTotal) * blockScale / measured[t] for t in targets}
        ratios.update({t: 1.0 for t in held})
        d = float("nan")
        print(f"partial solve: {len(targets)} free, {len(held)} held "
              f"(held share {heldSum:.4f}, free {freeMeasured:.4f} -> {freeTotal:.4f})")
    else:
        d = measured[anchor] / targets[anchor]
        ratios = {t: targets[t] * d / measured[t] for t in targets}

    steps = float(rows[-1].get("Total Timesteps", 0) or 0)
    print(f"probe:   {args.csv}  ({steps/1e6:.2f}M steps, last {args.tail} iterations)")
    if anchor:
        print(f"anchor:  {anchor} (weight frozen; D = s/t = {d:.4f})")
    print()
    print(f"{'term':16s} {'measured':>9s} {'target':>8s} {'x weight':>9s}  budget = old x ratio")
    print("-" * 74)
    order = sorted(targets, key=lambda t: -targets[t]) + sorted(held)
    for term in order:
        r = ratios[term]
        tgt = targets.get(term)
        note = "  (frozen anchor)" if term == anchor else ("  (held)" if term in held else "")
        scale = f" [rate term: budget = weight x {args.reference_steps:.0f}]" \
            if term in RATE_TERMS else ""
        tgtStr = f"{tgt:8.3f}" if tgt is not None else f"{'-':>8s}"
        print(f"{term:16s} {measured[term]:9.4f} {tgtStr} {r:9.4f}{note}{scale}")

    print()
    print("Multiply each budget in bot/src/Config.h by the 'x weight' column.")
    print("Rate terms: the ratio applies to the WEIGHT, and RateWeight() already")
    print(f"divides by {args.reference_steps:.0f}, so the budget field scales by the")
    print("same ratio -- no extra conversion needed as long as")
    print("REFERENCE_EPISODE_SECONDS did not change since the probe.")

    if args.show_ledger:
        try:
            asr = tail_mean(rows, "Average Step Reward", args.tail)
            eps = tail_mean(rows, "Episode/Mean Steps", args.tail)
        except SystemExit:
            return 0
        print()
        print(f"projected per-episode ledger at {eps:.0f} steps "
              f"(probe mean step reward {asr:.4f}):")
        for term in sorted(targets, key=lambda t: -targets[t]):
            print(f"  {term:16s} {targets[term] * asr * eps:7.2f} touch-units "
                  f"({targets[term]*100:4.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
