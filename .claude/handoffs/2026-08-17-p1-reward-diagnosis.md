# Handoff: P1 has a reward problem, not a PPO problem (2026-08-17)

Follows `.claude/handoffs/2026-08-17-phase0-execution-and-ppo-fix.md`. That
session found and fixed a PPO bug that had frozen every prior run. This session
ran the validation that the fix was supposed to pass. **It failed**, and the
failure is informative.

Read this first, then `runs/RUNLOG.md` (newest entry is p1-validate), then
`docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md` for the binding
decisions D1–D8.

## Why we ran p1-validate, and what it was supposed to show

The previous session derived a PPO config by probe (entropyScale 0.002,
epochs 2, LR 3e-4) and recorded probe-f as a "BREAKTHROUGH... the chosen
config". That claim rested on 30M steps. The open question it left was the only
one that mattered: **does the fixed config keep learning, or does it stall?**

The run was `train --max-steps 150000000 --games 128 --label p1-validate`.
Success would have looked like: entropy continuing to fall, KL staying well
above zero, then `Player/Ball Touch Ratio` rising and `Player/In Air Ratio`
falling from ~0.90.

## What actually happened

Stopped by hand at **117M steps** (of 150M) once the answer was unambiguous.
Bucket-averaged over 8 windows — never read single iterations, they are noisy:

| metric | 7M | 36M | 65M | 93M | 108M |
|---|---|---|---|---|---|
| Policy Entropy | 0.719 | **0.650** | 0.654 | 0.669 | **0.700** |
| Mean KL | 1.25e-3 | 5.8e-4 | 5.3e-4 | 8.6e-4 | 7.0e-4 |
| SB3 Clip Fraction | 6.1e-3 | 1.0e-3 | 1.0e-3 | 5.7e-3 | 4.1e-3 |
| Ball Touch Ratio | 8.9e-4 | 9.9e-4 | 1.03e-3 | 1.07e-3 | 1.05e-3 |
| In Air Ratio | 0.913 | 0.924 | 0.925 | 0.924 | 0.911 |
| GAE/Avg Advantage | 0.151 | 0.107 | 0.096 | 0.091 | **0.088** |
| Avg Step Reward | 0.0142 | 0.0155 | 0.0152 | 0.0155 | 0.0153 |

`Rating/1v1`, which only logs every 100 iterations (10M steps):

```
10M=-2.54  20M=-2.54  30M=-2.54  40M=-2.54  50M=-5.00
60M=-5.00  70M=-5.00  80M=-7.50  90M=-7.50 100M=-7.50 110M=-7.50
```

Four findings, in order of how much they should drive the next decision:

1. **The rating falls monotonically.** Reward flat while rating drops is
   precisely the farming signature `docs/metrics.md` names. Newer checkpoints
   lose to older ones. This is the strongest single piece of evidence and it
   was not available on any 30M probe — the tracker needs ~10M steps per point.
2. **Learning stops at roughly 40M.** Touch ratio rises 0.00089 → 0.00104 by
   50M, then is dead flat for the next 65M steps. Entropy bottoms at 0.650
   around 36M and then drifts back *up* to 0.700 — the policy is diffusing, not
   converging.
3. **In Air Ratio rises rather than falls** (0.913 → 0.925), and
   `Phase/Recover` sits at **0.86** for the whole run. The bot spends 86% of
   its life tumbling and landing. This is the flip-spam, unchanged.
4. **Every reward share is static to three significant figures across 115M
   steps.** VelPlayerToBall 0.156→0.139, Touch 0.030→0.034, StrongTouch
   0.064→0.074, VelBallToGoal 0.445→0.446, Goal 0.128→0.133. Nothing is being
   traded off against anything.

## What this means

PPO health is **no longer the binding constraint**, but it is not clean either:
KL and clip fraction are alive (unlike the pre-fix runs where they were 0.0000)
and the policy is demonstrably moving early on. So the fix was real. It is just
not sufficient, and past ~40M there is nothing left driving improvement.

Per the previous session's own method rule — check PPO health first, then
rewards — PPO health has now been checked and passes the bar that mattered.
**The reward function is the live suspect.** This is the conditional the last
handoff explicitly set up, and it has now triggered.

### The leading hypothesis: VelBallToGoal is a variance pump

`VelBallToGoal` takes **44.5%** of all absolute reward mass, rock-steady across
115M steps, at a weight of only 0.5. It is:

- **continuous** — it pays on every one of ~15 decision steps per second;
- **zero-sum** (coef 0.3) — so in 1v1 it is near-antisymmetric between the two
  cars and its mean is ~0;
- **mostly not caused by the policy** — with a touch ratio of 0.001, the ball's
  velocity comes overwhelmingly from kickoff momentum and wall bounces, not
  from the bot.

A large, near-zero-mean, policy-independent term is the textbook definition of
variance injected into the return with no learnable content. The corroborating
number: `GAE/Returns STD` is **21.4** against an `Avg Return` of 7.8, and
`GAE/Avg Advantage` decays monotonically 0.151 → 0.088 over the run. The
learning signal is shrinking inside a large noise floor.

`runs/RUNLOG.md`'s p1probe-b entry already flagged this once ("VelBallToGoal
absorbed it — mostly passive ball motion = zero-sum noise") and it was never
followed up, because the PPO bug was found first and took over the session.

### The competing hypothesis worth keeping alive

The touch reward may simply be unreachable. With ~90% air time the bot may
almost never be in a *state* from which a touch is achievable, so no reward
weighting can help — the fix would be curricular (more BallContact, or a
grounding term) rather than a reweighting. `Phase/Recover = 0.86` supports
this. Distinguish it with `Scenario/BallContact/EndedInGoal`, which sits at
~0.09 and is also flat: even when spawned next to the ball, outcomes do not
improve.

## Suggested change path

Standing constraints from the spec that bound all of this: **D4** — no
dribble/possession/ball-proximity reward, ever; **D6** — no weight without
measurement behind it; **D7** — gates are trends, not thresholds.

The probe loop that works here: change exactly one thing → 30M steps (~7 min at
~66k steps/s, 128 games) → compare bucket-averaged telemetry against
`p1-validate` as the baseline. **One variable per probe.** Three probes were
wasted last session tuning several knobs against a frozen policy.

One caveat on horizon: p1-validate shows the interesting divergence happens
between 40M and 80M, and `Rating/1v1` needs 10M steps per data point. A 30M
probe can rule a change *out* but cannot confirm one works. Promising probes
need a 100M+ confirmation run before anything is declared fixed.

Ordered by expected information per minute:

1. **Cut `velBallToGoal` to 0 for one probe.** Not a reduction — zero. It is
   44.5% of reward mass and the leading suspect; halving it would leave the
   result ambiguous. Watch `GAE/Returns STD` and `GAE/Avg Advantage`: if the
   variance hypothesis is right, STD drops sharply and advantage stops decaying.
   That is a mechanism-level prediction, so it is a real test, and it is
   falsifiable in 7 minutes.
2. **If that frees the signal, re-derive the P1 weights from the new shares**
   (spec Phase 0 item 6), targeting outcome terms (Goal + StrongTouch + Touch)
   carrying the majority of reward mass rather than the current ~24%.
3. **If it does not, attack the air time directly.** The bot cannot strike a
   ball it never drives to. Options, in D4-compatible order: raise the
   `recover` curriculum weight (currently 8) so landing on wheels is practised;
   or add an explicit grounded/wheels-down term. Note `velPlayerToBall` and
   `faceBall` are *already* gated on `GroundedReward`, so the existing shaping
   pays nothing while airborne — which means the bot is currently earning most
   of its money from terms it cannot influence. That is worth stating plainly
   in whatever gets tried.
4. **Only then reconsider entropy.** Entropy drifting back up 0.650 → 0.700
   after 36M suggests exploration pressure eventually outweighs a weak
   gradient. Fixing the gradient first is the right order; if a healthy reward
   still drifts, decay `entropyScale` over training rather than lowering it
   flat — D6 wants the decay schedule measured, not guessed.

Do **not** reintroduce a policy split or an MoE (D1), and do not add a
possession or ball-proximity term to force the touch ratio up (D4) — that is
the flick-bot local optimum the whole design is built to avoid.

## Tooling added this session

`spectate` — watch a checkpoint play live in RocketSimVis with no learner:

```bash
scripts/vis.sh &
scripts/spectate.sh p1-validate          # follows the run's newest checkpoint
scripts/spectate.sh p1-validate --spawns kickoff --deterministic
```

Safe to run against training in flight: measured cost **0.45%** of throughput.
That depends on `RunSpectate` pinning `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1
before the first inference — without it libtorch spreads inference over all six
cores and costs **4.4%**. Full option table in `docs/architecture.md`.

The user is a GC1 player and this is the evaluation instrument they are
uniquely good at — get them watching checkpoints. They have stated they do not
design reward functions and expect that driven for them, with the reasoning
explained. Do not hand them a menu of candidate weights.

## State at end of session

- `p1-validate` stopped at 117M. Checkpoints in
  `bot/build/checkpoints/main-p1-validate/` (newest 116075520), metrics CSV at
  `bot/build/metrics/main-p1-validate.csv`. This is the **baseline every P1
  probe should be compared against** — it is the first run in the project with
  a trustworthy long-horizon record.
- 39/39 tests pass (`cd bot/build && ./HiveTests`).
- **Uncommitted**, awaiting the user's review: the `spectate` subcommand
  (`bot/src/eval/Spectate.*`, `bot/src/eval/Checkpoints.*`,
  `bot/tests/test_checkpoints.cpp`, `scripts/spectate.sh`), the `Env.h` export
  of `BuildGeneralCurriculum`, CMakeLists entries, and doc updates to
  `docs/architecture.md`, `docs/metrics.md`, `scripts/watch.sh`,
  `runs/RUNLOG.md`.

## One correction to carry forward

`runs/RUNLOG.md` calls p1probe-f a "BREAKTHROUGH" and "the chosen config". That
entry compares the *first row to the last row* of a noisy 299-iteration CSV.
Bucket-averaged, probe-f's KL and clip fraction actually **decay ~3x across its
30M steps** and its touch ratio plateaus at 15M. The config is better than what
preceded it, but it was oversold, and p1-validate is what that oversell looks
like at 4x the horizon.

Always read a run bucket-averaged. Every table in this document came from:

```bash
scripts/summarize_runs.py --trend bot/build/metrics/main-p1-validate.csv
```
