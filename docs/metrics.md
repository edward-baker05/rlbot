# Metrics guide

A living document. For each metric family: what it measures, what healthy
looks like (trends and comparisons, never absolute thresholds), and which
decision it feeds. Metrics land in wandb and in `bot/build/metrics/<run>.csv`.

**The reward curve is never evidence.** It rises in every run, including
broken ones. Everything below exists so you never have to argue from it.

## PPO health — read these first

These say whether the policy is *moving at all*. Every other metric below is
meaningless until they look right: a frozen policy produces flat behavioural
metrics and perfectly plausible reward shares, and the two are
indistinguishable from a bad reward function by inspection.

| Metric | Measures | Healthy | Feeds |
|---|---|---|---|
| `Policy Entropy` | Action-distribution entropy, **normalized to [0,1]** by GigaLearn (divided by log(numActions)) | Falling over a run. A value near the ~0.78 start held flat means the policy is still near-uniform, i.e. untrained | Is anything being learned at all |
| `Mean KL Divergence` | Change between successive policies | Clearly above zero (order 1e-3 here). ~0 means successive policies are identical | Same |
| `SB3 Clip Fraction` | Fraction of samples hitting the PPO clip | Non-zero. ~0 means updates are too small to clip | Same |
| `Policy Relative Entropy Loss` | Entropy bonus as a multiple of the policy-gradient term | Order 0.1 or less. Large negative values (-1, -22) mean the entropy bonus is drowning the gradient — lower `entropyScale` | `entropyScale` |
| `Policy Update Magnitude` | Size of the parameter step | Non-trivial and not collapsing to zero | LR / epochs |

This ordering is not stylistic. In the 2026-08-17 probe sequence, three
consecutive reward rebalances (probes a–c, ~100M steps) moved `RewardShare/*`
exactly as computed and changed behaviour not at all, because the policy was
pinned at near-uniform by `entropyScale`. `RewardShare/*` tells you where
reward mass goes; it says nothing about whether the policy can act on it.

## Nulls — read this before citing any behavioural metric

**No run conclusion may cite a metric whose chance value is unknown.**

This project spent eight runs reading `Player/Velocity Alignment ~ 0.30` as a
low but real number. It is the null. Nothing had been learned, and the entire
p1-p7 sequence of reward verdicts was drawn against a quantity that had never
moved off random.

| Metric | Null | Where it comes from |
|---|---|---|
| `Player/Velocity Alignment` | **0.3183** grounded, **0.25** airborne | `E[max(0,cos)]` for a uniform direction: `1/pi` in a plane, `1/4` in 3D |
| `FaceBall/Rectified` | same as above | same quantity, nose instead of velocity |
| `Action/Jump When Grounded *` | **0.4286** masked with boost, **0.50** masked dry, **0.20** unmasked | jump actions / actions the mask leaves available |
| `Action/Steer Nonzero` | **0.3810** masked, **0.5333** unmasked | steering actions / available, sampled on grounded upright steps only |
| `Player/In Air Ratio` | **~0.87** masked, **~0.75** unmasked (estimate) | air stint ~15 steps against ground dwell `1/p_jump`; not exact, but p1advnorm measured 0.886 at jump rate 0.43 |

Every figure above except the last is asserted in `bot/tests/test_actionspace.cpp`,
so an upstream change to the action table breaks a test instead of quietly
invalidating this page.

The jump prior is a **range**, not a point: `GetActionMask` re-enables the
boosted jump actions after filtering boost, so a dry car sees 50%. p7approach
ran at `Player/Boost` 12-15 out of 100, i.e. mostly in the dry regime.

Two rules follow. A metric at its null is **not** a weak signal, it is the
absence of one. And any new behavioural metric ships with its null in this
table, computed, in the same commit that adds it.

## Reward and behaviour

| Metric | Measures | Healthy | Feeds |
|---|---|---|---|
| `RewardShare/*` | Fraction of realized \|weighted reward\| per term (one sampled player per arena; zero-sum terms measured pre-zero-sum) | Outcome terms (Goal, StrongTouch) growing over a phase; no shaping term dominating late | Phase-gate decisions; farming detection |
| `Scenario/*/Share` | Fraction of arenas running each curriculum scenario | Matches configured curriculum weights | Verifying setters actually run |
| `Scenario/*/EndedInGoal` | Episode outcome per scenario | Rising within a phase | Whether a scenario is being learned |
| `Player/Ball Touch Ratio` | Touch frequency | Stable or rising; a collapse after a reward change = new degenerate behavior | Reward-change rollback |
| `Player/Touch Height` | Air game development | Rising from P2 onward without touch ratio falling | P2/P3 gates |
| `Phase/*` | Time share per play phase (metric labels only, not curriculum) | Neutral dominant; shifts tracking curriculum changes | Curriculum tuning |
| `Rating/*` (skill tracker) | Elo vs. past versions | Monotonic rise; falling while reward rises = farming | The primary gate signal |
| `Game/Goal Speed` | Shot power at goals | Rising through P1–P2 | Striking quality |

## How to watch a run

1. **PPO health first.** Entropy falling, KL above zero, clip fraction
   non-zero. If not, stop — no reward or curriculum conclusion drawn from this
   run is valid, and the fix is a hyperparameter, not a weight.
2. **Then `RewardShare/*`.** If a shaping term (VelPlayerToBall, FaceBall)
   is absorbing the reward mass late in a phase, the policy is farming it.
3. **Then `Rating/*`.** Reward up + rating down is the farming signature.
   Note it only logs every `skillUpdateInterval` iterations (100 by default,
   = 10M steps), so short probes produce one or two points and no trend.
4. **Then RocketSimVis** (`scripts/vis.sh` + `scripts/spectate.sh <label>`)
   for a few minutes — always within the first hour after any reward or
   curriculum change, which is when new degenerate behavior appears. A GC1 eye
   catches what no metric does (it caught backwards-driving once).
   `spectate.sh` follows a live run without disturbing it; `watch.sh` starts a
   separate learner and is for checking code you just wrote, not for watching
   a run. See `docs/architecture.md`.

Behavioural metrics are noisy iteration to iteration; compare windowed
averages, never single rows:

```bash
scripts/summarize_runs.py --trend bot/build/metrics/<run>.csv   # one run, over time
scripts/summarize_runs.py <run-a>.csv <run-b>.csv               # run vs run
```

`--trend` buckets a single run and averages each bucket, which is how a plateau
becomes visible. The default tail-mean view cannot show one: a run that learned
for 40M steps and then stalled for 80M still has a healthy last quarter. That
is exactly how `p1-validate` was read.

Comparisons are only valid between labeled runs recorded in `runs/RUNLOG.md`.
