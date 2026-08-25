# Ball-Prediction Observation Block

**Date:** 2026-08-25
**Status:** Design approved, pending implementation plan

## Problem

The policy is a feed-forward MLP (`sharedHeadLayers [1024, 1024, 512]`, `policyLayers
[512]`) with no recurrence and no frame stack. It observes exactly one ball snapshot per
step — position, velocity and angular velocity (`AdvancedObsPadded.cpp:23-25`) — sampled
at 15 Hz with a 7-tick action delay.

Future ball position is a deterministic function of that snapshot plus fixed arena
geometry, so precomputing it adds no information in the Shannon sense. It is nonetheless
worth adding, because the function is *piecewise*: floor, wall, ceiling and corner
bounces are discontinuities, and MLPs represent sharp piecewise maps poorly. Supplying
the trajectory converts a hard function-approximation problem into a lookup. This is the
same reason every scripted RLBot carries a ball predictor.

## Non-goals

- Predicting opponent or teammate trajectories.
- Any change to the reward budget, state setters or terminal conditions.
- Replacing the existing `Default`, `Advanced` or `Relative` obs modes.

## Feature block: 24 dimensions

Appended to the end of the existing observation, so all current dimensions keep their
indices. Current `Advanced` obs at `maxPlayersPerTeam=3` is **225** dims
(9 ball + 8 prev-action + 34 boost pads + 6 × 29 player blocks); the new mode is **249**.

### Trajectory samples — 18 dims

Six predicted ball positions on a geometric schedule, ratio ≈ 1.75, each snapped to the
nearest 120 Hz tick:

```
t = 0.15, 0.30, 0.55, 0.95, 1.60, 2.60 seconds
```

Geometric rather than uniform because prediction accuracy decays with horizon: in 1v1 the
ball is touched every one to two seconds, so anything past ~1.5s is counterfactual
("where it goes if nobody intervenes"). Dense near-horizon resolution is where the
accurate, actionable signal lives.

Each sample is encoded in the car's local frame, matching the convention
`AdvancedObs::AddPlayerToObs` already uses at line 15 for current ball position:

```cpp
obs += phys.rotMat.Dot(predPos - phys.pos) * POS_COEF;
```

**No per-sample velocity.** It is approximately the finite difference of adjacent
samples — redundant, and it would double the block size.

### Event features — 6 dims

| Dim | Feature | Encoding |
|-----|---------|----------|
| 1 | Time to next bounce (first contact with any surface: floor, wall, ceiling, corner) | Divided by 2.60s, clamped to 1.0 if no bounce occurs within the sample horizon |
| 2-4 | Position of that bounce | Team-normalized field frame, `POS_COEF` scaled; zero if no bounce |
| 5 | Trajectory enters a net within the sample horizon | −1 own net, 0 neither, +1 opponent net |
| 6 | Time to that goal | Divided by 2.60s, 1.0 if no goal |

All times normalize against the **2.60s sample horizon**, not the 6s simulation horizon —
the extra simulation is a caching buffer, not part of the feature space.

These earn their place on a criterion the samples cannot meet: they encode
**discontinuities the sparse schedule straddles and misses**. Between the 0.95s and 1.60s
samples the ball can hit the floor and reverse, and nothing in the position block records
where or when. Likewise, crossing the goal plane is a binary event no position sample
represents.

A "time until ball drops below reachable height" feature was considered and **rejected**:
it is interpolable from the z-components the samples already carry (MLPs handle smooth
interpolation fine — it is only discontinuities they struggle with), and "reachable"
is not a well-defined quantity, since it depends on the car's boost, flip availability
and distance to the intercept. Any fixed threshold would encode the wrong abstraction.

## Prediction mechanism

### Source

A car-less `Arena::Create(GameMode::SOCCAR)` owned by the obs builder. Ball state is
copied in from `state.ball` and the arena stepped forward.

Deliberately **not** RLBot's `ballPrediction` packet parameter (`DashBot.h:54`), for a
reason that has nothing to do with the quality of RLBot's predictor — which is purpose-built
for this and perfectly good:

**RLBot is not present during training.** Training runs 256 headless RocketSim arenas
inside `Train.cpp`; there is no RLBot framework, no packet, and no `BallPrediction`
anywhere in that process. A RocketSim predictor therefore has to exist regardless of what
the deploy path does. It is not avoidable work being chosen over the built-in — it is the
only way the feature can exist at all.

Given that it must exist, using RLBot's at deploy would mean maintaining two predictors
and hoping they agree. The observation vector is the network's entire world: if column 231
means "ball position at 0.55s per RocketSim" across the whole training run and then means
"per RLBot core" at deploy, the network is silently fed a different world, producing
degraded play with no error anywhere. The saving would be nil, since the RocketSim
predictor was written either way.

The speed argument points the same direction. Prediction cost only matters in training,
where RLBot is unavailable; at deploy it is one arena at 15 Hz on an otherwise idle
machine, where 360 ball-only ticks is free.

Note also that `main.cpp` currently connects with `ballPrediction=false`, so RLBot is not
even sending it today — using it would mean turning that on as well.

Owning the arena means training, `Spectate`, `MatchBench`, `NectoBench` and `DashBot` all
derive identical features from one code path.

The obs builder must not depend on `state.lastArena`, which is documented as possibly
null and is not available on the deploy path.

### Caching — the performance-critical part

Naively this costs 360 ball-only ticks per 8 full-arena ticks. That is unaffordable
across 256 arenas on a 2060. Two facts make it tractable:

1. **A ball trajectory is invalidated only by a touch** (or goal/reset). Between touches
   it is deterministic and can be simulated once and consumed.
2. **Both players in an arena share one trajectory** (orange simply inverts).

Caching strategy:

- Simulate 6 seconds ahead, retain the tick-indexed trajectory.
- Consume it as time advances; sampling becomes table lookup.
- Re-simulate only when the touch tick advances, or when remaining horizon drops below
  the 2.6s the schedule needs.
- Key the within-step cache on `state.lastTickCount` so both players reuse one result.
- `ObsBuilder::Reset(initialState)` clears the cache on episode start.

**Implementation step 1 is measuring the amortized cost, not assuming it.** If the
measured throughput loss exceeds ~15% SPS on the target hardware, the feature does not
ship as designed and the schedule/horizon must shrink.

### Wiring

New `ObsMode::Predictive` added alongside the three existing modes in `Obs.h`, dispatched
in `MakeObsBuilder`. It **extends `AdvancedObsPadded`** — the mode t1 trains on — emitting
the identical 225 dims followed by the 24 new ones. `RelativeObs` is not extended in this
change.

`Train.cpp:426` prints the obs mode with a two-way ternary
(`cfg.obs == ObsMode::Advanced ? "Advanced" : "Default"`) that would mislabel the new
mode; replace it with a proper name lookup as part of this work.

Existing modes are untouched, so the `t2`, `main-p19pool`,
`687282952` and `747642208` checkpoints still load and `MatchBench` comparisons stay
valid. `CONFIG.json` records obs mode as a string, so the new value slots in without a
format change.

## Migrating t1's 890M-step checkpoint

### Why zero-padding is exactly equivalent

In `Models.cpp:17-22`, LayerNorm is applied *after* each hidden Linear, never to the
input. The raw observation feeds straight into `Linear(obsSize, 1024)`. Widening that
layer's input with zero-initialized columns is therefore **exactly functionally
identical**: same policy distribution, same value estimates, same entropy, no warm-up
collapse. Had LayerNorm been applied to the input, changing the input width would have
shifted the normalization statistics of every existing dimension and this approach would
not work.

### Procedure

An offline `migrate-obs` subcommand in this repo (`main.cpp` dispatches subcommands, not flags). C++ rather than a Python round-trip,
because the checkpoints are libtorch `torch::save(seq)` archives and the same `GGL::Model`
class is already available here.

1. Build `Model` with `numInputs = 225`; load `SHARED_HEAD.lt`.
2. Build `Model` with `numInputs = 249`.
3. Copy every parameter. For `seq[0]`'s weight, copy into `[:, :225]` and zero
   `[:, 225:]`. Bias copied verbatim. (Confirmed: the saved weight tensor is (1024, 225).)
4. `POLICY.lt` and `CRITIC.lt` sit downstream of the shared head — copy untouched.
5. `SHARED_HEAD_OPTIM.lt`: pad that weight's Adam moments the same way.
   `AdamParamState` exposes `step`, `exp_avg`, `exp_avg_sq` and `max_exp_avg_sq` as
   public accessors, and `Optimizer::state()` is keyed by
   `param.unsafeGetTensorImpl()`, so the old state can be read after loading into the
   old-shaped model and re-inserted against the new parameters with padded tensors.
   *Fallback if this proves impractical:* reset the shared-head optimizer state and run
   a few hundred iterations at reduced LR. At `policyLR = 1e-4` the transient is small,
   but this is second choice because it perturbs a converged policy.
6. `RUNNING_STATS.json` needs no migration. It holds only `return_stat`,
   `skill_ratings`, `run_id` and counters. `standardizeObs` defaults to `false`
   (`LearnerConfig.h:48`) and `Train.cpp` never sets it, so `obsStat` is NULL — there is
   no observation normalizer whose width would need extending.
7. **`policy_versions/` must be migrated too.** Those snapshots are the self-play
   opponent ladder (`maxOldVersions = 16`), and `Model::Load` hard-checks parameter
   sizes and calls `RG_ERR_CLOSE` on mismatch. Apply the same padding to each. Dropping
   the pool instead would forfeit the entire opponent-diversity ladder.
8. Necto is unaffected — separate obs pipeline and separate model.

### Acceptance test

Run the migrated checkpoint against the pre-migration checkpoint in `MatchBench`,
deterministic, same seed. With zero columns the two must be behaviourally identical.

This test validates the surgery independently of whether the feature helps, and must
pass before any training resumes.

## Run strategy

Snapshot t1 at its current 890,047,828 steps. Graft into a **new run `t3`**. Stop t1 and
retain the frozen snapshot as a fixed benchmark opponent.

Rationale: grafting in place would make it impossible to attribute any later Elo movement
to this feature, and measurement is the binding constraint on this lineage. Running t1
and t3 concurrently would be cleaner science but halves throughput on each on a single
2060 — a frozen benchmark opponent gives most of the attribution benefit at no
throughput cost.

## Success criteria

1. Amortized prediction cost under ~15% SPS loss (gate — measured before anything else).
2. Migration acceptance test passes: bit-identical behaviour with zero columns.
3. Over the first 10-20M steps of t3, the new columns acquire non-trivial weight
   magnitude in `seq[0]` — evidence the network is using them at all.
4. Necto benchmark Elo for t3 exceeds frozen t1's over a comparable step budget.

## Prior art in this repo

Commit `2636cc2 "Removed lookahead"` deleted a `RolloutPlanner` that generated 32
candidate actions and rolled each out 12 ticks in a scratch `Arena` to pick the best. That
is a **different mechanism** — inference-time action search that overrode the policy's
output — and it ran only on the deploy/eval paths, never in training. This proposal instead
feeds the policy better inputs and lets training decide what to do with them. The relevant
inheritance is the scratch-arena lifecycle pattern (`EnsureArenaInitialized`, arena deleted
in the destructor), which this design reuses.

## Known risks

- **Cost.** The dominant risk. Mitigated by the measurement gate at step 1.
- **Passivity.** A network that can see where the ball will land has an easy route to
  "drive to the bounce point and wait". Given the recent defense-oriented reward work
  this may be desirable or may be a failure mode; worth watching in `Spectate`.
- **Horizon fiction.** The 1.60s and 2.60s samples are mostly counterfactual in 1v1.
  If success criterion 3 shows the far samples acquiring no weight, drop them.
- **Optimizer state migration** (step 5) is the fiddliest part of the surgery and has an
  identified fallback.
