# v2 design probes: test the day-one decisions where they are valid

**p20air is CANCELLED.** It proposed testing two things -- doubling the jump
actions (issue 03) and the `aerial` spawn scenario (issue 04) -- on the p18
checkpoint. Both are interventions that only work from a fresh policy.
`Action/Jump When Grounded Upright` is **0.050 against an unmasked null of
0.20**, i.e. 4x below chance, and it held there across 483M steps at two
different entropy levels. PPO's gradient is proportional to the probability of
the action taken, so an action at near-zero probability receives near-zero
gradient no matter what is offered it. A null from p20air would not be weak
evidence, it would be **misleading** evidence, and it would retire a design that
is probably correct.

Everything in this project from here is input to a from-scratch v2. So the
probes below test v2's day-one decisions **in the regime where v2 will actually
live**: fresh, small, and cheap.

## The prerequisite, now done

`eval` built ONE observation builder and ONE action parser and handed both to
both sides, so two checkpoints could only ever meet if they shared an
observation and an action table. Each side now builds its own pipeline from its
own `CONFIG.json` (`Eval.cpp`, `RunEval`). Verified 2026-08-23: p19pool
(`RelativeObs`, 109 floats) vs p8ref (`DefaultObsPadded`, 89 floats) plays and
scores 23-0.

**This is what lets v2 be measured against this entire lineage.** Without it, the
day v2 changes its observation is the day it loses every reference point and
goes back to an unanchored self-play Elo -- which is the failure that cost this
project eighteen runs.

## Sizing, from this project's own numbers

`main-p15manual` reached `Rating/1v1` ~98 and `Touch/Above 450` 0.19 by 300M
from a cold start, so a fresh policy is playing a recognisable game by then. At
the measured **64.1k steps/s**, 300M is **1.4 h**. Three probes is ~4.5 h.

## The probes

All three start **fresh** (`--fresh`, no seed checkpoint), run to **300M**, and
are identical except for the single named change. Reward stack, obs,
architecture, LR, gamma, entropy and `tsPerItr` frozen at p18 values.

### Probe A -- control

Current stack, `SpawnMode::Random`, the 90-action table. Establishes what 300M
from cold is worth on the ladder, and is the opponent B and C are measured
against.

### Probe B -- doubled jump actions (issue 03)

Zealan's own remedy, one of the four trusted sources, and **never once tried in
this project**: *"add more jump actions... doubling the jump actions seems to be
enough to eliminate the need for air rewards."*

**Implementation note.** `MakeActionParser(bool masked)` currently varies only by
mask, so B needs a new action-table option: add it to `TrainConfig`, record it in
`CONFIG.json`, and read it in `Eval.cpp`'s `CheckpointConfig` (the per-side
plumbing is already there and this is the only thing still selected globally).
Recompute every action-space null in `docs/metrics.md` in the same commit and
re-assert them in `bot/tests/test_actionspace.cpp` -- they all move.

**Prediction:** `Action/Jump When Grounded Upright` ends **above its new
unmasked null**, against p18's 4x-below. `Touch/Above 450` and `Phase/Aerial`
exceed probe A. Beats A on the ladder.

### Probe C -- the curriculum (issue 04, and much more)

`SpawnMode::Curriculum` has **never executed a single step in this project's
history.** `Env.cpp:19-21` only builds it on the `Curriculum` branch and
`TrainConfig::spawn` has always been `Random`, so the entire `CurriculumWeights`
block is dead code: `neutralPlay` 35, `ballContact` 10, `defend` 15, `recover`
8, `strike` 15, `aerial` 10, `kickoff` 8.

Two consequences already measured: the bot **has never trained on a kickoff**
(`docs/architecture.md:12` asserts the opposite), and `RandomState(true, true,
false)` spawns cars **in the air**, uniformly in z 150-1820, so every episode
begins mid-scramble at a state that never occurs in real play.

**Prediction:** C is worse than A at 300M on the in-distribution ladder (its
spawns are harder and more varied) and **better on the kickoff ladder**. If that
split appears, v2 should train on the curriculum and this project has been
measuring itself on the wrong distribution for 3B steps.

## How they are compared

**Primary: head-to-head, A vs B and A vs C.** Direct A/B with a true null of
50%, no anchor required.

**Secondary: a shared fixed anchor for absolute scale.** Use **p12goal (250M)**
and **p8ref (100M)** -- both weak enough that a 300M fresh probe can register
against them. p16 and p18 are far too strong to resolve anything at this size.

Every ladder run carries a **SELF-control** rung. It read 49.3% and 50.5%
against a truth of 50% on 2026-08-23; without it the first ladder I ran looked
meaningful and was not.

## Kill criteria

Stop any probe at 100M if `Policy Entropy` has not fallen, or if
`Obs/Non-Finite Rate` is non-zero. Both are cheap to check and both have cost
this project a run before.
