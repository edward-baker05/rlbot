# Hivemind roadmap: 1v1 bot from scratch-adjacent to high GC

Date: 2026-08-17
Status: approved design, pre-implementation

## Context

A Rocket League bot trained with GigaLearn (C++) on RocketSim, deployed via
RLBot v5, on a Ryzen 3600 + RTX 2060. The owner is a GC1 player; the goal is a
bot that beats them in 1v1, aiming for high GC and possibly SSL. This is their
first serious ML bot and a learning project as much as a product.

The existing code (4 commits, largely AI-authored) is functional: both training
targets run on GPU, ~90k steps/sec at 512-wide, checkpointing and wandb metrics
work. Its inherited comments and docs are untrusted and the previous
architecture decisions are open for re-evaluation. The working tree (docs
deleted, config trimmed) is the baseline.

### Constraints

- Next ~month: part-time local training only (electricity costs at home);
  short runs are fine, 24/7 is not.
- After that (term time): 24/7 local training is fine indefinitely.
- Rented GPU compute is on the table later, but only if the trajectory
  justifies it.
- 1v1 only for this project. Multi-mode (2s/3s) is explicitly a possible
  future redesign that will reuse lessons, not code lineage.

## Decisions

### D1. Single policy; the kickoff/general split is removed

The two-policy design (kickoff policy to first touch, then general policy) is
deleted. Reasons:

- No top community bot works this way; kickoffs are learned by the main policy
  through a kickoff-weighted curriculum.
- The split doubles the training/deployment parity surface and adds a
  handover discontinuity immediately post-kickoff — a moment high-level humans
  attack.
- A dedicated kickoff policy can be added back later if evaluation shows
  kickoffs are specifically weak. Nothing else in the design depends on it.

### D2. 1v1-native observation

`DefaultObsPadded` stays but is built with `maxPlayersPerTeam = 1`: no dead
teammate slots, a narrower input, faster convergence. Moving to multi-size
later is a config change plus a fresh run — which any obs change forces
anyway. Slot shuffling and team inversion (orange plays "as blue") are kept;
humans still need no special handling.

### D3. Proven mechanics are kept as-is

`DefaultAction` (90 discrete actions), `tickSkip = 8`, `actionDelay = 7`.
These are community-standard and not worth innovating on.

### D4. No dribble/possession rewards, ever

Dribbling and flicking are a known local optimum: bots that learn them early
plateau, fail to develop air mechanics, can't defend what they can't do, and
undervalue boost (flicks work on empty tanks). Therefore:

- No reward term ever pays for ball proximity, carry time, or "possession".
- The `groundDribble` curriculum state is removed permanently.
- Possession-like play is expected to emerge from zero-sum outcome pressure
  and self-play, or not at all.

### D5. The air game starts before the ground game is comfortable

Aerial exploration must happen while entropy is still high and habits are
unformed. `TouchHeightReward` and aerial-heavy state setters enter at phase
P2 (as soon as ground striking is reliable), not after a possession phase.

### D6. Weights are derived from telemetry, not guessed

Per-term reward-share logging is built before any long run. Every phase
transition re-weights based on the measured share each term actually paid in
the preceding run. No magic numbers anywhere in the project without
measurement behind them; where the spec below names a starting value, it is a
starting point for a measured sweep, not a commitment.

### D7. Gates are trends and comparisons, never absolute thresholds

Progress gates are expressed as: rating rising monotonically over a window,
metric X climbing while metric Y holds, checkpoint N beats checkpoint N−1
head-to-head. Any eventual concrete number is calibrated from this project's
own baseline runs.

### D8. The reward curve is never evidence

It rises in every run including broken ones. Evidence is: skill rating against
frozen references, behavioral metrics (touch ratio, touch height, per-scenario
outcomes), watching games in RocketSimVis, and match results.

## Code restructure

Removed:

- `policy/Regime.*`, `train-kickoff`, `TrainTarget`, `KickoffRewardWeights`,
  `BuildKickoffRewards`, the kickoff branch of `CreateEnv`.
- `PolicySet` becomes a single-model `Policy` wrapper over one
  `GGL::InferUnit`. Batched hivemind inference stays; regime routing goes.
- `TeamSizeMix`, asymmetric team sizes, the Weyl stratification in
  `PickTeamSizes`. Every game is 1v1.
- The `groundDribble` state setter entry (D4).

Kept (trimmed to 1v1 where applicable): the `env/` module layout, the metrics
callback and step-budget hook in `Train.cpp`, the CLI (`train | play`), the
RLBot client and `PacketConvert`, `Config.h` as the single tunables file, and
all build arrangements (PIC, explicit NCCL/NVSHMEM paths, the two `external/`
patches).

Comment policy: inherited AI commentary is removed or rewritten as files are
touched; only verified *why*-comments survive. CLAUDE.md is updated to match
the new architecture (it currently mandates the two-policy design and links
deleted docs).

## Training configuration

Starting values, with the change rationale; all subject to D6/D7:

| Knob | Value | Note |
|---|---|---|
| Network | shared {512,512}, policy/critic {512,512,512}, LayerNorm, ReLU | unchanged; widening (768→1024) is a planned event at a phase gate, justified by a plateau with healthy metrics |
| tsPerItr / batch | 100k (up from 50k) | 1v1 steps are ~3x cheaper; larger batches are the most-repeated advice from GC-level trainers |
| gaeGamma | 0.99 → ~0.995 at P2 | short horizon for touch learning; longer once positioning matters |
| entropyScale | 0.035, decayed from P3 | exploration pressure early; precision late |
| LR | 1.5e-4, halved at plateau events | not on a schedule |
| numGames | re-measured sweep at 1v1 (expect ≥256) | take whatever maximizes measured steps/sec |

Run hygiene: every run has a `--label`, a wandb group, and a line in
`runs/RUNLOG.md` recording the config diff and the reason for the run.
Comparisons are only made between labeled runs.

## Reward and curriculum roadmap

Reward/curriculum changes resume the same checkpoint; only shape changes
(obs, action, network) force fresh runs. Expect a temporary rating dip after
every transition; judge the phase, not the first day.

### P1 — Contact & striking

Intent: touch the ball, hit it hard, toward the net.
Terms (current stack survives, weights re-derived per D6): velPlayerToBall,
zero-sum StrongTouch, zero-sum velBallToGoal, goal, boost pickup, faceBall.
Curriculum: ballContact-heavy, neutralPlay substantial, kickoff ~10%,
defend/recover/aerial minor.
Gate: striking is reliable (watched and measured), goals occur regularly,
rating rising monotonically.

### P2 — Striking + aerial foundations

Intent: powerful/placed shots, saves, and the beginnings of an air game —
before ground habits set.
Changes: goal weight dominant and rising; velPlayerToBall cut hard (its job
is done); TouchHeightReward enters at moderate weight; aerial setter weight
up; boost economy terms present (D4 note: pickup + outcome pressure, no
possession shaping). Self-play against old versions switches on here and
stays on permanently — the structural defense against farming.
Gate: touch height climbing while touch ratio and ground striking hold;
rating still rising; shot/save behavior visible in vis.

### P3 — Advanced air mechanics

Intent: the mechanics that matter at the target level — flip resets, double
taps, air-dribble-to-bump.
Approach: dedicated state setters (backboard/double-tap and wall-carry are
new code; FlipResetState exists) plus touch-quality rewards scoped to those
situations. Deliberately under-specified: this phase is designed at its own
gate, informed by P2's telemetry and a fresh survey of community results.
Committing to details now would be guessing (D6).
Gate: wins the mechanic-specific scenario episodes; air mechanics appear
unprompted in neutral play; ground game has not regressed.

### P4 — Outcome-dominant refinement

Intent: the long soak where high-GC play does or doesn't emerge.
Changes: shaping annealed toward zero (reward is mostly goals plus a thin
zero-sum layer), entropy low, LR low, gamma high. Multi-billion steps. The
network-widening decision sits at this boundary.

## Evaluation

Layers, cheapest first:

1. **Live wandb metrics** (every run): existing stream plus per-term reward
   shares (the farming detector) and per-scenario outcome rates ("is it
   learning aerials" as a number).
2. **RocketSimVis observation** (frequent, free): the owner watches games —
   always within the first hour after any reward/curriculum change (when new
   degenerate behavior appears), and at any checkpoint of interest. Their GC1
   eye is an evaluation instrument; qualitative findings feed gate decisions
   and reward fixes (this already caught backwards-driving once).
3. **Skill tracker** (measurement runs and gates): rating vs a frozen
   reference pool — checkpoints saved at each phase gate — rather than only
   drifting recent selves. Costs ~30% throughput, so production runs use it
   sparingly.
4. **RLBot benchmark matches** (semi-manual, roughly weekly once available):
   `scripts/match.sh` runs a checkpoint against a configured opponent and
   records the score. Opponent ladder escalates: RLBot pack bots → Necto →
   Nexto. Open probe: Nexto is RLBot v4; v4-bridge-under-v5 compatibility
   needs a one-evening test.
5. **Humans**: periodic 1v1 vs the owner (GC1), later vs a stronger player
   they know. `scripts/play.sh` exists.

A short **metrics guide** doc lists, per metric: what it measures, what
healthy looks like, and which decision it feeds. It is a living document.

## Timeline

### Phase 0 — Infrastructure month (now → term time; part-time, short runs)

1. Code restructure per above; comments purged as touched.
2. Per-term reward-share and per-scenario metrics.
3. Throughput re-measure at 1v1; numGames sweep.
4. **Deployment verified live**: `pipx install rlbot`, throwaway checkpoint,
   real RLBot match. This is the single biggest unknown in the pipeline and
   blocks evaluation layers 4–5. Includes the Nexto/v4-bridge probe.
5. `scripts/match.sh`, frozen-reference-pool support, `runs/RUNLOG.md`,
   metrics guide.
6. Short validation runs (single-digit hours) to re-derive P1 weights from
   telemetry and confirm the touch→strike trajectory.
7. Fresh, accurate docs replacing the deleted set; CLAUDE.md updated.

Exit criterion: a two-week unattended run could start tomorrow and every
number it produces would be trusted.

### Phases 1–4 (term time, 24/7)

P1 run (expect hundreds of millions of steps) → gate → P2 (order 1B) → gate →
P3 (designed at its gate) → gate → P4 (multi-billion soak). No calendar dates:
boundaries are metric-gated (D7), and step-count estimates are expectations,
not plans.

Rented-hardware decision points sit at the P2→P3 and P3→P4 gates. Rule: rent
only if the trajectory is good *and* measured steps/sec is the binding
constraint. Never rent to rescue a run whose metrics look wrong — compute
amplifies a recipe, including a bad one.

## Testing & verification

- **Unit tests** (Catch2 or doctest, single-header, small CMake test target)
  for pure logic: state setter spawn validity (in bounds, wheels down where
  claimed, ball where claimed), reward terms (hand-built states → expected
  sign and magnitude ordering), metric/gate computations. These are the
  components where a silent bug wastes a two-week run. TDD applies to this
  new pure-logic code.
- **Executable parity check**: a `verify` subcommand loads a checkpoint the
  way the RLBot client does, feeds recorded observations through the
  training-side and deployment-side paths, and asserts outputs match. Run
  before every deployment session. This mechanizes CLAUDE.md's prose-only
  parity warnings.
- **Smoke runs**: `--max-steps` short runs after any env/reward change,
  watched in vis, per-term reward shares checked against intent.

## Risks and open questions

- **RLBot v5 client has never run live.** Addressed first (Phase 0 item 4);
  until then it is the project's largest unknown.
- **P3 mechanics training is uncharted here.** Deliberately deferred to its
  gate; mitigated by scenario setters + scoped rewards as the working theory
  and by re-surveying community practice at that time.
- **Hardware ceiling.** If SSL turns out to genuinely need order-10B+ steps,
  the 2060 needs weeks-to-months or rented compute; the gates make that a
  data-driven decision.
- **Skill tracker measures relative progress only.** Mitigated by frozen
  reference pools and external benchmark matches.
- **GigaLearnCPP-Leak is unmaintained upstream.** Local patches are
  documented in CLAUDE.md; no updates are expected or required.
