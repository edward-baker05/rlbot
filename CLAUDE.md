# CLAUDE.md

Context for working in this repo.

## What this is

A Rocket League bot: trained with GigaLearn (C++) on RocketSim, deployed through
RLBot v5. Target is a bot that beats a GC1 human in 1v1.

**One policy.** The old kickoff/general two-policy split was removed
deliberately on 2026-08-17 (see
`docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`, decision D1).
Do not reintroduce a policy split or an MoE design without a new decision.
Situation labels (`PlayPhase`) survive only as metrics; curriculum scenarios
(`CurriculumState`) only as an optional spawn distribution.

## Where the project actually is (2026-08-18)

**Read `docs/superpowers/specs/2026-08-18-known-good-baseline-design.md` before
changing rewards, spawns, the action space or PPO settings.** It supersedes the
reward direction in the 2026-08-17 roadmap and the whole of
`2026-08-18-reward-redesign-design.md`.

Two measurements set the current direction:

1. **`Player/Velocity Alignment` has never left its null.** It reads 0.3135 in
   p7approach against a chance value of `1/pi = 0.3183` for a uniform direction
   in the ground plane. The bot has never driven at the ball in any run, in
   ~25 experiments, while 40-45% of reward mass paid for exactly that. The null
   was never computed, so 0.30 was read as low-but-real.
2. **The reference reaches frequent ball contact in "a few dozen million
   steps"** (Zealan's `making_a_good_bot.md`). We were at 0.16 touches per
   episode at 100M. The gap is not patience.

So the project is in **phase C: reproduce a known-good configuration before
building anything of its own.** The current stack is Zealan's early-stage
config ported as literally as the C++ allows -- four rewards, no goal term,
signed FaceBall, `RandomState` spawns, unmasked actions, LR 2e-4, 50k rollouts.
None of it is this project's invention, deliberately.

Phase B (bisect our ideas back in, one variable per run) and phase A (relative
observation, bigger network, higher gamma, external-opponent Elo) come after,
and only after, the reproduction gate passes.

### Three rules that outlast the phase

**Every behavioural metric ships with its null.** No run conclusion may cite a
metric whose chance value is unknown. The table lives in `docs/metrics.md` and
the action-space figures are asserted in `bot/tests/test_actionspace.cpp`. A
metric at its null is not a weak signal; it is the absence of one.

**One variable per run, with the prediction written down first.** Almost every
row in `runs/RUNLOG.md` changed two or more things at once (p6budget: 8-term
rewrite + `tsPerItr`; p7approach: 5-term rewrite + `entropyScale`), which is
why the log keeps retracting itself. Predictions go in the RUNLOG *before* the
run, along with the step count at which the run gets killed.

**Every reward weight is a budget, converted in exactly one place**
(`Hive::GeneralRewardSpecs`). Never write a bare per-step float. **The unit is
one ball touch**, not a goal -- a goal arrives 0.116 times per episode and
cannot be audited against telemetry, which is why every post-mortem had to
reconstruct the ledger by hand.

### Two traps that cost this project runs

**`entropyScale` is not comparable to rlgym-ppo's.** GigaLearn normalizes
entropy by `log(numActions)`. 0.035, 0.018 and 0.01 have each pinned the policy
near-uniform (p1probe-d, p1probe-g, p7approach); 0.002 produced the only
breakthrough. Check `Policy Entropy` is falling within the first 10M steps of
any run, and stop if it is not -- no reward conclusion is available from a
policy that cannot move.

**`ObsBuilder::Reset` is called in training and NOWHERE on the deployment
path.** `EnvSet` calls it on every episode reset; `HivemindBot` never does. Any
stateful observation (action stacking, frame stacking, running normalizers)
would therefore train and deploy differently with no symptom. `RelativeObs` is
stateless for exactly this reason. If you add state to an obs builder, plumb a
reset into `HivemindBot::Initialize`/kickoff first.

**`DefaultAction::GetActionMask` doubles the grounded jump prior.** It offers a
grounded car 42 of 90 actions, 18 of which press jump (42.9%, rising to 50% on
an empty tank because the jump mask is OR-ed back in after the boost filter).
Python RLGym's `LookupAction` applies no mask: 18/90 = 20%. Eight runs were
spent fighting air time whose prior our own mask doubles. `TrainConfig::maskActions`
selects it, `Hive::MakeActionParser` is the only construction site, and it is a
**parity trap** -- it must match at deployment via `HIVE_MASK_ACTIONS`.

## Layout

- `bot/src/` — the only code that is ours. Everything else is third-party.
- `bot/tests/` — doctest suite; builds as the `HiveTests` target, runs from
  `bot/build` (`cd bot/build && ./HiveTests`) so collision meshes resolve.
- `external/` — GigaLearnCPP-Leak, cpp-interface (RLBot v5), RocketSimVis. Each
  has its own git history; do not commit changes here without noting them.
- `libs/` — libtorch, NCCL, NVSHMEM. Gitignored; `scripts/setup_libs.sh`
  reinstalls them.

## Build

```bash
scripts/build.sh
```

Builds `HivemindBot` and `HiveTests`. Three things about the build that are
easy to trip over:

1. **PIC is required globally.** GigaLearnCPP is a shared library that links
   RLGymCPP and RocketSim statically, which only links on x86-64 if those are
   position-independent. `CMAKE_POSITION_INDEPENDENT_CODE ON` in
   `bot/CMakeLists.txt` handles it.

2. **NCCL and NVSHMEM must be linked explicitly.** `libtorch_cuda.so`
   references their symbols even for single-GPU use. They ship as versioned
   files (`libnccl.so.2`) with no unversioned symlink, so `-lnccl` does not
   resolve — full paths are linked instead (into `HiveCore`, PUBLIC).

3. **`CMAKE_POLICY_VERSION_MINIMUM` is pinned to 3.10.** Upstream declares
   minimums that CMake 4 rejects. Pinned in our CMakeLists rather than by
   editing `external/`, so `git pull` there stays clean.

### Local patch to external/

`external/cpp-interface/library/Client.cpp` has an added `#include <climits>`.
GCC 16 no longer pulls it in transitively and `CHAR_BIT` fails to resolve. If
you re-clone cpp-interface, reapply it.

## Verified working (2026-08-18)

- The reproduction stack is wired and smoke-tested (`HiveTests` 49 cases pass,
  run from `bot/build`). A 250k-step run confirms four `RewardShare/*` terms
  reach the CSV, `Scenario/*` correctly disappears under `RandomState`, and
  `Action/Jump When Grounded Upright` reads **0.216** — the unmasked null of
  0.20, i.e. the unmask took effect. `p8ref` then passed all four gates: see
  `runs/RUNLOG.md`.
- The observation is `Hive::RelativeObs` (109 floats at 1v1, up from 89): the
  old absolute layout plus a car-frame relative block per body — `dirToBall` as
  a unit vector, distance, offset, and closing velocity. `TrainConfig::obs`
  selects it, `ObsMode::Default` restores the p8ref layout, and it is a parity
  item (`HIVE_OBS_DEFAULT`). Team-invariance and padding are asserted in
  `bot/tests/test_relativeobs.cpp`.
- Training runs on GPU end to end at 1v1; observation size 89 at
  `maxPlayersPerTeam = 1`.
- Throughput at 128 games (the measured optimum; see `runs/RUNLOG.md`),
  512-wide network. **Two different numbers, and the difference matters when
  you budget a run:** the `throughput-*` sweep measured ~81k steps/sec with no
  skill tracking, but a real labelled run with `--track-skill` gets ~52k
  (`main-p5goalpot.csv`: 100.2M steps, final `Overall Steps/Second` 53,990,
  mean 51,949). Skill tracking's 8 evaluation arenas compete with training for
  CPU on a 6-core machine. **Plan runs with ~52k**, or a 100M-step run looks
  like 21 minutes when it is 31.
- `HiveTests` passes; `verify` and `eval` subcommands work against smoke
  checkpoints.

## Not yet verified

The RLBot v5 client compiles but has never connected to a live match. There is
no `rlbot` CLI: v5's Python package (`pip install --user --pre rlbot`,
installed) is a library, and matches are started with `scripts/run_match.py`,
which drives `rlbot.managers.MatchManager` plus the RLBotServer binary in
`libs/rlbot/` (downloaded from RLBot/core releases; gitignored).

## Parity traps

Training and deployment must agree on `maxPlayersPerTeam`, `tickSkip`,
`actionDelay`, `maskActions`, `obs` and `ModelShape` (the last defined in
`bot/src/policy/Policy.h`, default-constructed by both sides). Action parsers
are built only by `Hive::MakeActionParser` (`bot/src/env/Actions.h`) so the
mask setting cannot drift between the five places that need one.

A mismatch does not crash — the bot loads, plays, and is quietly worse.
Training values live in `bot/src/Config.h`; deployment reads `HIVE_*`
environment variables set by
`bot/rlbot-config/run.sh`. **`./HivemindBot verify <checkpoint>` mechanizes
the check** — run it before every deployment session.

## Conventions

- Comments explain *why*, especially where a choice looks arbitrary or where a
  bug would be silent. The packet conversion carries the most of this.
- `Hive::` namespace for our code; `RLGC::` is RLGymCPP, `GGL::` is GigaLearn.
- Tabs, matching the surrounding GigaLearn/RLGymCPP style.
- Every run gets a `--label`; record runs that matter in `runs/RUNLOG.md`.

### Second local patch to external/

`external/GigaLearnCPP-Leak/.../Util/KeyPressDetector.cpp` is guarded with
`isatty(0)`.

Upstream, when stdin is not a terminal — any backgrounded or redirected run —
`tcgetattr`/`tcsetattr` fail and `read()` returns EOF immediately. The caller
loops on it forever, so it busy-spins a full CPU core for the whole run and
emits three `perror` lines per iteration. One 50M-step run wrote an 8.1 GB log.
The patch parks the thread instead; there is no interactive 'Q' to detect
without a terminal anyway. Reapply if you re-clone.

### Third local patch to external/

`external/GigaLearnCPP-Leak/.../PPO/PPOLearner.cpp` standardizes GAE advantages
per minibatch before the clipped objective:

```cpp
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
```

Upstream feeds raw advantages in, so the policy step scales with their absolute
magnitude — and that magnitude shrinks as the critic improves, because a better
critic means smaller TD residuals. The effective step size therefore *decays*
over a run. Measured on `p1-validate` (117M steps): `GAE/Avg Advantage` 0.151 →
0.088, `Mean KL` 1.25e-3 → 6.9e-4, `SB3 Clip Fraction` 6.1e-3 → 4.1e-3, against
a healthy PPO clip fraction of roughly 0.05–0.2. This is what "learning stops at
~40M" was: not a reward plateau, an update size decaying to nothing.

With the patch (`p1probe-j`), clip fraction rose ~4x and entropy finally fell
under its own steam (0.683 → 0.533 in 10M steps). **It does not on its own make
the bot better** — it makes the policy converge faster on whatever the reward
actually rewards, which is a separate problem. Reapply if you re-clone.

### Fifth and sixth local patches: NaN containment

Both carried in `scripts/apply_external_patches.py`, so a re-clone cannot drop
them.

**`PPOLearner.cpp` skips an optimizer step whose gradients are non-finite.**
`clip_grad_norm_` does NOT protect against this and actively makes it worse: it
computes a total norm and rescales by `max_norm/norm`, so one NaN gradient gives
a NaN norm, a NaN scale, and NaN in **every** parameter. Clipping converts a
single bad gradient into a fully destroyed network.

**`GAE.cpp` guards `returnStd` with `std::isfinite`.** The original test was
`returnStd != 0`, which is TRUE for NaN, so a NaN standardizer would silently
turn every reward in the batch into NaN.

p11boost died at 29.8M steps with `Policy Update Magnitude: nan` and then a CUDA
device-side assert (`probability tensor contains either inf, nan or element
< 0`) when the poisoned policy was next sampled. Everything upstream was
healthy: entropy 0.538, KL 0.0053, reward 0.0624, `GAE/Returns STD` 2.113,
`Critic/V All` 2.72. The only GAE outputs that went NaN were
`GAE/Avg Advantage` and `GAE/Avg Val Target` -- exactly the two that depend on
critic value predictions -- so a non-finite value entered through inference,
not through the reward function.

`Obs/Non-Finite Rate` was added at the same time and is **zero in every healthy
run**. If it is ever non-zero, the observation is the source; if this recurs
while it stays zero, the critic is. Non-finite observation values are replaced
with zero so the run survives either way.

### Fourth local patch to external/

`external/GigaLearnCPP-Leak/.../PPO/PPOLearner.cpp` mixes the policy with a
uniform distribution over the *valid* actions before sampling
(`ACTION_EXPLORE_EPS = 0.02` in `InferPolicyProbsFromModels`).

This project has lost three control dimensions to extinction: jump on p1air
(`Action/Jump When Grounded Upright` 0.49 -> 0.0000 by ~20M steps, and the
p2 probes then proved curriculum, entropy and reward changes are all inert
against it), then steer and throttle on p3strike (`Action/Steer Nonzero`
0.0006 at 100M). PPO's gradient is proportional to the probability of the
action taken, so once an action reaches ~0 probability it gets no gradient and
cannot recover, whatever the reward says.

The entropy bonus does not substitute: p2entropy raised `entropyScale` 2.5x and
moved measured entropy 0.0665 -> 0.0659. Entropy describes the whole
distribution and says nothing about whether one particular action retains
support. The pre-existing `ACTION_MIN_PROB` clamp cannot serve either, since it
is applied after masking and raising it would give *disabled* actions sampling
probability.

Verified live: resuming the extinct p3strike policy with the patch moved
`Action/Steer Nonzero` 0.0005 -> 0.0082 and Policy Entropy 0.155 -> 0.224 with
no retraining. Applied inside `InferPolicyProbsFromModels` so that collection,
the update's log-probs and the KL computations all use the identical
distribution, which PPO's importance ratio requires.

**This one is reapplied automatically.** `scripts/apply_external_patches.py`
carries the patch body in-tree and `scripts/build.sh` runs it before
configuring, so a re-clone of `external/` cannot silently drop it. The script is
idempotent, and `--check` reports status without writing. The other three
patches above are still manual; they can be moved into the same script by
adding an entry to its `PATCHES` list.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
