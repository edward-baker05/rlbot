# CLAUDE.md

Context for working in this repo.

## What this is

A Rocket League bot: trained with GigaLearn (C++) on RocketSim, deployed through
RLBot v5. Target is a bot that beats a GC1 human in 1v1.

**One policy.** The old kickoff/general two-policy split was removed
deliberately on 2026-08-17 (see
`docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`, decision D1);
kickoffs are learned via a curriculum entry. Do not reintroduce a policy split
or an MoE design without a new decision. Situation labels (`PlayPhase`)
survive only as metrics; curriculum scenarios (`CurriculumState`) only as
spawn distributions.

Two more standing decisions from that spec worth knowing before touching
rewards: no dribble/possession reward terms, ever (D4 — the flick-bot local
optimum); and no magic numbers without measurement behind them (D6).

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

## Verified working (2026-08-17)

- Training runs on GPU end to end at 1v1; observation size 89 at
  `maxPlayersPerTeam = 1`.
- ~81k steps/sec at 128 games (the measured optimum; see `runs/RUNLOG.md`),
  512-wide network, default skill tracking.
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
`actionDelay`, and `ModelShape` (defined in `bot/src/policy/Policy.h`,
default-constructed by both sides). A mismatch does not crash — the bot loads,
plays, and is quietly worse. Training values live in `bot/src/Config.h`;
deployment reads `HIVE_*` environment variables set by
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
distribution, which PPO's importance ratio requires. Reapply if you re-clone.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
