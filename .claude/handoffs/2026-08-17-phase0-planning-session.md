# Session narrative — Phase 0 planning (2026-08-17)

## What this session did

Brainstormed, specced, and planned the rebuild of this Rocket League bot from
a two-policy multi-size design into a lean single-policy 1v1 bot. No
implementation was started. Two documents were produced and committed on
`master`:

- **Spec:** `docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`
  (commit 525233e) — eight design decisions (D1–D8), the four reward phases
  P1–P4, five evaluation layers, the Phase 0 → P4 timeline.
- **Plan:** `docs/superpowers/plans/2026-08-17-phase0-single-policy-rebuild.md`
  (commit cc05168) — 13 tasks implementing Phase 0 only. This is the document
  the next session executes.

Read both before doing anything; the plan argues from the spec.

## Decisions made by the user (do not re-litigate)

- Approach A chosen explicitly over alternatives: single policy, kickoff via
  curriculum, 1v1-native observation (`maxPlayersPerTeam = 1`). Multi-mode
  (2s/3s) is a possible *future separate design*, not a constraint here.
- The uncommitted working tree (deleted `docs/`, trimmed `Config.h` /
  `Rewards.cpp`) is the **deliberate baseline** — the user stripped
  AI-written content on purpose. Plan Task 1 Step 1 commits it as-is.
- All inherited code comments are untrusted (written by a previous AI);
  delete or rewrite as files are touched. New docs are welcome.
- No dribble/possession rewards, ever (user's domain knowledge: the
  flick-bot local optimum). Air game starts at P2, before ground comfort.
- No magic numbers without measurement; gates are trends/comparisons, never
  absolute thresholds.
- Execution mode chosen: **inline execution** (superpowers:executing-plans),
  fresh-context session.

## User context

- GC1 player; goal is a bot that beats them in 1v1, aiming high-GC/SSL.
  First serious ML bot — it's a learning project too.
- Hardware: Ryzen 3600 + RTX 2060. Next ~month: part-time short runs only
  (home electricity costs). After that (back at uni): 24/7 fine. Rented GPU
  possible later, gated on trajectory.
- They frequently watch the bot in RocketSimVis and want that treated as a
  real evaluation instrument (it already caught a backwards-driving bug that
  led to the faceBall reward).

## API recon done this session (verified against sources, saves re-derivation)

- `GGL::Learner` exposes public `envSet` (`RLGC::EnvSet*`) and
  `totalTimesteps`. Learner loop order per step: reset-terminal-arenas →
  step → stepCallback — so in the callback, terminal arenas are NOT yet
  reset (plan Task 7 depends on this for scenario attribution).
- `LearnerConfig.addRewardsToMetrics` defaults true → `envSet->state.
  lastRewards[arena][term]` is populated (unweighted, pre-zero-sum, one
  sampled player per arena).
- `RLGC::Reward::GetName()` returns near-mangled typeid names on GCC — this
  is why the plan introduces `RewardSpec` with explicit names.
- RLBot cpp-interface generated headers (in `bot/build/rlbot-cpp/
  misc_generated.h`) include the flatbuffers **object API** (`GamePacketT`,
  `PlayerInfoT`, etc.); `Physics` struct ctor order is
  `(location, rotation, velocity, angular_velocity)`.
- RocketSim goal callback:
  `typedef std::function<void(Arena*, Team scoringTeam, void*)>
  GoalScoreEventFn; Arena::SetGoalScoreCallback(fn, void* userInfo)`.

## Known caveats for the executor

- Plan Task 13 (live RLBot match + Nexto v4-bridge probe) **requires the
  user present** with Rocket League running. Everything before it is
  agent-executable.
- The repo's default branch is `main` per tooling hints but the working
  branch is `master` with all commits; don't "fix" this unprompted.
- CLAUDE.md still describes the two-policy architecture — it becomes stale
  the moment Task 3 lands and is rewritten in Task 12; in between, the plan
  and spec override it where they conflict.
- Build quirks (PIC, NCCL full paths, two `external/` patches) are real and
  documented in CLAUDE.md; never modify `external/`.
