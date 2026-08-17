# CLAUDE.md

Context for working in this repo.

## What this is

A Rocket League bot: trained with GigaLearn (C++) on RocketSim, deployed through
RLBot v5. Target is a bot that beats a GC1 human.

**Two policies, not a mixture of experts.** A kickoff policy and a general
policy, split at first ball touch. An MoE design was considered and rejected;
`docs/architecture.md` explains why, and that reasoning should not be quietly
undone. Situation labels survive only as metrics and as curriculum weights.

## Layout

- `bot/src/` — the only code that is ours. Everything else is third-party.
- `external/` — GigaLearnCPP-Leak, cpp-interface (RLBot v5), RocketSimVis. Each
  has its own git history; do not commit changes here without noting them.
- `libs/` — libtorch, NCCL, NVSHMEM. Gitignored; `scripts/setup_libs.sh`
  reinstalls them.

## Build

```bash
scripts/build.sh
```

Three things about the build that are easy to trip over:

1. **PIC is required globally.** GigaLearnCPP is a shared library that links
   RLGymCPP and RocketSim statically, which only links on x86-64 if those are
   position-independent. `CMAKE_POSITION_INDEPENDENT_CODE ON` in
   `bot/CMakeLists.txt` handles it.

2. **NCCL and NVSHMEM must be linked explicitly.** `libtorch_cuda.so`
   references their symbols even for single-GPU use. They ship as versioned
   files (`libnccl.so.2`) with no unversioned symlink, so `-lnccl` does not
   resolve — full paths are linked instead.

3. **`CMAKE_POLICY_VERSION_MINIMUM` is pinned to 3.10.** Upstream declares
   minimums that CMake 4 rejects. Pinned in our CMakeLists rather than by
   editing `external/`, so `git pull` there stays clean.

### Local patch to external/

`external/cpp-interface/library/Client.cpp` has an added `#include <climits>`.
GCC 16 no longer pulls it in transitively and `CHAR_BIT` fails to resolve. If
you re-clone cpp-interface, reapply it.

## Verified working

- Both training targets run on GPU end to end.
- Observation size 165 at `maxPlayersPerTeam = 3`.
- ~120k steps/sec at 128 games; ~1.9 GB of 6 GB VRAM.

## Not yet verified

The RLBot v5 client compiles but has never connected to a live match. Needs the
`rlbot` CLI (`pipx install rlbot`, not currently installed) and a trained
checkpoint.

## Parity traps

Training and deployment must agree on `maxPlayersPerTeam`, `tickSkip`,
`actionDelay`, and the model layer shapes. A mismatch does not crash — the bot
loads, plays, and is quietly worse. Training values live in `bot/src/Config.h`;
deployment reads `HIVE_*` environment variables set by
`bot/rlbot-config/run.sh`, except `ModelShape`, which is compiled into
`bot/src/rlbot/HivemindBot.h`.

## Conventions

- Comments explain *why*, especially where a choice looks arbitrary or where a
  bug would be silent. The packet conversion and the regime split carry the most
  of this.
- `Hive::` namespace for our code; `RLGC::` is RLGymCPP, `GGL::` is GigaLearn.
- Tabs, matching the surrounding GigaLearn/RLGymCPP style.

### Second local patch to external/

`external/GigaLearnCPP-Leak/.../Util/KeyPressDetector.cpp` is guarded with
`isatty(0)`.

Upstream, when stdin is not a terminal — any backgrounded or redirected run —
`tcgetattr`/`tcsetattr` fail and `read()` returns EOF immediately. The caller
loops on it forever, so it busy-spins a full CPU core for the whole run and
emits three `perror` lines per iteration. One 50M-step run wrote an 8.1 GB log.
The patch parks the thread instead; there is no interactive 'Q' to detect
without a terminal anyway. Reapply if you re-clone.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
