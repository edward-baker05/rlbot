# Hivemind

A Rocket League bot trained with [GigaLearn](https://github.com/ZealanL) on
RocketSim, deployed through [RLBot v5](https://wiki.rlbot.org/v5/).

One policy, trained on 1v1, kickoffs included via curriculum. See
[docs/architecture.md](docs/architecture.md) for the design and
[the roadmap spec](docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md)
for where it is going.

---

## Layout

```
bot/                    Our code
  src/
    Config.h            Every tunable, in one place
    main.cpp            Entry point: train | play | verify | eval
    Verify.*            Checkpoint/deployment parity check
    env/                Obs, rewards, state setters, curriculum, play phases
    eval/               Headless checkpoint-vs-checkpoint matches
    policy/             Policy: single-model inference wrapper, ModelShape
    rlbot/              RLBot v5 hivemind client and packet conversion
    train/              Training loop and metrics
  tests/                doctest suite (HiveTests)
  rlbot-config/         bot.toml, loadouts, match configs
  build/                Build output (gitignored)
external/
  GigaLearnCPP-Leak/    GigaLearn + RLGymCPP + RocketSim
  cpp-interface/        RLBot v5 C++ interface
  RocketSimVis/         Live viewer
libs/                   libtorch, NCCL, NVSHMEM (gitignored; scripts/setup_libs.sh)
tools/                  Collision mesh dumper and the meshes it produced
scripts/                build / train / watch / vis / play / match / throughput / sync
checkpoints/            Latest checkpoint of each test, alongside CONFIG.json and CONFIG_HISTORY.json
runs/                   RUNLOG.md: one line per run that matters
docs/                   Architecture, metrics guide, specs and plans
```

---

## Quickstart

```bash
scripts/build.sh              # First build takes a while; later ones are fast
cd bot/build && ./HiveTests   # Run the test suite
scripts/train.sh              # Start training. Press Q to save and quit.
```

Watch what it is doing:

```bash
scripts/vis.sh                # Terminal 1: start RocketSimVis
scripts/watch.sh              # Terminal 2: stream one game at real time
```

Before deploying a checkpoint, and to compare two of them:

```bash
./HivemindBot verify checkpoints/main/50000000
./HivemindBot eval --blue <ckpt A> --orange <ckpt B> --games 20
```

Play against it once you have a checkpoint (needs `pip install --user --pre
rlbot` and the RLBotServer binary — `scripts/run_match.py` prints the
download command if it is missing):

```bash
scripts/play.sh                       # 1v1, you vs the bot
scripts/match.sh /path/to/opponent/bot.toml   # bot vs an RLBotPack bot
```

---

## Measured on this machine

Ryzen 3600 + RTX 2060. Current (1v1, 89-wide obs, 512-wide network, default
skill tracking), measured 2026-08-17 with `scripts/throughput.sh`:

| --games | Steps/sec |
|---:|---:|
| 64 | ~71,000 |
| **128** | **~81,000** |
| 192 | ~79,000 |
| 256 | ~78,000 |
| 320 | ~75,000 |

128 games wins; it is the default. Pre-1v1 figures (multi-size obs, up to 6
cars per arena, hence more player-steps per sim step): ~120k/s at 256-wide,
~91k/s at 512-wide, ~1.9 GB VRAM.

---

## Where the strength comes from

Ordered by how much they matter, most first:

1. **Total timesteps.** Nothing substitutes for this. Nexto-level play is a
   billions-of-steps proposition.
2. **A reward function that cannot be farmed.** An early version paid 98.4% of
   its reward for shaping and the policy's skill rating fell for 20M steps
   while its reward curve climbed. `RewardShare/*` metrics exist so this is
   caught in minutes, not weeks.
3. **Curriculum weights** (`CurriculumWeights` in `bot/src/Config.h`). Rare
   skills need to be spawned deliberately — and the curriculum must match what
   the reward actually pays for.
4. **Network size.** There is VRAM spare; widening is a phase-gate decision.

**Never judge a run by its reward curve.** It rises in every run, including
broken ones. See [docs/metrics.md](docs/metrics.md) for what to watch instead.

---

## Status

- Builds clean, trains on GPU end to end at 1v1; test suite passes.
- The RLBot v5 client compiles and its packet conversion is covered by
  round-trip tests, but it has **not** been run against a live match yet.
  Note there is no `rlbot` CLI: v5's Python package is a library, so matches
  are started by `scripts/run_match.py` (which drives `MatchManager` and the
  RLBotServer binary in `libs/rlbot/`).

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — the single-policy design,
  observation, parity, and the deployment path
- [docs/metrics.md](docs/metrics.md) — every metric: what it measures, what
  healthy looks like, which decision it feeds
- [Roadmap spec](docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md)
  — design decisions D1–D8, reward phases P1–P4, evaluation layers, timeline
- [runs/RUNLOG.md](runs/RUNLOG.md) — the run log; comparisons are only valid
  between runs recorded there
