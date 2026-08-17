# Hivemind

A Rocket League bot trained with [GigaLearn](https://github.com/ZealanL) on
RocketSim, deployed through [RLBot v5](https://wiki.rlbot.org/v5/).

Two policies, one shared observation space:

| Policy | Covers | Trained by |
|---|---|---|
| **Kickoff** | Kickoff reset until first ball touch | `train-kickoff` |
| **General** | Everything else | `train-general` |

One general policy handles 1s, 2s and 3s, on either team, with humans filling
any number of slots on either side. See
[docs/architecture.md](docs/architecture.md) for why it is two models and not
nine, and how the size-agnostic observation works.

---

## Layout

```
bot/                    Your code
  src/
    Config.h            Every tunable, in one place
    main.cpp            Entry point: train-general | train-kickoff | play
    env/                Observation, rewards, state setters, terminal conditions
    policy/             Kickoff/general split and the two-model holder
    rlbot/              RLBot v5 hivemind client and packet conversion
    train/              Training loop and metrics
  rlbot-config/         bot.toml, loadouts, match configs
  build/                Build output (gitignored)
external/
  GigaLearnCPP-Leak/    GigaLearn + RLGymCPP + RocketSim
  cpp-interface/        RLBot v5 C++ interface
  RocketSimVis/         Live viewer
libs/                   libtorch, NCCL, NVSHMEM
tools/                  Collision mesh dumper and the meshes it produced
scripts/                build / train / watch / vis / play
docs/                   Architecture, training plan, tuning
```

---

## Quickstart

```bash
scripts/build.sh              # First build takes a while; later ones are fast
scripts/train.sh general      # Start training. Press Q to save and quit.
```

Watch what it is doing:

```bash
scripts/vis.sh                # Terminal 1: start RocketSimVis
scripts/watch.sh general      # Terminal 2: stream one game at real time
```

Play against it once you have a checkpoint:

```bash
scripts/play.sh                       # 1v1, you vs the bot
scripts/play.sh match-3v3-human.toml  # 3v3, you on blue with two bot teammates
```

---

## Measured on this machine

A Ryzen 3600 and RTX 2060, at the default settings:

| Games | Steps/sec | Notes |
|---:|---:|---|
| 8 | ~24,000 | Inference-bound; too few games to amortise the GPU |
| 128 | ~120,000 | Default. ~400M steps/hour |

VRAM at 128 games sits around **1.9 GB of 6 GB**, so there is real headroom to
grow the network — see [docs/tuning.md](docs/tuning.md). The default 256-unit
layers are a starting point chosen to get a run going, not a ceiling.

---

## Where the strength comes from

Ordered by how much they matter, most first:

1. **Total timesteps.** Nothing substitutes for this. Nexto-level play is a
   billions-of-steps proposition; at 400M/hour that is days of wall clock, not
   hours.
2. **Curriculum weights** (`CurriculumWeights` in `bot/src/Config.h`). Rare
   skills need to be spawned deliberately — a flip reset essentially never
   occurs under random play.
3. **Network size.** You have VRAM spare. See tuning.
4. **Reward shaping** (`RewardWeights`). Easy to fiddle with, easy to make
   worse. Change one thing at a time and compare on the metrics, not on vibes.

Watch `Player/Touch Height` and the `Phase/*` metrics at least as closely as
reward — reward goes up in every run, including ones where the bot has quietly
stopped doing anything interesting.

---

## Status

- Builds clean, trains on GPU, both targets verified end to end.
- The RLBot v5 client compiles and its packet conversion is written against the
  v5 schema, but it has **not** been run against a live match yet — that needs
  the `rlbot` CLI installed (`pipx install rlbot`) and a trained checkpoint.
  See [docs/architecture.md](docs/architecture.md#deployment-status).

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — the two-model design, the
  observation, and how humans and variable team sizes are handled
- [docs/training.md](docs/training.md) — the plan from zero to a bot that beats
  you
- [docs/tuning.md](docs/tuning.md) — hardware knobs and what to change first
