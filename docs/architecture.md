# Architecture

One PPO policy, trained by GigaLearn on RocketSim 1v1 games, deployed through
RLBot v5. For the phase roadmap (rewards, curriculum, gates), see the spec at
`docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`.

## Single policy

Earlier versions split play between a kickoff policy and a general policy,
handing over at first ball touch. That design was removed on 2026-08-17: no
top community bot works that way, the split doubled the training/deployment
parity surface, and the handover created a discontinuity at the exact moment
strong opponents attack. Kickoffs are now learned by the one policy through a
curriculum entry (`FuzzedKickoffState`, ~8% of episode resets). A dedicated
kickoff policy can be reintroduced later if evaluation shows kickoffs are
specifically weak; nothing else depends on the choice.

## Observation

`DefaultObsPadded` built with `maxPlayersPerTeam = 1` (see `env/Obs.cpp`),
giving an 89-float observation. Team slot shuffling and team inversion are
kept, so orange plays "as blue" and humans need no special handling.

**Changing anything that alters the observation width invalidates every
existing checkpoint** — the first network layer is sized to it, and a
checkpoint saved at another width will not load. `maxPlayersPerTeam` is the
usual culprit; treat it as permanent once a run you care about starts.

## Training/deployment parity

Training (`train` subcommand, values in `bot/src/Config.h`) and deployment
(`play` subcommand, `HIVE_*` environment variables set by
`bot/rlbot-config/run.sh`) must agree on:

- `maxPlayersPerTeam` (1) — observation width
- `tickSkip` (8) / `actionDelay` (7) — action cadence; the client replays the
  trained cadence in `HivemindBot::update`
- `ModelShape` — network layer sizes, defined once in `bot/src/policy/Policy.h`
  and default-constructed by both sides

A mismatch does not crash. The bot loads, plays, and is quietly worse. Run

```
./HivemindBot verify <checkpoint-folder>
```

before every deployment session: it loads the checkpoint the way the RLBot
client does, checks deterministic inference is repeatable and state-sensitive,
and cross-checks any set `HIVE_*` variables against the compiled training
values.

## Deployment path

RLBot v5 launches `bot/rlbot-config/run.sh` (via `bot.toml`'s
`run_command_linux`), which picks the newest checkpoint under
`bot/build/checkpoints/main*/` unless `HIVE_MODEL` is set, exports the
runtime `HIVE_*` variables, and execs `HivemindBot play`. The client connects
to the RLBot server, converts each `GamePacket` to an RLGymCPP `GameState`
(`rlbot/PacketConvert.cpp`), and runs one batched inference per decision step.

`PacketConvert` is the highest-risk file in the deployment path because its
failures are silent:

1. **Boost pad order.** RLGymCPP indexes pads by its own location table;
   RLBot orders them by field info. An explicit nearest-location index map is
   built once at connect time; a mapping failure logs a warning rather than
   silently scrambling the observation.
2. **Flip availability.** The observation reads RocketSim's derived
   `HasFlipOrJump()`; RLBot reports the same fact directly via
   `dodge_timeout`. The converter sets the internal fields to whatever makes
   the derivation agree with RLBot's ground truth.

Both areas are covered by round-trip tests in `bot/tests/test_packetconvert.cpp`.

## Evaluation tooling

- `./HivemindBot eval --blue A --orange B` — headless checkpoint-vs-checkpoint
  matches in RocketSim, replaying the training cadence. The
  frozen-reference-pool tool for gate decisions.
- `scripts/match.sh <opponent bot.toml>` — a live RLBot match against an
  external bot (via `scripts/run_match.py`; requires the rlbot v5 Python
  package, the RLBotServer binary in `libs/rlbot/`, and Rocket League).
- `scripts/spectate.sh <label>` — watch a checkpoint play in RocketSimVis
  while training continues. See below.
- `scripts/watch.sh` — `train --render`: a *learner* on one arena, for
  inspecting a state setter or reward change you just wrote.
- Metrics: see `docs/metrics.md`.

### Watching a run without disturbing it

```bash
scripts/vis.sh &                        # the viewer, UDP 9273
scripts/spectate.sh p1-validate         # follows that run's newest checkpoint
```

`spectate` loads a checkpoint, plays it against itself in one arena at
wall-clock speed, and streams the gamestate. It is not a learner: no optimizer,
no experience collection, no checkpoint writes. It infers on CPU by default, so
it is safe to point at a run that is in flight.

Measured cost against the 128-game `p1-validate` run on the 3600: **0.45%** of
training throughput (65.3k vs 65.6k steps/sec idle). That number depends on one
non-obvious thing — `RunSpectate` pins `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1
before the first inference. Without it, libtorch spreads two-car inference
across all six cores and the cost is **4.4%** (62.7k steps/sec), which is enough
to matter over a long run. Passing `--gpu` skips the pinning and contends for
VRAM instead; don't, during a run.

With `--follow <label>` it re-checks the run folder between episodes and picks
up the newest *complete* checkpoint, so the bot visibly improves as training
proceeds. Completeness matters: the trainer writes a new folder every
`tsPerSave` steps while you are reading, and a half-written one would either
crash the load or, worse, load without `RUNNING_STATS.json` and play with an
unnormalized observation. `FindLatestCheckpoint` (`eval/Checkpoints.cpp`)
handles this and is covered by `bot/tests/test_checkpoints.cpp`.

| Option | Default | Notes |
|---|---|---|
| `--follow LABEL` | — | Follow a run; mutually exclusive with `--model` |
| `--model FOLDER` | — | Pin one checkpoint folder |
| `--spawns MODE` | `curriculum` | `curriculum` shows the scenario mix training actually samples; `kickoff` reads as a normal match but is only ~8% of training resets |
| `--deterministic` | off | Argmax instead of sampling. Off matches what training is exploring; on matches what deployment does |
| `--time-scale X` | `1.0` | Real time |
| `--episodes N` | until Ctrl-C | |
| `--gpu` | off | Leave it off during a run; training owns the VRAM |

Useful variants:

```bash
scripts/spectate.sh p1-validate --spawns kickoff --deterministic
scripts/spectate.sh --model bot/build/checkpoints/main-p1probe-f/30012928
```

## Code map

```
bot/src/
  Config.h        every tunable; TrainConfig defaults are the training config
  Verify.*        the verify subcommand
  env/            Obs, Rewards (RewardSpec), StateSetters, Curriculum, PlayPhase
  eval/           the eval subcommand
  policy/         Policy: one InferUnit wrapper; ModelShape lives here
  rlbot/          HivemindBot client + PacketConvert
  train/          Train loop, StepCallback metrics, Metrics helpers
bot/tests/        doctest suite (HiveTests target)
```
