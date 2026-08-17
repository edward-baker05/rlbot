# Tuning

Everything here lives in `bot/src/Config.h`. Rebuild after changing it.

## Measured baseline

Ryzen 3600 (6c/12t), RTX 2060 (6 GB), defaults:

| Games | Overall steps/sec | Collection steps/sec | Notes |
|---:|---:|---:|---|
| 8 | ~24,000 | ~25,000 | Inference-bound; GPU barely used |
| 128 | ~120,000 | ~164,000 | Default |

VRAM at 128 games: **~1.9 GB of 6 GB**.

The jump from 8 to 128 games is a 5x throughput gain, almost entirely from
amortising GPU inference across more cars per batch. There is little reason to
run fewer than ~64 games.

---

## What to change first

### 1. Network size — you have the headroom

`modelShape` in `Config.h`. The defaults are conservative:

```cpp
sharedHeadLayers = {256, 256};
policyLayers     = {256, 256, 256};
```

That is ~530k parameters total. With 4 GB of VRAM spare, doubling to `{512,
512}` / `{512, 512, 512}` is comfortable and raises the skill ceiling. Bots at
the level you are targeting are generally larger than 256-wide.

The cost is throughput — expect a meaningful drop in steps/sec. Whether that
trade is worth it depends on where you are: early on, raw timesteps win; once
the policy plateaus, capacity is usually the binding constraint.

> Changing layer sizes invalidates existing checkpoints. Start a fresh run, and
> update `ModelShape` in `bot/src/rlbot/HivemindBot.h` to match or the deployed
> bot will load garbage.

### 2. `numGames` — the CPU knob

Default 128. Raise it while collection steps/sec keeps climbing; stop when it
plateaus or you run short of RAM. Each game holds a full RocketSim arena, so
memory scales roughly linearly.

With 32 GB you have room to try 192 or 256. Watch `Collection Steps/Second` in
the iteration output — if it stops improving, the CPU is saturated and more
games only cost memory.

### 3. `miniBatchSize` — the VRAM knob

Default 25,000. At 1.9 GB used you can raise this, which slightly improves
gradient quality per step. If you ever hit CUDA OOM, halve it — and note that
OOM often appears several iterations in rather than immediately, so a run that
started fine is not proof the setting is safe.

Dropping to 12,500 is the standard fix if you scale the network up and run out.

---

## PPO settings

| Setting | Default | Notes |
|---|---:|---|
| `tsPerItr` | 50,000 | Steps per iteration. Larger is more stable, slower to react. |
| `epochs` | 1 | GigaLearn's author finds 1–2 near-optimal for time-to-skill. Try 2. |
| `entropyScale` | 0.035 | Raise if `Policy Entropy` collapses early; lower if the bot stays random. Scale-invariant to action count, unlike `ent_coef` elsewhere. |
| `gaeGamma` | 0.99 | Reward decay. Starting low and raising later is a known-good pattern. |
| `policyLR` / `criticLR` | 1.5e-4 | Lower them as the run matures; high LR late in training destabilises a good policy. |

---

## Team size mix

`TeamSizeMix`, default 30/30/30/10 across 1v1, 2v2, 3v3 and asymmetric.

Sizes are assigned deterministically by game index, so the realised mix is
exactly the configured ratio every run rather than merely converging to it. That
makes run-to-run comparisons honest.

Skew this towards the format you actually care about. If you only ever play 1s,
weighting `weight1v1` higher will get you there faster — but keep some 2s/3s so
the policy does not forget how to handle a crowded field, and keep
`weightAsymmetric` non-zero so uneven teams (a human joining or leaving) are not
off-distribution.

---

## Curriculum weights

`CurriculumWeights`. These are relative, not percentages.

The one rule worth stating: **keep `neutralPlay` dominant.** It is the situation
the bot is in most of the time. Over-weighting exotic scenarios produces a bot
that can air dribble but cannot rotate, which loses to a bot that only rotates.

Raise a specific weight when the corresponding `Phase/*` metric shows the bot
spending less time there than you want. Verify the setter actually works
(`scripts/watch.sh`) before concluding a weight is too low.

---

## GPU vs CPU at deployment

`run.sh` defaults deployment to **CPU** (`HIVE_USE_GPU=0`). Inference for at
most 3 cars is tiny, and the GPU is busy rendering Rocket League. Using it for
inference costs frames and can add latency spikes.

Training is the opposite — always use the GPU there.
