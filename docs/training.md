# Training plan

Goal: a bot that beats you at GC1. That is roughly the Nexto benchmark, and it
is a realistic target on this hardware given enough wall clock.

## The honest headline

**Total timesteps dominate everything else.** Reward shaping, curriculum weights
and network size all matter, but they matter as multipliers on a number that has
to get large. At ~400M steps/hour (128 games, measured), the rough scale is:

| Timesteps | Wall clock | Roughly |
|---:|---:|---|
| 100M | ~15 min | Chases and hits the ball |
| 500M | ~1.5 hr | Scores on purpose, some aerials |
| 2B | ~5 hr | Coherent play, recognisable rotations |
| 10B | ~1 day | Plat/Diamond-ish |
| 50B+ | ~5 days | Champ and upward |

These are order-of-magnitude, not promises. The published bots at this level
trained for far longer than a single desktop run, usually on much more compute.
Getting to GC1 on a 3600 + 2060 is a matter of weeks of accumulated training,
not an afternoon — but it is reachable, and checkpoints let you stop and resume
freely.

## Order of work

### 1. General policy first

The kickoff model is worthless until the general model can play, because a
kickoff hands over to it two seconds in. Train `general` and leave `kickoff`
alone at the start — with no kickoff model loaded the bot uses the general
policy for kickoffs, which is the correct fallback.

```bash
scripts/train.sh general
```

Let it run. Press `Q` to save and quit cleanly; killing the process loses
progress since the last checkpoint (every 1M steps by default).

### 2. Sanity-check early, with your eyes

At around 20–50M steps, look at it:

```bash
scripts/vis.sh          # terminal 1
scripts/watch.sh general # terminal 2
```

You are checking that the *scenarios* are sane, not that the bot is good. Things
that show up here and nowhere else:

- Cars spawning inside the ball in the air dribble setter
- The flip reset setter putting the car somewhere unreachable
- Cars spawning inside each other in the demo setter
- Anything falling through the floor (means collision meshes are wrong)

Fix these before spending days of compute on them.

### 3. Watch the metrics that matter

Reward goes up in every run, including bad ones. These are more informative:

| Metric | What it tells you |
|---|---|
| `Player/Touch Height` | Whether an air game is developing. The single best early indicator. |
| `Phase/*` | What the policy actually spends time doing. Compare against your curriculum weights. |
| `Player/Ball Touch Ratio` | Whether it can reliably reach the ball at all. |
| `Policy Entropy` | Collapsing towards zero means it has stopped exploring — usually too little `entropyScale`. |
| `Game/Goal Speed` | Whether goals are deliberate strikes or slow dribbles over the line. |

A classic failure: reward climbing steadily while `Player/Touch Height` falls.
The bot has found that ground touches are safer and has quietly abandoned
aerials. Raise `rewards.touchHeight` or the aerial curriculum weight.

### 4. Kickoff policy, later

Once the general policy plays competently, train the kickoff model. It is a much
smaller problem — short episodes, narrow reward set — and converges quickly.

```bash
scripts/train.sh kickoff
```

Then point `HIVE_KICKOFF_MODEL` at the checkpoint (or just let `run.sh` find it
automatically) and it will take over kickoffs.

### 5. Iterate on the curriculum

Once you have a baseline, change **one thing at a time** and compare metrics
against the previous run. Use `--seed` to make runs comparable; the random seed
genuinely affects outcomes, so a difference smaller than seed-to-seed variance
is not a result.

---

## Things worth trying, roughly in order of expected value

**Scale the network.** You are using 1.9 GB of 6 GB VRAM. Doubling the layer
widths is the most straightforward way to raise the skill ceiling. See
[tuning.md](tuning.md).

**More timesteps.** Boring, and still the highest-value item on the list.

**Curriculum rebalancing.** If `Phase/Aerial` sits far below where you want the
bot's air game, raise `curriculum.aerial`. Keep `neutralPlay` dominant — a bot
that can air dribble but cannot rotate loses to one that only rotates.

**Self-play against older versions.** Wired up — see the section below.

**Replay-based state setters.** Setting states from real human replays gives a
far more realistic distribution than hand-written setters. Tools exist in the
RLGym community (Rolv-Arild's replay work is the usual starting point). This is
more work than the others but is how several strong bots got their edge.

**Reward phases.** The reward function is staged — see
[rewards.md](rewards.md). Advancing a phase introduces a new skill once the
previous one has consolidated. Do not advance early; a bot given dribble rewards
before it can strike the ball just gets more ways to earn without improving.

**Ad-hoc reward tweaking.** Listed last on purpose. It is the most tempting knob
and the easiest way to make things quietly worse. If you do change a weight,
decompose the resulting run (`Rewards/*` columns x weights) and check the farm
ceiling before trusting the reward curve.

---

## Self-play

By default the policy only plays against its current self. That works, but both
sides co-adapt: the policy can drift towards strategies that beat *itself right
now* rather than strategies that are actually strong, and a weakness neither
side exploits never gets punished.

Training against saved older versions fixes that. The current weights are
snapshotted periodically, and a fraction of games are played against a randomly
chosen snapshot. The opponent pool does not co-adapt with you, so an exploit
that only works against your current self stops paying off.

```bash
scripts/train.sh general --self-play
```

That enables both halves: training against old versions, and the skill tracker
that measures it.

### Measuring it

The skill tracker runs evaluation matches between the current policy and old
versions, maintaining an ELO-style rating per game mode (`Rating/1v1`,
`Rating/2v2`, …). This is the only honest answer to "is it getting better" —
average reward rises in every run, including ones where the policy has merely
found a better way to farm shaping rewards.

Enable it without self-play to get a comparable baseline:

```bash
scripts/train.sh general --track-skill
```

Ratings only appear once the version pool is non-empty, i.e. after
`tsPerVersion` steps.

### Settings

`SelfPlayConfig` in `Config.h`:

| Setting | Default | Notes |
|---|---:|---|
| `trainAgainstOldChance` | 0.15 | Fraction of iterations played against a snapshot. Low on purpose — the goal is to keep the policy honest, not to stop it improving. |
| `tsPerVersion` | 5M | Snapshot interval. GigaLearn's own default is 25M, suited to multi-billion-step runs; 5M builds a usable pool fast. Raise it for long runs — a pool of near-identical recent versions is not much of a test. |
| `maxOldVersions` | 32 | Pool size. |
| `skillArenas` | 8 | Evaluation arenas. They compete with training for CPU; keep well under core count. |
| `skillUpdateInterval` | 20 | Iterations between evaluations. Higher is cheaper and coarser. |

### Comparing two configurations

```bash
scripts/compare_runs.sh 50000000 128
scripts/summarize_runs.py bot/build/metrics/general-baseline.csv \
                          bot/build/metrics/general-selfplay.csv
```

Runs a baseline and a self-play run over an identical **step budget** with the
same seed, then prints a side-by-side comparison.

Budget rather than wall clock, deliberately: self-play costs time per step, so
comparing by clock would penalise it for having seen less data rather than for
being worse. `--max-steps` enforces the budget exactly.

Use `--label` to keep runs apart — without it the second run resumes from the
first one's checkpoints and the comparison is meaningless.

### Metrics CSV

`bot/metrics/metric_receiver.py` replaces GigaLearn's wandb-only receiver and
writes every metric to `bot/build/metrics/<run>.csv`. This exists because
GigaLearn's console output prints a fixed subset that excludes `Rating/*`,
`Player/*` and `Phase/*` — the metrics you most need. wandb still works if
installed; set `WANDB_DISABLED=1` for CSV only.

---

## Resuming

Training automatically loads the most recent checkpoint in its target folder, so
re-running the same command continues where it left off:

```
bot/build/checkpoints/general/   <- general policy, numbered by timestep
bot/build/checkpoints/kickoff/   <- kickoff policy
```

`checkpointsToKeep` (default 8) bounds how many are retained. Copy one somewhere
safe before a risky config change — it is the only way back to a known-good
policy.
