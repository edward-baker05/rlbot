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

**Self-play against older versions.** GigaLearn has `trainAgainstOldVersions`
and an ELO-based skill tracker (`savePolicyVersions`, `SkillTrackerConfig`).
Neither is wired into `Config.h` yet. This is the standard way to avoid the
policy overfitting to its current self, and is probably the biggest structural
improvement available after network size.

**Replay-based state setters.** Setting states from real human replays gives a
far more realistic distribution than hand-written setters. Tools exist in the
RLGym community (Rolv-Arild's replay work is the usual starting point). This is
more work than the others but is how several strong bots got their edge.

**Reward shaping.** Listed last on purpose. It is the most tempting knob and the
easiest way to make things quietly worse.

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
