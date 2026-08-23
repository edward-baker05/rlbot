# 11 - Opponent diversity: the plateau may be a self-play fixed point

Status: open
Priority: this is the leading hypothesis after p18entropy

## Why this jumped the queue

p17 and p18entropy together rule out the two obvious explanations for a policy
that will not move. It is not a broken learner (clip 0.044, KL 0.0045, update
magnitude 0.069, all healthy). It is not the entropy controller (removed; every
PPO health metric improved; behaviour still flat).

What remains is that there is nothing left to learn **against this opponent**.

## The mechanism

- `trainAgainstOldChance = 0.2`, so 80% of training is against a **current copy
  of the same network**.
- The old-version pool is `maxOldVersions = 32` at `tsPerVersion = 5M`, i.e. a
  ~160M-step window of policies nearly identical to the current one.
- Roughly **46% of reward mass is in zero-sum terms** (`TouchGoalAccel` 0.31 at
  k=0.8, `ShotOnTarget` 0.11 at k=1.0, `Save` 0.044 at k=1.0). Against a copy of
  yourself, a purely differential term has zero expected advantage by
  construction.
- `Rating/1v1` is computed against that same pool, so the measurement cannot
  detect the problem: a saturated policy and an improving one both show a
  rating oscillating about a fixed point.

## Things to try, cheapest first

1. **Raise `trainAgainstOldChance`** well above 0.2 and **widen the pool**
   (larger `tsPerVersion`, or keep a few permanent early snapshots) so the
   opponent distribution spans a real skill range rather than 160M steps of
   near-clones.
2. **External-opponent Elo.** Already named in CLAUDE.md's phase A and never
   built. Until it exists, no rating number in this project is anchored to
   anything outside itself.
3. Only then reconsider reward work.

## What this does NOT license

Do not conclude the reward stack is fine. It is untested, which is different.
Items 01, 02, 07, 08 and 10 stay open -- they simply cannot be evaluated until
there is an opponent whose exploitation produces a gradient.
