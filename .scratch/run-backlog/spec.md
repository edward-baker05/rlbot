# Run backlog: everything deferred, and what blocks what

Compiled 2026-08-21, at the point p17 finished and `p18` was allocated to the
entropy discriminator. Every item here was raised, argued and deliberately
deferred during p17's design. Nothing in this file is a new idea.

> **SUPERSEDED 2026-08-23 by `PLAN-post-p18.md`.** The tickets below are still
> authoritative for their own content. The *ordering* is not: p18entropy ran to
> 483M and answered its question, so the "blocked by p18entropy" column is
> stale. Read `PLAN-post-p18.md` first.

## The structural fact that orders the whole list

**p17 proved a converged policy stays converged.** Over 300M steps the largest
behavioural change in the entire metric set was `Flip/Neutral Share`
0.009 -> 0.020; `Shot/Distance` moved 0.0%. PPO was in perfect steady state
throughout (clip 0.0354, KL 0.0038, update magnitude 0.0696, constant to three
significant figures).

So **every reward item below is blocked on `p18entropy`.** If the policy cannot
move, a reward experiment measures nothing, and p13-p17 would all need
re-reading. Do not spend a run on a reward term until p18entropy says the
policy can respond to one.

Items 03, 04, 05, 09 and 11 are NOT blocked: they are action-space, state
distribution, budget-hygiene and instrumentation, none of which depend on the
answer.

## Index

| # | Item | Blocked by | Why it was deferred |
|---|---|---|---|
| 01 | Possession / next-touch term | p18entropy, 06 | The operator's original request, re-keyed to an outcome. Needs horizon |
| 02 | `touchGoalAccelOpponentScale` 0.8 -> 1.0 | p18entropy | Same `S(1-k)` leak p17 closed on ShotOnTarget; left alone as load-bearing |
| 03 | Air game: double the jump actions | — | Zealan's own remedy, never tried in this project |
| 04 | Air game: run the `aerial` spawn scenario | — | The scenario exists and `SpawnMode::Random` never runs it |
| 05 | `airTouch` breaks the finishing-block guard | — | Cutting it is a retarget; would have confounded p17 |
| 06 | `gaeGamma` 0.99 -> 0.995 | p18entropy | Changing gamma invalidates an inherited critic everywhere at once |
| 07 | Re-test `SaveReward` | p18entropy | Unfalsified, not falsified. It was never given a policy that could respond |
| 08 | Is zero-sum at exactly k=1 self-defeating? | p18entropy, 07 | The p4pbrs lesson says the no-farm guarantee and the teaching signal are one property |
| 09 | Positioning / rotation instruments | — | The metric set covers none of it, and that is where the operator's eye saw change |
| 10 | Rebound / multi-save pressure | p18entropy, 06 | Spans 150-225 steps; invisible at gamma 0.99 |
| 11 | **Opponent diversity / self-play fixed point** | — | **Leading hypothesis after p18entropy. Start here** |

## Minor, not worth their own tickets

- **`Shot/Resolved Rate` should be published** next to `Shot/Saved Share`. The
  denominator is resolved on-target shots only -- a shot the shooter recovers
  itself silently leaves the denominator. Publishing the denominator beside a
  conditional is the standing p10touch lesson and this one currently breaks it.
- **`ShotOnTarget = 32` is a ceiling, not a choice.** It is the largest value
  that keeps a shot on target paying less per event than a strong touch. If
  `touchGoalAccel` changes (item 02), re-derive it.
- **Disk**: `main-p17cal` and `main-p18entropy` each carry a 143 MB copy of
  `policy_versions`. Safe to delete for finished probe labels.
- **`ASSUMED_SHOT_STRENGTH` is now measured** (0.5247, p17cal). If the policy's
  striking changes materially, re-measure it -- every "stay below finishing"
  guard scales with it.
