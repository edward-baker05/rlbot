# Plan after p18entropy-extended (2026-08-23)

Supersedes the ordering in `spec.md`. The tickets themselves are unchanged and
still authoritative for their own content; what changed is which of them can be
*read*, and in what order.

## What the 483M run settled

`spec.md` opened with: "every reward item below is blocked on `p18entropy`. If
the policy cannot move, a reward experiment measures nothing." That block is now
resolved, and the answer is worse than either branch anticipated.

1. **The freeze is not the entropy controller, at two entropy levels.** The
   50M probe ran the controller off (entropy 0.285); the 433M extension
   accidentally ran it back on at target 0.4 (entropy 0.400). 26% below and 26%
   above p17's operating point. Segment-matched drift is identical:
   `Touch/Above 450` +10.1% vs +10.6%, `Shot/Distance` +0.2% vs -0.6%.

2. **The objective itself has stopped rising.** `Average Step Reward` +0.7% net
   over 483M against a 2.6% within-run oscillation. Split into nine ~54M chunks,
   **exactly one metric trends monotonically across all nine, and it is
   `GAE/Returns STD`** (41.62 -> 40.89, 153 sigma) -- the critic tightening, not
   the policy changing. Every behavioural metric's net change is smaller than
   its own oscillation.

3. **PPO is converged, not broken.** Clip 0.0342-0.0375, KL 0.0037-0.0040,
   update magnitude 0.068-0.072, `Obs/Non-Finite Rate` 0.000, constant to three
   significant figures across 483M steps.

4. **57.6% of the reward budget has zero expected advantage in symmetric
   self-play** (`Goal` 17.4% + `ShotOnTarget` 11.2% + `Save` 4.4% +
   0.8x`TouchGoalAccel` 24.6%), and `trainAgainstOldChance = 0.2` means 80% of
   training is against a copy of the same network. The only axis with residual
   movement is `AirTouch` -- 13.3% of mass, the largest term that is *not*
   cancelled by symmetry, and the one issue 05 says already pays 1.33x the
   entire finishing block per episode.

5. **No instrument in the repo can say whether any of this is good.**
   `Rating/1v1` is an unanchored Elo over a rolling ~160M-step pool whose new
   entrants inherit the current rating. Measured: 41 samples, per-sample sigma
   6.77, random-walk sigma over 40 updates **42.8**. The observed 424 -> 394 is
   inside chance and supports no conclusion in either direction.

**So the binding constraint is not a reward term. It is that the project is
optimising a mostly-symmetric objective against a copy of itself and measuring
the result with a ruler made of the same material.** Both halves need fixing,
and measurement comes first, because every open ticket asks a question that the
current instruments cannot answer.

## Ordering

> **RUN 0 IS DONE (2026-08-23) AND IT RESOLVED THE FORK.** p18 is **+478 Elo**
> over p12goal, so the lineage works and the "restart because it has been flat
> all along" branch is dead. The plateau is real but **starts at p16**: p17 and
> p18 together bought **+42 Elo** over ~800M steps, against p16's +273 alone.
> SELF-control read 49.3%/50.5% (truth 50%), so the noise floor is ~+-13 Elo.
> Run 1 below is unchanged and is now measurable. Two new items, both ahead of
> it: **reconstruct what p15manual and p16 changed** (they are the two biggest
> gains ever measured here and neither is in `RUNLOG.md` nor in their own
> `CONFIG_HISTORY.json`), and **audit `verify`** -- `eval` sat broken behind a
> compiled-out assert and `verify` has not been checked.

### Run 0 -- the anchored ladder. No training. Do this first.

The single highest-value action available, and it costs zero GPU-hours of
training.

`HivemindBot eval --blue A --orange B --games N --seconds S --seed K` already
exists and already takes a seed. Build `scripts/ladder.py` on top of it: play
the current policy against a **fixed, permanently retained** set of its own
ancestors, both sides swapped to cancel side bias, and report goal differential
per rung.

**Rungs available today**, verified against each checkpoint's `CONFIG.json`:

| Class | Runs | Loadable by `eval` as built |
|---|---|---|
| `obs=Relative`, `addLayerNorm=true` | p13strike, p14aerial, p15manual, **p16**, **p17**, **p18** | yes |
| `obs=Relative`, `addLayerNorm=false` | p9rel, p10touch, p11boost, **p12goal** | needs the fix below |
| `obs=Default` | p3strike - p8ref | no, and not worth it |

`eval` default-constructs `ModelShape`, so it can only load one layer-norm class
at a time. **Small fix worth making: have `eval` read `ModelShape` from the
checkpoint's own `CONFIG.json`.** That adds p12goal, which is the rung that
matters most -- `RUNLOG.md` calls it "the best bot this project has made" and it
predates the alleged plateau by 800M steps.

**Write the prediction down first.** The plausible outcomes are genuinely
different and they fork everything after:

- **p18 ~= p17 ~= p16, and all clearly beat p12goal.** The plateau is real and
  began around p16. Proceed to Run 1 as written.
- **p18 > p17 > p16 by a clear margin.** The plateau is a measurement artefact
  of the self-referential Elo, the last three runs did work, and the reward
  tickets are readable after all. Re-read p17's "clean negative".
- **p18 <= p12goal.** The stack has been regressing for 800M steps behind an
  instrument that could not see it. Everything stops until that is understood.

**Null: 50% goal share** -- and unlike `Rating/1v1` this is a real null,
because the opponents are frozen and cannot move to meet the policy.

Retain the chosen rungs permanently, outside the rolling `policy_versions`
rotation, so this ladder means the same thing in six months.

### Run 1 -- `p19pool`: give the 58% a gradient. One variable: the opponent.

Ticket 11, and the ladder from Run 0 is its precondition.

The k-arithmetic is not a hypothesis, it is arithmetic: at k=1 a differential
term has exactly zero expected advantage against a copy of itself. Raising the
share of training spent against genuinely different opponents is the cheapest
available way to make 58% of the budget mean something.

- `trainAgainstOldChance` 0.2 -> 0.5.
- Widen the pool so it spans a real skill range rather than 160M steps of
  near-clones: raise `tsPerVersion`, and pin the Run 0 ladder rungs as permanent
  pool members.
- Everything else frozen at p18 values.

**Predictions, written before the run:** the zero-sum-term behaviour metrics
become responsive for the first time since p16 -- `Shot/Saved Share` and
`Save/Converted` moved 0.6% and 1.5% across the whole 483M, so any movement
beyond their oscillation bands (4.9% and 2.0%) is signal. Ladder score against
the frozen rungs rises.

**Kill:** 100M steps with no ladder movement and no zero-sum-term response.

**Cost:** at the measured 64.1k steps/s, 100M is 26 minutes and 400M is 1.7 h.
Running longer is cheap; running *unreadably* is what has been expensive.

### Run 2 -- ~~`p20air`~~ **CANCELLED 2026-08-23. See `PLAN-v2-probes.md`.**

p20air would have tested doubled jump actions and the `aerial` spawn on the p18
checkpoint. Both only work from a fresh policy -- `Action/Jump When Grounded
Upright` is 4x below its null and immovable -- so a null would have been
misleading rather than merely weak. Replaced by three fresh 300M probes that
test v2's day-one decisions in the regime v2 will live in.

<details><summary>original text, kept for the reasoning</summary>

### Run 2 -- `p20air`: contingent on Run 0, and it decides issue 05

The one axis still moving is `AirTouch`, and the ladder tells you which way to
take that.

- **If p18 >= p17 on the ladder**, `AirTouch` is buying something real. Then
  give it the two things it has never had, from tickets 03 and 04: **double the
  jump actions** (Zealan's own remedy, never tried in this project, and
  `Action/Jump When Grounded Upright` sits at 0.050 against an unmasked null of
  0.20) and **actually run the `aerial` spawn scenario** (`CurriculumWeights`
  already defines it; `SpawnMode::Random` has never invoked it). One of the two
  per run, not both.
- **If p18 <= p17 on the ladder**, `AirTouch` has been eating the run for 483M
  steps. Cut it to satisfy the guard.

Either branch closes issue 05's red test, which is the only failing case in the
suite and has been drifting since p17.

</details>

### Deferred, and the reason has changed

**Ticket 06 (`gaeGamma` 0.99 -> 0.995) drops behind Run 1.** It still gates
tickets 01, 07 and 10 and the horizon arithmetic in it is still correct. But a
longer horizon on a term with zero expected gradient still yields zero gradient.
**Buy the horizon only once there is something at the end of it worth
discounting.** When it is taken, budget explicitly for critic recovery -- this
run just measured how slowly that critic moves (`GAE/Returns STD` needed 483M
steps to shift 1.8%).

**Tickets 01, 02, 07, 08, 10 stay blocked**, now on Run 0 and Run 1 rather than
on p18entropy. Ticket 07's standing verdict is unchanged and now better
supported: `SaveReward` is UNFALSIFIED, and it was carrying 4.4% of a budget
that had stopped producing gradient at all.

**Ticket 09 (positioning instruments) partly folds into Run 0.** The ladder is
the outcome measurement it was really asking for. The per-behaviour instruments
in that ticket are still worth building, each with its null, but they are now
diagnostics for explaining a ladder result rather than the primary evidence.

## Housekeeping, small and worth doing

- **Put the entropy carry-forward in `Config.h`.** `entropyScale 0.003 ->
  0.002` and `entropyTarget 0.40 -> 0.0`. `RUNLOG.md` declared these "the
  default worth keeping" on 2026-08-21, they lived only on the command line,
  and the 08-23 resume silently reverted them. **Command-line overrides do not
  survive a resume.**
- **CLAUDE.md's throughput figure is 23% stale.** It says plan with ~52k
  steps/s; this run measured **64,112 mean with `--track-skill` on**. 483M
  steps in 3.4 h wall-clock.
- **Sync p18entropy's checkpoint into the repo.** It was launched as
  `./HivemindBot train` directly rather than through `scripts/train.sh`, so the
  EXIT trap never fired and `scripts/sync_checkpoints.py` never ran.
- **Publish `Shot/Resolved Rate`** next to `Shot/Saved Share` (spec.md minor):
  publishing the denominator beside a conditional is the standing p10touch
  lesson and this one still breaks it.
- **Disk**: `main-p17cal` and `main-p18entropy` each carry a 143 MB
  `policy_versions` copy. Safe to drop for finished probe labels -- but keep the
  Run 0 ladder rungs.
