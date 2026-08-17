# Reward design

## What went wrong first

The first reward function was assembled from the weights in GigaLearn's example
config. After 50M steps the bot looked like it was learning — average reward
climbed 0.27 → 0.93 — but the metrics told a different story:

- **Air ratio pinned at 0.87–0.91** for the entire run, never moving.
- **Goal reward net negative.** It conceded slightly more than it scored.
- **Skill rating declining monotonically** from +2.5 at 28M to −20.5 at 50M.
  The policy was losing to its own earlier snapshots.

> **A correction worth keeping.** The first reading of this was that `AirReward`
> *caused* the 87% air ratio. It did not. Removing `AirReward` entirely left the
> air ratio at 0.894 — because a fresh, high-entropy policy presses jump
> constantly and cannot recover, so it starts at 0.898 before any training at
> all. What `AirReward` actually did was remove the *pressure to fix it*: it
> paid the bot for a state it was already stuck in, turning a problem to be
> solved into an income stream. The distinction matters, because the fix is not
> just deleting the reward — it is making sure something else rewards car
> control. See the note on `velPlayerToBall` below.

Decomposing the reward showed why:

| Component | Weighted/step | Share |
|---|---:|---:|
| VelocityPlayerToBall | 0.586 | 63.3% |
| **AirReward** | **0.217** | **23.5%** |
| FaceBall | 0.050 | 5.3% |
| SaveBoost | 0.044 | 4.7% |
| StrongTouch | 0.024 | 2.6% |
| Goal | −0.015 | −1.6% |
| everything else | ~0.01 | ~1% |

**98.4% of reward came from shaping. 1.6% came from doing anything with the
ball.** `AirReward` alone paid 15x more than every outcome reward combined, for
the trivially achievable act of not being on the ground.

The policy was not failing to learn. It was learning exactly what it was paid
for.

---

## The rule

**Never pay per-step for a state the policy fully controls.**

Such a reward is an income stream that requires no skill, and PPO will find it.
Being airborne, holding boost, facing the ball — all are states the bot can
enter at will and hold indefinitely. Each is a farm.

Every reward in the current function is instead one of four safe forms:

| Form | Why it cannot be farmed | Example |
|---|---|---|
| **Telescoping** | It is the derivative of a potential, so any closed path in state space sums to zero | `VelocityPlayerToBall` is exactly `−d(distance)/dt`; approaching then retreating nets nothing |
| **Bounded** | Total obtainable value is capped by a real resource | `PickupBoost` is a sqrt-delta: empty→full is worth 1.0 total, topping up from full is worth ~0 |
| **Impulse** | Requires an actual change in the world | `StrongTouch` needs Δball-velocity; resting against the ball pays zero forever |
| **Terminal** | It is the thing you actually want | `Goal` |

Two rewards were dropped during design for failing this test:

- **`TouchBallReward`** returns `ballTouchedStep`, which is true on *every step
  you are in contact*. A bot leaning on the ball would collect it continuously —
  at the weight I had planned, ~15x the entire intended reward. `StrongTouch`
  covers the same lesson without the exploit, so it carries the touch signal
  alone.
- **`SaveBoostReward`** is pure occupancy and, worse, pays the bot *not to use
  boost* — directly fighting the aerial game the later phases want.

---

## How the weights were derived

Not guessed. The procedure:

1. **Measure raw per-step values.** GigaLearn logs unweighted per-component
   means as `Rewards/*`, which the CSV receiver captures. From the 50M run:
   `VelocityPlayerToBall` 0.146, `StrongTouch` 0.0004, `PickupBoost` 0.0015, etc.

2. **Estimate them for the policy you are training towards**, not the one you
   have. The measured touch ratio was 0.0039 — one touch per 17 seconds. A
   competent early bot is nearer 0.02. Sizing weights against the broken policy
   would just entrench it.

3. **Choose target shares of return**, then solve for weights.

4. **Check the farm ceiling**: what is the most reward obtainable *without ever
   touching the ball*? Compare it to the reward for playing well.

Target composition at a **competent** policy (touch ratio ~0.02):

| Component | Weighted/step | Share | Form |
|---|---:|---:|---|
| VelocityPlayerToBall | 0.44 | 45% | telescoping |
| StrongTouch | 0.15 | 30% | impulse |
| VelocityBallToGoal | 0.07 | 14% | outcome |
| Goal | 0.03 | 6% | terminal |
| PickupBoost | 0.01 | 2% | bounded |

The structure is self-correcting: shaping contributes a roughly fixed amount per
step regardless of skill, while outcome terms grow as the bot improves. The
better it plays, the larger the fraction of its reward comes from playing well.

> **This table describes the destination, not the starting point.** Early on,
> composition is ~100% shaping — not because the design is wrong, but because a
> bot with a touch ratio of 0.0008 has no outcome available to earn. Do not
> "fix" a shaping-heavy composition at 20M steps by inflating outcome weights;
> that just adds variance to a signal the policy cannot yet reach. Judge the
> composition only once touch ratio is healthy.

### On the approach reward, and a mistake worth recording

`VelocityPlayerToBall` was first set to **0.75**, down from the old 4.0, on the
theory that a large approach reward produces a ball-chaser. Measured over 20M
steps, that was wrong in an instructive way:

| | old (4.0) | v2 (0.75) | v3 (3.0) |
|---|---:|---:|---:|
| VelocityPlayerToBall (raw) | 0.073 | **0.018** | 0.080 |
| touch ratio | 0.0012 | **0.0005** | 0.0008 |
| air ratio | 0.895 | 0.894 | **0.873** |

At 0.75 the bot simply **stopped driving at the ball**, and with nothing else
dense enough to bootstrap from, learning stalled outright.

The lesson is that in phase 1 this term is not really shaping at all — it is the
**car-control curriculum expressed as a reward**. To earn it, the bot must land
on its wheels, orient itself, and drive somewhere deliberate. That is precisely
the skill a fresh policy lacks, and nothing else in the function pays for it.
Ball-chasing is a phase-2 problem; being unable to reach the ball is a phase-1
one.

A high weight is safe here in a way it would never be for an occupancy reward,
because the term telescopes: there is nothing to farm, only a bias towards being
near the ball.

### Farm ceiling

What is the most reward obtainable *without ever touching the ball*?

| | Max/step, never touching | Playing well | Separation |
|---|---:|---:|---:|
| Old | 0.65 (Air 0.25 + SaveBoost 0.20 + pads) | 0.92 | **1.4x** |
| New | ~0.02 (pads only) | ~0.7 | **~35x** |

The old design let a bot earn 70% of a good policy's reward by doing nothing but
hovering with a full tank. The new one leaves only boost pads, and those are
sqrt-bounded so the rate self-limits.

`VelocityPlayerToBall` contributes **zero** to this ceiling despite its large
weight, because a bot that never reaches the ball must eventually turn away, and
the term sums to zero over any closed path. That is the entire reason it can
safely be the largest weight in the function.

### On the goal weight

Goals are far too rare to size by their share of *average* reward. They are
sized against the **discount horizon** instead.

At `gaeGamma = 0.99` and 15 steps/sec (tickSkip 8), the horizon is ~100 steps
(~7 seconds). Good ongoing play accrues ~0.3/step, so total discounted future
value from play alone is ~`0.3 / 0.01 = 30`. A goal weighted at **30** therefore
roughly doubles the value of the moment it happens — strong enough to dominate
credit assignment within the window where the value function can actually see
it, without drowning the dense signal that makes early learning possible.

### On team spirit

`ZeroSumReward`'s first argument is the fraction of a reward shared across
teammates. It is kept **low (0.0–0.3)** in this phase.

High spirit is how you eventually get rotations and passing, but it smears credit
across cars — and a bot that cannot yet reliably strike the ball needs to know
precisely which of *its own* actions worked. Raise it in the teamplay phase.

---

## Phases

Rewards are staged: teach one idea, let it consolidate, then add the next.
Introducing dribble or demo rewards before the bot can strike the ball reliably
just gives it more ways to earn without improving.

`RewardPhase` in `Config.h`. **Only `Foundations` is designed.** The later
phases are deliberately left unimplemented rather than filled with plausible
numbers — each should be derived from the run that precedes it, using the
procedure above, against that run's measured values.

### Phase 1 — Foundations *(current)*

Reach the ball, strike it, aim it. Five components.

**Advance when:** `Player/Ball Touch Ratio` > ~0.02, `Rating/1v1` trending up
rather than down, and goal reward reliably positive.

### Phase 2 — Possession *(not designed)*

Keeping the ball rather than just hitting it. Would add ground-dribble and
ball-control terms, and a boost-economy term now that boost use is meaningful.
Raise `curriculum.groundDribble`.

### Phase 3 — Aerial *(not designed)*

`TouchHeightReward` belongs here — it already exists in `Rewards.h` and is
deliberately unused. Raise `curriculum.aerial`, and switch on `airDribble` and
`flipReset`.

### Phase 4 — Teamplay *(not designed)*

Raise team spirit substantially. Add bump/demo terms and rotation-aware shaping.
Switch on `curriculum.demo`.

> The curriculum must move with the reward. Spawning air-dribble scenarios while
> the reward pays nothing for air dribbling does not teach air dribbling — it
> spends samples on a situation the policy has no gradient to improve at.
> `CurriculumWeights` is annotated with the phase each scenario belongs to.

> Start a fresh run when you advance a phase. A policy carried across a reward
> change is optimising a different objective than the one it was trained on, and
> comparisons across the boundary are meaningless.

---

## Diagnosing a reward function

The reward curve alone will not tell you if this is working — it rises in every
run, including the broken one above. Watch instead:

| Signal | Meaning |
|---|---|
| `Rating/*` trending **down** | Losing to your own past selves. Almost always reward farming. |
| A shaping component >40% of weighted reward | Something is being farmed. Decompose and check. |
| A behaviour metric pinned flat (e.g. air ratio) | The policy found a fixed strategy that satisfies the objective. |
| Goal reward ≈ 0 or negative late in a run | It is not actually playing the game. |

To decompose a run's reward, take the `Rewards/*` columns from its CSV and
multiply each by its configured weight. That table is what exposed the original
problem, and it is the first thing to produce whenever a run looks wrong.
