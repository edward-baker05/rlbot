# Reward redesign: goal-referenced budgets and car control

Date: 2026-08-18
Status: approved design, pre-implementation
Supersedes: the reward stack described in `runs/RUNLOG.md` entries
`deploy-probe` through `p5goalpot`

## Context

Five reward configurations have been measured (`p1air`, `p2*`, `p3strike`,
`p4pbrs`, `p5goalpot`). The best of them, `p1air`, reached a touch ratio of
0.0131 over 245M steps and could not jump; every attempt since has been worse,
topping out at 0.0021. Three control dimensions have been driven to extinction
across the sequence (jump, steer, throttle).

The diagnosis in the RUNLOG is that the reward function, not PPO health, is the
binding constraint. Two structural faults run through every previous version:

1. **Weights were per-step floats with no accounting of what they integrate
   to.** `grounded = 0.05` over a 150-step episode is 9.0 — against
   `goal = 100`, that is 9% of a goal per second for holding still on your
   wheels, forever. This was invisible because nobody wrote down the integral.
   It is the mechanism behind p1air's do-nothing attractor.

2. **The dense shaping terms were all ball-directed**, and ball-directed
   shaping in this action space has a specific pathology (see D3 below) that
   penalises steering twice and extinguishes it.

This design guts the reward stack and rebuilds it around car control, with
goals as the reference unit and every weight expressed as a budget.

## Decisions

### D1. Goals are the unit; every other weight is a budget in goal-units

`GoalReward` has weight exactly **1.0** (it is already zero-sum: +1 scored,
−1 conceded with `concedeScale = -1`). No shaping weight may be written as a
bare per-step float. Every term declares either:

- a **rate budget**: goal-units earned by holding the behaviour perfectly for
  one reference episode, or
- an **event budget**: goal-units per occurrence.

`Rewards.cpp` converts at exactly one site:

```
rate weight   = budget / REFERENCE_EPISODE_STEPS
per-second    = budget / STEPS_PER_SECOND
event weight  = budget                     (pass-through)
```

with `STEPS_PER_SECOND = 15` (tickSkip 8 at 120 Hz) and
`REFERENCE_EPISODE_STEPS = 150` (10 s).

`REFERENCE_EPISODE_STEPS` is provisional. `noTouchTimeout` caps a never-touching
bot at 180 steps and goals end episodes early. A new `Episode/Mean Steps`
metric exists so run A re-derives it from telemetry rather than keeping a
guess (D6 of the roadmap spec).

**Why this is structural and not cosmetic:** it makes the p1air failure
unrepresentable. A term cannot integrate to 9 goal-units per episode without
someone typing `9.0` into a field labelled goal-units per episode.

### D2. Raising the goal weight cannot improve the goal signal

Scaling goals up is a recurring intuition and it is mechanically void.

`GAE.cpp:52` is `curReward = _rews[step] / returnStd` with
`standardizeReturns = true`, so absolute magnitude cancels. This is why
p1probe-h deleted 44.3% of all reward mass and moved `Returns STD` by 4%.

More strongly, for a per-episode terminal payoff `X = k` with probability `p`:

    E[X] = k·p      SD[X] = k·sqrt(p(1-p))      SNR = sqrt(p / (1-p))

`k` cancels. The goal term's signal-to-noise is set entirely by how often goals
happen, and scaling raises signal and noise in exact proportion. There is also a
hard ceiling: `rewardClipRange = 10` clamps the *standardized* reward, so beyond
some `k` further increases are discarded outright.

The only lever on goal SNR is `p`. That is what the curriculum and the shaping
terms are for, and it is why shaping is deliberately dominant in run A (D8).

### D3. Ball-directed dense shaping is banned; it is replaced by its additive factoring

"Rate of reduction of distance to the ball" is not an alternative to the terms
that failed — it is identical to them. That rate is the radial velocity
component `dirToBall · v`, i.e. `VelocityPlayerToBallReward` times a constant;
its potential-based form is `BallProgressReward`, which is p4pbrs. Both were
measured:

- Farmable form (p1air, p5goalpot): `RewardShare` climbed to 0.451, the
  approach-farm signature. On p3strike it drove `Action/Steer Nonzero` to
  **0.0006** at 100M steps.
- Potential form (p4pbrs): touch ratio **fell** 0.0011 → 0.0007. Chase-hit-chase
  is a cycle in that potential, so it telescopes to zero and teaches nothing.

The mechanism behind the steering extinction is the *product* structure:

    R_old = (|v|/V)·cos(theta)          d2R/d|v| dcos = 1/V  != 0
    R_new = w_s·(|v|/V) + w_f·cos(theta) d2R/d|v| dcos = 0

In the product form the marginal cost of scrubbing speed is proportional to how
well you are aimed, and the marginal value of improving your aim is proportional
to how fast you are going. At high speed *and* good alignment — exactly the
state reached while approaching — a steering input is charged on both factors at
once. In the additive form it is charged once, on speed only.

**Speed + FaceBall is therefore a factored, additive `VelocityPlayerToBall`.**
Same intent, gradient geometry that does not extinguish steering.

The same factoring explains why previous bots never learned to flip or boost for
speed. A flip's impulse is along the car's forward axis and costs ~1.25 s of
steering authority. Under the product form its value is gated by `cos(theta)`
and it forfeits the terminal adjustment; under `|v|` it is paid unconditionally.
The flip's advantage is strictly larger under generic speed.

### D4. Asymmetric FaceBall is implemented as its decomposition, not as rectified weights

Facing away from the ball is sometimes correct (shadow defence, retreating for a
bounce), so the negative side should be weaker than the positive. Implementing
that as rectified weights has a provable side effect:

    w+·max(0,c) + w-·min(0,c)  ==  ((w+ + w-)/2)·c  +  ((w+ - w-)/2)·|c|

The second component pays **identically for nose-at-ball and nose-directly-
away**, and zero for perpendicular. It is also an annuity: for orientation
uniform on the sphere, `c = cos(theta)` is uniform on [-1,1] so `E[|c|] = 1/2`,
and a policy with no facing preference at all collects `(w+ - w-)/4` per step.

It is shipped **as two specs** — `FaceBall` (symmetric, signed) and
`FaceBallAxis` (`|c|`) — so the annuity gets its own `RewardShare` line and its
own budget. Identical reward, visible instead of hidden. `RewardShare` reports
`|r·w|` and cannot distinguish a signed term from a rectified one, which is
exactly how this would have gone unnoticed.

Ratio fixed at 2:1 (`w- = w+/2`), giving `FaceBallAxis` one third of
`FaceBall`'s budget.

### D5. The wrong-surface penalty is flat, and needs no orientation grading

`Player` inherits RocketSim's `CarState`. `worldContact.hasContact` is set only
from `Arena::_BtCallback_OnCarWorldCollision` — the car's **chassis hitbox**
producing a Bullet manifold against world geometry. Wheels are raycast
suspension and never generate a manifold. So

    worldContact.hasContact && !isOnGround

is exactly "a part of me that is not wheels is against a surface", correct on
walls, the corner curve and the ceiling, with no plane assumption.

Grading by `rotMat.up · worldContact.contactNormal` was considered and
rejected. `isOnGround` is defined as 3+ wheels in contact, so the gate is
already the in-control discriminator: if the wheels were doing their job you
would never be inside the penalty. Grading would only distinguish 45°-wrong
from 90°-wrong, and being on your side is as useless as being on your roof.

The recovery gradient that grading would have bought is unnecessary:
`Car::_UpdateAutoFlip` means escaping your roof is a single discrete input
(jump, while chassis-contacting with `contactNormal.z > CAR_AUTOFLIP_NORMZ_THRESH`),
and the epsilon-floor patch keeps that input sampled.

**Known cost:** a front flip drives the nose into the floor for a few ticks.
`Player` samples the final tick of an 8-tick step, so the scrape is caught ~25%
of the time. Priced in D7 and instrumented as
`Surface/Wrong Contact While Flipping`.

### D6. The clean-landing bonus scales with impact speed *squared*

A landing bonus is what makes going airborne net-positive rather than merely
permitted — without it the only term touching air play is a penalty, in a
project that has extinguished jump three times.

Naive forms are farmable by bunny-hopping. Scaling by impact speed squared is
not, and the jump constants prove it:

| | leave-ground speed | flight | bonus | reward rate |
|---|---|---|---|---|
| bunny hop | ~450 uu/s | ~1.4 s | (450/1100)^2 = 0.17 | 0.10 /s |
| real aerial (~1000 uu) | ~1140 uu/s | ~3.5 s | capped 1.00 | 0.27 /s |

Hop speed is `JUMP_IMMEDIATE_FORCE` (875/3 = 291.7) plus `JUMP_ACCEL`
(4375/3 = 1458.3) held for `JUMP_MAX_TIME` (0.2 s), less gravity
(`GRAVITY_Z` −650) over the hold: ~453 uu/s. Aerial return speed is
`sqrt(2 · 650 · 1000)` = 1140 uu/s.

Under **linear** scaling the rates are 0.24 /s versus 0.27 /s and hopping is
competitive — a farm. Under **squared** scaling the hop drops to 0.098 /s and
the aerial dominates 2.8x. The square is what kills the farm and it is derived,
not picked. Reference impact speed is 1100 uu/s.

Measured as downward speed `max(0, -prev->vel.z)`. Known limitation: a wall
landing has no vertical component and scores zero. Accepted — using
`|prev->vel|` instead would pay for horizontal speed, double-counting Speed and
biasing toward the wall.

### D7. Budget table

| Term | Kind | Budget (goal-units) | Per-step weight |
|---|---|---|---|
| Goal | terminal | ±1.0 per goal | 1.0 |
| Speed² | rate | 0.30 | 0.00200 |
| FaceBall (signed) | rate | 0.20 | 0.00133 |
| FaceBallAxis (`\|c\|`) | rate | 0.067 | 0.00044 |
| Touch (rising edge) | event | 0.15 per touch | 0.15 |
| CleanLanding | event | 0.10 per max-impact landing | 0.10 |
| WrongSurface | rate penalty | −0.10 per **second** | 0.00667 |
| HarshSpeedLoss | event penalty | −0.10 per full-speed crash | 0.10 |

**Sign convention:** penalty *classes* return negative values and their
*weights* are positive, matching upstream `BumpedPenalty` / `DemoedPenalty`.
The budget column carries the sign for readability; the weight column is what
is passed to `WeightedReward`. Writing a negative weight against a negative
class value would double-negate, which is why the convention is stated rather
than inferred.

Three checks were run on these numbers:

**Flip tax is affordable.** Scrape cost ~0.25 steps × 0.00667 = 0.0017
goal-units per flip. Flip speed gain ~500 uu/s moves Speed² from 0.375 (at
1410) to 0.69: +0.315 × 0.002 = 0.00063/step held ~2 s = 0.019 goal-units. A
flip pays **11x** its own tax.

**Landing asymmetry points the right way.** Best landing +0.10; three seconds on
your roof −0.30. Staying wrong costs 3x what landing right pays, so recovery is
worth more as an avoided loss than as a bonus. This is what stops the landing
bonus becoming the objective.

**Free annuity is metered.** Speed²'s coasting floor
(0.375 × 0.30 = 0.113, see D8) plus FaceBallAxis for an orientation-indifferent
policy (0.067 × 0.5 = 0.033) totals **0.146 goal-units per episode obtainable
with no skill** — ~15% of the shaping budget, both components individually
instrumented. Compare p1air's `Grounded` at 9.0, unmeasured.

### D8. Speed is squared, and the anti-farm contingency is pre-committed

`DRIVE_SPEED_TORQUE_FACTOR_CURVE` reaches zero at **1410 uu/s**: throttle-only
top speed. So `|v|/2300 = 0.613` is available forever, free, with no boost and
no skill — 61% of the term's maximum as a do-nothing annuity.

Squaring reduces it and tilts the term toward the behaviour that motivated it:

| state | `\|v\|/2300` | squared |
|---|---|---|
| throttle-only cap (1410) | 0.613 | 0.375 |
| supersonic (2300) | 1.000 | 1.000 |
| ratio | 1.63x | 2.67x |

Free annuity falls 61% → 37.5%; the payoff for boost/flip-derived speed over
coasting rises 1.6x → 2.7x; the 0→1410 gradient survives, so the term still
bootstraps.

**Farming is capped by construction.** A bot that spends an entire episode doing
nothing but going as fast as physically possible earns 0.30 goal-units — exactly
what two ball touches earn, and 30% of one goal.

**Slow play is priced.** Speed² is positive, so slow play is not punished, but
forgone income is a real cost. Giving up *max* speed for 2 s costs 0.060 against
a touch's 0.15 (2.5x in favour of the touch); giving up *coasting* speed costs
0.022 (6.8x). Break-even is about **5 seconds** of deliberate slow play per
extra touch. Beyond that the term does bias against slow control. The correct
fix for that is off-limits (a term recognising "carrying slowly is good here" is
a possession reward, D4 of the roadmap spec), so the tool is keeping the
opportunity cost small.

Two fixes were considered and rejected:

- *Cap speed earnings per episode.* Non-Markov: the accumulated total is not in
  the observation, so the critic cannot predict it and the term degrades to
  noise.
- *Gate speed by ball distance.* `R = w(d)·(|v|/V)^2` has
  `d2R/dd d|v| != 0`, so when fast, being *far* pays more — an explicit
  incentive to drive away from the ball at speed. The same cross-derivative
  pathology as D3, on a different axis.

**Pre-committed contingency.** If run A shows farming, swap to the above-floor
form:

    (|v| / 2300)^2   ->   (max(0, (|v| - 1410) / 890))^2

which pays zero at throttle-only top speed, deleting the coasting annuity and
paying only for boost- and flip-derived speed. It costs the 0→1410 bootstrapping
gradient, which is why it is not the default; it also reduces the slow-play
opportunity cost to zero.

**Trigger (a decision rule, not a judgement call):** swap if
`RewardShare/Speed > 0.25`, **or** if `Speed/Above Throttle Cap Share` rises
while `Touch/Edge Rate` stays flat. Speed improving without touches improving is
the farm signature by definition.

### D9. Touch is a rising edge

`Touch` fires on `ballTouchedStep && !prev->ballTouchedStep`, not on
`ballTouchedStep`.

A per-step touch reward *is* a dribble reward: carrying the ball on the nose
would pay it every step, ~180x per episode. That is D4's flick-bot local optimum
arriving through the back door. The rising edge makes carrying the ball worth
exactly one touch.

### D10. HarshSpeedLoss is thresholded, and the threshold must be validated

    d = |v_prev| - |v|          (only when positive)
    penalty = clamp((d - T) / (2300 - T), 0, 1)

with `T = 400` uu/s per step. Derivation: RL braking decelerates at roughly
3500 uu/s², which over a 1/15 s step is 233 uu/s, so 400 is comfortably above
any input-driven deceleration and only reachable by collision.

**That 3500 figure is empirical, not a RocketSim constant** (`BRAKE_TORQUE_AMOUNT`
is a wheel torque and does not convert directly). The `Speed/Max Step Decel`
metric exists to validate `T` against the real distribution in run A. Do not
treat 400 as settled.

Exemptions: `ballTouchedStep` (a hard hit *should* cost speed — that is a good
outcome, not a bad recovery), and `!state.prev || !player.prev`.

**Deliberate overlap with Speed.** A per-step speed reward already means losing
speed costs future reward, and at gamma 0.99 the ~6.7 s horizon covers it. What
this term adds is a sharp, single-step, attributable signal at the collision,
which is worth real money for credit assignment. It also fires on the same
events as WrongSurface. Both overlaps are intentional and are recorded here so
the shares are not misread later.

### D11. Nothing except Goal is zero-summed

These are car-control terms; wrapping them adds opponent variance for no
competitive meaning. `GoalReward` alone carries the adversarial structure, and
it is already zero-sum by construction.

Zero-summing Speed was considered — it would delete the absolute annuity, since
in self-play both sides collect it and the mean is exactly zero. Rejected
because it also deletes the dense do-something signal that is the term's entire
purpose.

### D12. Deleted outright

`AirRecoveryReward`, `GroundedBonusReward`, `GroundedReward`,
`TouchHeightReward`, `AimedStrongTouchReward`, `AimMultiplier`,
`StrongTouchValue`, `TOUCH_MIN_KPH`/`TOUCH_MAX_KPH`, `BallProgressReward`,
`BallGoalProgressReward`, `VelocityPlayerToBallReward`, `PickupBoostReward`,
the `RewardPhase` enum, the `--reward-phase` flag, and the zero-weight-spec
index-alignment hack in `GeneralRewardSpecs`.

Consequences accepted: the p1air comparison baseline is gone, and boost
management is unrewarded, so the bot may run itself dry. Anything readded must
justify itself against measurement.

### D13. The horizon change is a separate run

`gaeGamma = 0.99` at tickSkip 8 gives a 100-step horizon = **6.7 s**, a 4.6 s
half-life, and discounts a goal 12 s away to **0.163**. A kickoff → possession →
drive → shot is 8–15 s, so the bot learns kickoffs from a signal attenuated 6x.

Worse, `gaeLambda = 0.95` means the horizon over which reward reaches the policy
*directly* is `1/(1 - gamma·lambda) = 1/(1 - 0.9405) = 16.8` steps = **1.1 s**.
Everything beyond that arrives only through V's bootstrap in the delta terms, so
the critic is the entire long-range credit path. Raising gamma alone barely
moves this: gamma 0.995 gives 18.3 steps = 1.2 s.

The intended fix is **gamma → 0.995 and lambda → 0.98** (40 steps = 2.7 s of
direct credit, at higher advantage variance — affordable now that the
advantage-standardization patch is in, since standardization is what makes a
noisier advantage tolerable). Not higher than 0.995: `noTouchTimeout` caps
episodes at 12 s of no contact, so horizon past ~13 s has nothing to discount.

| gamma | horizon | half-life | goal 12 s away |
|---|---|---|---|
| 0.99 (current) | 100 steps = 6.7 s | 4.6 s | 0.163 |
| 0.995 | 200 steps = 13.3 s | 9.2 s | 0.407 |
| 0.997 | 333 steps = 22.2 s | 15.4 s | 0.583 |
| 0.999 | 1000 steps = 66.7 s | 46 s | 0.835 |

**This does not ship in run A.** Run A is the reward rewrite at unchanged
gamma/lambda, so it is comparable to p1air and p5goalpot on the one axis that
matters. Run B is identical rewards at gamma 0.995 / lambda 0.98. p5goalpot
moved two variables and its RUNLOG row has to say "attribution is not clean";
that is not repeated here.

## Implementation

### Files

- `bot/src/env/Rewards.{h,cpp}` — rewritten.
- `bot/src/Config.h` — `RewardWeights` → `RewardBudget`; `RewardPhase` deleted;
  `STEPS_PER_SECOND` / `REFERENCE_EPISODE_SECS` / `REFERENCE_EPISODE_STEPS`
  added.
- `bot/src/main.cpp` — `--reward-phase` deleted.
- `bot/src/train/Train.cpp` — the entire `Pay/*` block deleted (it recomputes
  `StrongTouchValue`, `AimMultiplier` and `grounded`, all gone); new metrics
  added; `g_RewardWeights` → `g_RewardBudget`.
- `bot/tests/test_rewards.cpp` — rewritten.

### Reward classes

| Class | Fires | Value |
|---|---|---|
| `GoalReward` (upstream) | terminal | ±1 |
| `WrongSurfaceReward` | `worldContact.hasContact && !isOnGround` | −1 |
| `CleanLandingReward` | `!prev->isOnGround && isOnGround && !worldContact.hasContact` | `min(1, max(0,-prev->vel.z)/1100)^2` |
| `TouchEdgeReward` | `ballTouchedStep && !prev->ballTouchedStep` | 1 |
| `FaceBallReward` (upstream) | every step, ungated | `forward · dirToBall` |
| `FaceBallAxisReward` | every step | `\|forward · dirToBall\|` |
| `SpeedSquaredReward` | every step | `(\|vel\|/2300)^2` |
| `HarshSpeedLossReward` | every step | see D10 |

All difference-based terms guard on `!state.prev || !player.prev`.
`EnvSet::ResetArena` calls `prevGameStates[index].MakeEmpty()`, so `prev` is
genuinely null on the first step after a reset and a state-setter teleport
cannot be read as a velocity discontinuity.

### Metrics

New: `Surface/Wrong Contact Rate`, `Surface/Wrong Contact While Flipping`;
`Landing/Rate`, `Landing/Clean Share`, `Landing/Impact Speed`; `Speed/Mean`,
`Speed/Above Throttle Cap Share`, `Speed/Harsh Loss Rate`,
`Speed/Max Step Decel`; `FaceBall/Mean Cos`, `FaceBall/Axis Share`;
`Touch/Edge Rate`; `Episode/Mean Steps`.

Kept: `RewardShare/*` (the farm detector) and
`Action/Jump When Grounded Upright` (the extinction canary).

### Tests

Written first, `bot/tests/test_rewards.cpp`:

1. Budget conversion: a rate budget `B` yields `B/150`; a per-second budget
   yields `B/15`.
2. `WrongSurface`: on wheels, no contact → 0; chassis contact **with**
   `isOnGround` → 0 (the gate); chassis contact without → −1.
3. `CleanLanding`: no edge → 0; edge with chassis contact → 0; edge at
   450 uu/s ≈ 0.167; edge at ≥1100 → 1.0.
4. `TouchEdge` **dribble guard**: touched-now and touched-prev → 0;
   touched-now, not-prev → 1.
5. **Decomposition identity**: `FaceBall + FaceBallAxis` reconstructs the
   rectified asymmetric form exactly, with `w- = w+/2`, for `c > 0` and
   `c < 0`. The algebra of D4 as an executable assertion.
6. `HarshSpeedLoss`: below `T` → 0; above → scaled; exempt on
   `ballTouchedStep`; 0 when `prev` is null.
7. `SpeedSquared`: 1410 → 0.375; 2300 → 1.0.
8. Spec-list integrity: no zero-weight placeholders, names unique
   (the `RewardShare` metric indexes by spec order).

## Success criteria for run A

Primary: `Player/Ball Touch Ratio` beats p5goalpot's 0.0021 at equal step
count, and ideally tracks toward p1air's 0.0131.

The run is *informative* regardless of the primary, provided these are readable:

- `Action/Jump When Grounded Upright` stays above the epsilon-floor
  (~0.011). Below it, D5/D6 have failed to keep air play alive.
- `RewardShare/Speed` under 0.25, else trigger the D8 contingency.
- `Surface/Wrong Contact While Flipping` shows the flip tax is noise, per D7.
- `Speed/Max Step Decel` validates or refutes `T = 400`, per D10.
- `Episode/Mean Steps` re-derives `REFERENCE_EPISODE_STEPS`, per D1.

## Open items

- `REFERENCE_EPISODE_STEPS = 150` is a working figure pending run A telemetry.
- `T = 400` uu/s in D10 rests on an empirical braking figure, not a constant.
- Boost is unrewarded (D12). If run A shows the bot running dry, boost
  management returns as its own decision with its own budget.
- Wall landings score zero on `CleanLanding` (D6).
