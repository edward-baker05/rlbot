# Reproduce a known-good baseline before building anything else

Status: design, approved 2026-08-18. Supersedes the reward direction in
`2026-08-18-reward-redesign-design.md` (that spec's D19-D20 stack shipped as
p7approach and is recorded as a failure in `runs/RUNLOG.md`).

## The problem this solves

This project has run 25+ experiments and has **no known-good reference point**.
Every run is measured against another failure, which is why the RUNLOG keeps
producing confident verdicts that later get retracted (p1probe-j retracted by
p1pay; D3 reversed by D20; D20 now falsified by p7approach).

Two measurements define the situation.

**M1. The bot has never learned to drive at the ball.**
`Player/Velocity Alignment` = `mean(max(0, v-hat . dirToBall))` reads
**0.3135 -> 0.3135** across p7approach's 100M steps. The analytic null for a
uniformly random direction is `E[max(0,cos)] = 1/pi = 0.3183` in the ground
plane and `0.25` in 3D. p6budget's split was grounded **0.286** -- below the
planar null -- and airborne 0.301.

The bot's velocity direction is at chance with respect to the ball, in every
run, while 40-45% of reward mass paid for that exact quantity. The null was
never computed, so 0.30 was read as low-but-real.

**M2. The reference says this should take a few dozen million steps.**
Zealan's `making_a_good_bot.md`: *"After running these rewards for a few dozen
million steps, your bot should be hitting the ball quite frequently."* At 100M
steps p7approach has `Player/Ball Touch Ratio` 0.0011 and 0.16 touch events per
episode, i.e. ~84% of episodes never touch the ball.

So the gap is not patience. Something in this codebase's bundle -- reward,
spawn distribution, action mask, or observation -- prevents a result the
reference gets in a third of the steps.

## Decision D1: reproduce before diverging

Port Zealan's early-stage configuration as literally as the C++ allows, run it,
and gate on his stated outcome. Only once it reproduces do we reintroduce this
project's own ideas, one variable per run.

Rejected alternative: rebuild observation + reward + hyperparameters together
and run long (the "fix everything" path). It changes five things at once, which
is the failure mode that produced the existing RUNLOG. If it did not work we
would learn nothing.

Rejected alternative: bisect the current failing config with single-variable
probes first. Better discipline, but every arm is anchored to a broken
baseline, so a null result is uninformative. Bisection is the *second* phase,
once there is a working config to bisect against.

## Decision D2: the reward stack is the guide's, verbatim in shape

Four terms. No goal reward, no boost reward, no ball-to-goal term, no tuning
penalties.

| Term | Class | Guide weight |
|---|---|---|
| Touch | `RLGC::TouchBallReward` | 50 |
| SpeedToBall | `Hive::SpeedToBallReward` | 5 |
| FaceBall | `RLGC::FaceBallReward` (**signed**) | 1 |
| Air | `RLGC::AirReward` | 0.15 |

Three deliberate consequences, each of which reverses a decision this project
made on its own and then measured failing:

**No goal reward.** The guide is explicit: *"Having these rewards before the bot
is capable of actually hitting the ball just adds lots of noise to the overall
reward and will slow learning."* p7approach's goals arrive 0.116 times per
episode and 49% of them come from `Scenario/Defend`, whose `EndedInGoal` is
0.87 -- the defender conceding, not the attacker scoring.

**FaceBall is signed, not rectified.** rlgym's `FaceBallReward` returns the raw
cosine, so pointing away is punished. p7approach rectified it on the reasoning
that "shadow defence and retreating for a bounce both need the nose off the
ball". Combined with a rectified `SpeedToBall` and a `WrongSurface` term firing
on 4.6% of steps, that left a stack in which **no state the bot can enter is
ever penalised**. The measured argmax of that stack is: carry speed in a
straight line and never turn, because turning is the only action that costs
speed. p7approach converged toward exactly that -- `Action/Steer Nonzero`
0.160 -> 0.087, `Jump When Grounded Upright` 0.755 -> 0.878.

**Air is paid, not taxed.** The guide's `AirReward` is *positive*, because his
bots stop jumping. Ours are 93% airborne, so porting this verbatim looks
perverse. It is ported anyway: at 0.15 against SpeedToBall's 5 it is 3% of the
dense budget, and if the air problem survives a reward that actively pays for
air, the cause is the action mask (D4), not the reward. That is precisely what
a reproduction is for.

## Decision D3: budgets stay, denominated in touches instead of goals

The budget discipline from `2026-08-18-reward-redesign-design.md` -- every
weight declared as an episode integral, converted in exactly one place -- is
kept. It is the right instinct and it is what made p1air's `grounded = 0.05`
(9.0 goal-units per episode for holding still) legible in hindsight.

The *unit* changes. A goal is received ~0.116 times per episode and cannot be
audited from telemetry; **a touch is the unit**, occurs 0.16-2 times per
episode, and is directly measured by `Touch/Edge Rate` and
`Player/Ball Touch Ratio`.

Restating the guide's weights in touch-units (touch = 1.0, 171-step reference
episode) makes the actual gap visible:

| Term | Guide, per episode | p7approach, per episode | Ratio |
|---|---|---|---|
| Dense approach (Speed + Face) | **20.5 touch-units** | 1.83 touch-units | **11x** |
| Touch | 1.0 per event | 1.0 per event | 1 |

p7approach paid for a perfect episode of approach at 1.83 touches. The guide
pays 20.5. That is the single largest numerical difference between the two
stacks and it was never computed, because the two were denominated in
different units.

## Decision D4: the action mask is ported too

`RLGC::DefaultAction::GetActionMask` restricts a grounded car to 42 of 90
actions, of which 18 press jump -- a **42.9%** jump prior. rlgym's
`LookupTableAction`, which the guide specifies, applies no mask at all: 18/90 =
**20%**.

With air stints of ~15 decision steps and ground dwell `1/p_jump`, a uniform
policy is airborne 87% masked versus 75% unmasked. p1advnorm measured 0.886 at
jump rate 0.43, matching the masked prediction.

The mask is not a bug -- it stops the policy spending capacity on no-ops -- but
it is an undocumented divergence from every reference implementation, and it
doubles the exploration probability of the behaviour this project has spent
eight runs fighting. It is part of the port.

Implemented as `Hive::UnmaskedAction : RLGC::DefaultAction` overriding
`GetActionMask` to return all-true, selected by `TrainConfig::maskActions`.

**Parity trap:** action masking changes inference behaviour and must be plumbed
to deployment through `HIVE_*` alongside `tickSkip`/`actionDelay`/`ModelShape`,
and checked by `./HivemindBot verify`.

## Decision D5: spawn from RandomState, not the curriculum

The guide specifies `RandomState(ball_rand_speed=True, cars_rand_speed=True,
cars_on_ground=False)`. `RLGC::RandomState` already implements exactly this,
including cars airborne on 50% of spawns.

The 10-entry curriculum is this project's own invention and has never been
validated against anything. It is also the most likely cause of the
**Early/Late collapse** that has stood unfixed since p1age: Early(0-1s) touch
rate 0.0081 vs Late(4s+) 0.00025, a **32x** gap, with `Episode/Late/Ball Dist`
4353 against Early's 2722 -- the bot ends further from the ball than it
started. A curriculum that hands the bot a favourable first second teaches the
first second.

The curriculum code stays in the tree. `TrainConfig::spawn` selects
`Random | Curriculum`; the port uses `Random`.

## Decision D6: deliberate non-ports, and why

| Item | Reference | Kept | Reason |
|---|---|---|---|
| `entropyScale` | 0.01 (rlgym-ppo) | **0.002** | GigaLearn normalizes entropy by `log(numActions)`; the scales are not comparable. This project measured 0.018 and 0.01 both pinning the policy near-uniform, and 0.002 producing its only breakthrough (p1probe-f). Porting 0.01 verbatim is porting a known-broken value. |
| No-touch timeout | 10 s | **12 s** | Shorter episodes raise the share of early-episode steps, where touch rate is 32x higher, inflating `Player/Ball Touch Ratio` by an amount comparable to the gate. Comparability with p1-p7 is worth more than 2 s of fidelity. |
| Observation | `DefaultObs` | unchanged | The guide calls the default "a decent starting point" and relative positions "a bit more advanced". The obs rebuild is real and planned (see Next phases) but including it would make this not a reproduction. |
| Network shape | `[2048,2048,1024,1024]` | unchanged 512s | Not what is broken; changing it invalidates comparison with every prior run and costs SPS on a 6 GB card. |
| eps-floor patch | absent | **kept** | Raises every valid action's probability by `0.02/N`. It is protective, verified to reverse an extinction on a dead policy, and the RUNLOG's standing lesson says keep it regardless of reward direction. |
| Advantage standardization patch | n/a | **kept** | rlgym-ppo standardizes advantages already. This patch makes us *match* the reference, not diverge from it. |

Ported as-is: `policyLR = criticLR = 2e-4` (guide: "bot that can't score yet"),
`epochs = 2`, `tsPerItr = 50_000` (guide: "50,000 are good for early
learning"), `miniBatchSize = 25_000`, `gaeGamma = 0.99`, `gaeLambda = 0.95`.

## Decision D7: every metric ships with its null

No run conclusion may cite a metric whose chance value is unknown.
`docs/metrics.md` gains a null column. The three that matter now:

| Metric | Null | Source |
|---|---|---|
| `Player/Velocity Alignment` | **0.3183** (planar), 0.25 (3D) | `E[max(0,cos)]` for a uniform direction |
| `FaceBall/Rectified` | same | same |
| `Action/Jump When Grounded` | **0.4286** masked, **0.20** unmasked | jump actions / available actions |

## Run protocol and kill criteria

The port runs as `p8ref`. Predictions are written into `runs/RUNLOG.md`
**before** the run starts, and the run is killed the moment a gate fails --
this is what the project has been missing, and it is why 25 experiments cost
15 GPU-hours and produced no reference point.

**Health gate, checked at 10M (~3 min):** `Policy Entropy` must be falling and
`SB3 Clip Fraction` must be in [0.02, 0.25]. If entropy is flat, stop: the
problem is a hyperparameter, and no reward conclusion is available. This is the
gate p7approach would have failed in its first three minutes.

**Primary gate, 25M (~8 min):** `Player/Velocity Alignment` > **0.35**, i.e.
clear of the 0.3183 null. This is the leading indicator; a bot that is not
driving at the ball cannot touch it. If this fails, the reference does not
reproduce in this codebase and the next suspect is the observation.

**Gate, 50M (~16 min):** alignment > 0.45 and `Player/Ball Touch Ratio` >=
**0.004** (3.6x p7approach; p1air reached ~0.0057 at 95M).

**Gate, 100M (~31 min):** touch ratio >= **0.008**, and
`Episode/Late/Touch Rate` within **5x** of `Episode/Early/Touch Rate`
(currently 32x). The Early/Late ratio is the real test: it separates "learned
to play" from "was handed a good spawn".

Past 100M the run continues to 500M+ with no further changes. Throughput is
~54k steps/s, so 500M is ~2.6 h and 2B is ~10 h.

## Scope of the gutting

Removed from `bot/src/env/Rewards.{h,cpp}`: `WrongSurfaceReward`,
`TouchEdgeReward`, `FaceBallRectifiedReward`. `SpeedToBallReward` stays.
`Touch/Edge Rate` survives as a Train.cpp metric, computed inline as it already
is, so the per-step-vs-rising-edge gap stays visible.

`Config.h`: `RewardBudget` replaced by a touch-denominated budget struct;
`CurriculumWeights` kept but no longer wired by default; `TrainConfig` gains
`spawn` and `maskActions`.

`bot/tests/test_rewards.cpp` rewritten for the four-term stack, including an
executable assertion that `FaceBallReward` is signed (returns < 0 when facing
away) -- the property whose loss defined p7approach.

Everything else -- instrumentation, curriculum setters, eval/verify/spectate,
the four external patches, PacketConvert, RLBot client -- is untouched.

## Next phases (not this spec)

- **Phase B, bisect:** with a working baseline, reintroduce this project's
  ideas one variable per 50M run -- masked actions, curriculum spawns, goal
  reward, rectified FaceBall, WrongSurface -- and keep only those that beat the
  baseline.
- **Phase A, build:** car-frame relative observation (`relative_physics` from
  rlgym-tools: `(target.pos - origin.pos) @ rot`) plus action stacking, which
  Necto, Nexto and current community obs all use and which this project's obs
  lacks entirely; larger network; higher gamma; an external-opponent Elo ladder
  against `libs/opponents/NectoFamily` so runs are comparable across the whole
  project rather than each against its own version pool.
