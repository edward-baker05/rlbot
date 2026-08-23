# Run log

One line per run that matters. Comparisons are only valid between labeled
runs recorded here. Append newest at the top.

Format: `date | label | config delta from previous entry | why | outcome`

## RECONSTRUCTED 2026-08-23: what p15manual and p16 actually changed

Their `CONFIG_HISTORY.json` files are `sync_checkpoints.py` reconstructions with
an empty `changed` map, and the originals were never written. The weights were
recovered from the metrics CSVs instead: **`RewardMass/<term>` / |`Rewards/<class>`|
returns the budget directly for event terms and after multiplying by
`REFERENCE_EPISODE_STEPS` (391.5) for rate terms.** Calibrated against p18, where
the true weights are known: AirTouch 1.00x, TouchEdge 0.99x, ShotOnTarget 1.01x,
PickupBoost 1.00x, SpeedToBall 1.00x, SaveBoost 0.99x, Air 1.01x. Signed
near-zero-mean terms do not recover (`Goal` is 327x out) and `Save` reads ~1.37x
high, so treat those two as indicative only.

**`p15manual` was a FRESH run** -- its CSV starts at timestep 0. p12goal,
p13strike and p14aerial are therefore an **abandoned lineage**, not ancestors:
the current bot descends from p15manual. The ladder's "+478 Elo over p12goal" is
a comparison between two trees, not an increment along one.

| boundary | change |
|---|---|
| within p15manual (from 842M, when the metric first appears) | `AirTouch` 31 -> 20, `ShotOnTarget` 5.1 -> 12.1, `FlipSpeed` 1.8 -> 2.5 |
| **p15manual -> p16** | **`ShotOnTarget` 12 -> 35 (2.9x)**, `AirTouch` 20 -> 12 |
| **mid-p16, ~1735M** | **`AirTouch` 12 -> 75 (6.3x)**, `ShotOnTarget` 35 -> 20, `Air` 0.88 -> 1.88 |
| p16 -> p17 | `AirTouch` 75 -> 55, `ShotOnTarget` 20 -> 32, `Save` added at ~22 |

p15manual also shows the manual tweaking directly: `Entropy Scale` was moved
0.0021 -> 0.0040 -> 0.0050 -> 0.0040 -> 0.0020 -> 0.0088 across the run, and
`Episode/Mean Steps` ballooned 622 -> 1023 -> 1846 -> **2813** before collapsing
back to 462, i.e. a farm appeared and was killed mid-run.

### The reframe: the policy was never frozen, and the instruments could not see it

Converting the ladder to a rate against the steps each run actually took:

| interval | steps | Elo gained | rate |
|---|---|---|---|
| p15manual -> p16 | 1237M | +273 | **22 Elo / 100M** |
| p16 -> p17 | 310M | +13 | **4.2 Elo / 100M** |
| p17 -> p18 | 483M | +29 | **6.0 Elo / 100M** |

**p18entropy's 483M gained +29 Elo against a ladder noise floor of ~13, so the
run that produced "behaviour did not move on any axis" was in fact improving the
whole time, at about 5 Elo per 100M steps.** The behavioural metric set cannot
resolve that; a calibrated ladder can. The standing p17/p18 conclusion -- "PPO
has converged on its objective" -- is **too strong**, and so is the version of it
written into the p18entropy-extended row above. What actually happened is a
deceleration from 22 to ~5 Elo/100M, which is the ordinary shape of a learning
curve, not a stop. **This does not retire the opponent-symmetry hypothesis; it
changes what a p19pool null result would mean**, since ~5 Elo/100M is the
baseline any intervention now has to beat rather than zero.

**Four of the seven ladder rungs are not in this table.** p13strike, p14aerial,
p15manual and p16 have no row, and the 2026-08-23 ladder measures p15manual and
p16 as **the two largest gains in the project's history (+174 and +273 Elo)**.
Their `CONFIG_HISTORY.json` files carry a single entry with an empty `changed`
map, so what they altered is not recorded anywhere. The log documents its
failures in extraordinary detail and omits its successes; reconstructing those
two runs is now a prerequisite for understanding where the gains came from.

**Deferred work lives in `.scratch/run-backlog/`** -- eleven tickets covering the
possession term, the air game, the zero-sum k=1 question, gamma, and the
positioning blind spot. **Read `PLAN-post-p18.md` first**: p18entropy ran to
483M and unblocked them, so the ordering in `spec.md` is superseded. Nothing is
readable until the anchored ladder in Run 0 exists.

| Date | Label | Config delta | Why | Outcome |
|---|---|---|---|---|
| 2026-08-23 | **LADDER** (no training) | Not a run. `scripts/ladder.py` + a repaired `HivemindBot eval`: p18 vs six frozen ancestors, 400 games per side per rung, both sides swapped, stochastic inference, run under BOTH the training spawn (`RandomState`) and kickoffs. 187 h of simulation | `Rating/1v1` is an unanchored Elo (random-walk sigma 42.8 vs an observed 29.8 drift), so no run conclusion has ever been anchored to anything outside itself. Frozen opponents cannot move to meet the challenger, so goal share has a real null of 0.5 | **The lineage works, and the plateau is precisely dated to p16.** Match scores and Elo gaps (p18 minus rung, in-distribution): p13strike **+593**, p14aerial **+526**, p12goal **+478**, p15manual **+304**, p16 **+31**, p17 **+18**, SELF-control **-11**. Per-run increments: p12goal->p15manual **+174**, p15manual->p16 **+273**, p16->p17 **+13**, p17->p18 **+29**. **SELF-control reads 49.3% / 50.5% goal share across the two spawn modes (truth 50%), so the apparatus is calibrated and the noise floor is about +-13 Elo.** Kickoff and in-distribution ladders agree throughout (p12goal 89.2% vs 92.2%, p16 53.9% vs 55.6%), so the out-of-distribution kickoff start is NOT what the earlier confusion was about. **Three things this settles.** (1) The operator's judgement that the bot improved enormously since p12goal is **confirmed and quantified: +478 Elo**, and the branch in `PLAN-post-p18.md` that would have justified a restart-on-stagnation is dead. (2) **The plateau is real but recent**: p17 and p18 together are ~800M steps and ~9.5 h of compute for **+42 Elo**, against p16's +273 alone, and p17's +13 is at the noise floor. (3) **The ladder independently recovers a known fact, which is the best evidence it works**: p13strike and p14aerial score BELOW p12goal (+593 and +526 against p18, versus p12goal's +478), i.e. both were regressions -- exactly what `RUNLOG` says of p13strike, whose steer channel went extinct at 0.0006. Nobody told the ladder that. **Process, and it is the uncomfortable part: `eval` had been broken since it was written, in two independent ways, and produced plausible scorelines throughout.** (a) `GameState::UpdateFromArena` orders players by `arena->_cars`, which is NOT `AddCar` order -- `players[0]` is the ORANGE car -- and `Eval.cpp` indexed positionally while holding `blueCar`/`orangeCar` pointers, so **each car was driven by the action computed for the other car**. An `assert` guarded exactly this and `scripts/build.sh` builds `Release`, so `NDEBUG` removed it. (b) `RelativeObs.cpp:63` puts `prevAction` in the observation and `Eval.cpp` rebuilt `GameState` from the arena every step, zeroing those 8 of 109 floats. **CLAUDE.md's claim that "RelativeObs is stateless for exactly this reason" is false.** Validation that now gates the tool, against training telemetry: `Ball Touch Ratio` **0.0001 -> 0.0136** (training 0.0157), episode length **13.7 s -> 25.8 s** (training 24.1 s), goals per episode **0.08 -> 1.00** (training ~0.99). **Training, the skill tracker, `spectate` and the RLBot deployment path were all checked and are unaffected** -- each stays inside one index space or walks `_cars` in lockstep; `eval` was the only code mixing the two. No prior RUNLOG conclusion used `eval`. **Also measured: the training goal rate is ~1 per 24 s and ~99% of episodes end in a goal**, derived from `RewardMass/Goal` 0.09316 / weight 34, and confirmed independently by the repaired eval |
| 2026-08-23 | p18entropy (extended) | **NOT one variable, and not intended: the resume dropped the command-line flags.** `--entropy-target 0 --entropy 0.002` were CLI-only on 2026-08-21; the 08-23 resume omitted them, so `Config.h` defaults (`entropyTarget 0.40`, `entropyScale 0.003`) silently reapplied and the entropy controller came back on. Recorded in `CONFIG_HISTORY.json` entry 1 at ts 2,641,746,944. Reward stack, obs, spawns, mask, LR, gamma, tsPerItr all still frozen at p17 values. 433M further steps, taking the label to **483M total (2592M -> 3075M cumulative)** | Spare compute. Not designed as an experiment; it became the longest single stretch this project has run, and by accident it tests the freeze at a SECOND entropy level | **The strongest plateau evidence this project has, and it did not need the confound to be clean.** The dropped flags mean the freeze is now measured at entropy **0.285 (controller off, 50M)** and at **0.400 (controller on, 433M)** -- 26% below and 26% above p17's operating point -- with **identical behaviour in both**. Segment-matched: `Touch/Above 450` +10.1% (off) vs +10.6% (on), `Phase/Aerial` +6.2% vs +4.7%, `Shot/Distance` +0.2% vs -0.6%, `Shot/Saved Share` -3.4% vs -0.4%. **Split into nine ~54M chunks, exactly ONE metric trends monotonically across all nine, and it is not behavioural: `GAE/Returns STD` 41.62 -> 40.89** (-1.8%, 153 sigma, the largest trend-to-noise in the entire metric set). `Critic/V All` +2.8% and `Critic/V Airborne` +8.0% follow it. **Every behavioural metric's net change is smaller than its own within-run oscillation**: `Touch/Above 450` net +10.6% against a 14.2% swing, `Player/Ball Touch Ratio` -1.2% against 7.4%, `Save/Converted` -1.5% against 2.0%, `Shot/Distance` +0.2% against 1.6%. **The decisive new fact: `Average Step Reward` is flat too -- 0.1090 -> 0.1098, +0.7% net against a 2.6% swing.** p17 could be read as "the reward is being optimised but the behaviour it buys is wrong"; at 483M that reading is dead. PPO is not stuck, it has **converged on its objective**: clip 0.0342-0.0375, KL 0.0037-0.0040, update magnitude 0.068-0.072, `Obs/Non-Finite Rate` 0.000, constant to 3 s.f. throughout. **The one axis with residual movement is the one large NON-zero-sum term.** `RewardMass/AirTouch` +5.9% (5.4 sigma) with `Touch/Above 450` +10.6%, `Player/Touch Height` +2.6%, `Phase/AirDribble` +4.6%. By the k-arithmetic, `Goal` (17.4%) + `ShotOnTarget` (11.2%) + `Save` (4.4%) + 0.8x`TouchGoalAccel` (24.6%) = **57.6% of reward mass has zero expected population advantage in symmetric self-play**. The policy is still climbing the ~42% that behaves like a solo objective, and `AirTouch` at 13.3% is its largest component -- the same term `.scratch/run-backlog/issues/05` says pays 1.33x the entire finishing block per episode. **`Rating/1v1` 424 -> 394 is NOT evidence of decline**: 41 samples, per-sample sigma 6.77, random-walk sigma over 40 updates = **42.8**, so a 29.8 drift is well inside chance for an unanchored Elo whose pool inherits the current rating. Item 11 predicted exactly this -- the instrument cannot answer the question it is being asked. **Throughput: 64,112 steps/s mean WITH `--track-skill`**, so CLAUDE.md's 52k planning figure is 23% stale; 483M steps took 3.4 h wall-clock. **Process:** the resume also forked wandb (a new run `ciygwkg4` alongside the original `hp16vami`) and the CSV lost its first 496 iterations; both repaired by the new `scripts/merge_wandb_runs.py`, see below |
| 2026-08-21 | p18entropy | **One variable.** Reward stack byte-identical to p17; `--entropy-target 0 --entropy 0.002`, disabling the controller and pinning `entropyScale` ~7x below the 0.0144 it had drifted to. Seeded from `main-p17/2591724288`, 50M steps | Discriminator, not a reward run: p17 proved a converged policy stays converged but not WHY. Either the entropy target was pinning it, or it is a genuine local optimum | **The pathology was real and fixing it changed nothing. 0 of 3 predictions on the letter; the mechanism confirmed, the conclusion inverted.** Entropy 0.389 -> 0.285 (-26.7%, 5.76 -> 3.61 effective actions of 90), crossing 0.30 at +27.8M against a predicted 25M. `Policy Relative Entropy Loss` **±940/±42 -> 0.37-0.92**: the entropy bonus was genuinely drowning the policy gradient and no longer is. `SB3 Clip Fraction` 0.0361 -> 0.0438 (+21.2%), the largest steps since p16 mid-life. **And behaviour still did not move**: `Shot/Distance` +0.2%, `Shot/Saved Share` -4.1% (continuing a drift already present in p17), `Save/Converted` 0.797 -> 0.796. No kill criterion fired. Entropy is ASYMPTOTING (per-decile drop -0.031 -> -0.004, a 7x deceleration), so this is a new equilibrium, not an unfinished transient |
| 2026-08-21 | p17 | **One concept, three coupled changes**: new signed `SaveReward` (change in threat at own net across a touch, same `ProjectShot` and `MISS_SCALE` as `ShotOnTarget`), zero-sum at k=1.0, budget 16.5 solved-then-capped; `shotOnTargetOpponentScale` 0.8 -> 1.0; `shotOnTarget` 22 -> 32 (forced by the cap). Seeded from `main-p16/2281639168`. Baseline budgets BACK-SOLVED from telemetry, not taken from HEAD | Stop paying for the shot attempt; pay for whether it survives. Aimed at possessions thrown away on hopeless long shots and clearances fed straight back | **CLEAN NEGATIVE, and the term was never tested. 0 of 5 predictions, 0 kill criteria fired, and the POLICY DID NOT MOVE.** Over 300M steps the largest behavioural change was `Flip/Neutral Share` 0.009 -> 0.020; `Shot/Distance` moved **0.0%** (4007 -> 4009), `Player/Velocity Alignment` 0.1%, `Shot/Toward Net Rate` 0.2%. `Shot/Saved Share` 0.313 -> 0.289, `Save/Converted` 0.789 -> 0.793, `Rating/1v1` 432 -> 425. PPO was in perfect steady state, not broken: clip 0.0354, KL 0.0038, update magnitude 0.0696, `GAE/Avg Advantage` 0.153, all constant to 3 s.f. across 300M. **The save term carried 4.4% of mass into a policy not responding to the other 96% either, so it is UNFALSIFIED, not falsified** -- as is the k=1 choice |
| 2026-08-19 | p12goal | **Three changes, one concept** (declared as not-one-variable): `StrongTouch` -> `TouchGoalAccel` (budget 3.0 carried over, signed goal-directed change in ball speed at contact); `Goal` 10.0 symmetric; `AirTouch` 2.0 = `min(airTimeFrac, heightFrac)`. Boost budgets, obs, spawns, mask, LR, entropy, tsPerItr, gamma all frozen. `infiniteBoostChance` stayed 0 (no `Infinite Boost Share` column; `Player/Boost` reads 5.5 at 10M). **Run to 250M, not 100M** -- deliberate overnight extension, stopped by hand | The guide's middle-stage gate: the bot can hit the ball, so tell it the net exists and pay for productive air | **2 of 5 predictions, BOTH kill criteria missed, and the best bot this project has made.** Scorecard: **(1) `Touch/Hit Force` > 700 -- FAILED, 801 -> 489 (100M) -> 422 (250M).** Third consecutive falling run: 878 (p10 peak) -> 551 (p11) -> 422. **(2) `Episode/Mean Steps` < 1200 -- CORRECT, 1788 -> 390 = 26.0 s.** Goals now end episodes; mean gap between contacts is 1/0.0291 = 34 steps = 2.3 s so `NoTouchCondition` almost never fires. **(3) `Touch/Above 450` > 0.15 and `Touch Height` > 260 -- FAILED AND INVERTED: 0.081 -> 0.037, 195 -> 155**, `Touch/Above 200` 0.195 -> 0.078. **(4) `Player/Boost` > 15 -- passed on the number (9.4 -> 25.3) and the mechanism is the OPPOSITE of the hypothesis.** `Action/Boost When Grounded` 0.72 -> 0.356 and `Boost When Airborne` 0.27 -> 0.197: the tank filled because the bot stopped SPENDING, and episodes are 4.6x shorter so mean boost is sampled far closer to spawn. **The p11 diagnosis is not confirmed, it is untested** -- it predicted boost rising *because* air touches did, and air touches fell. **(5) `Jump When Grounded Upright` > 0.02 -- FAILED, 0.0107 -> 0.00400, which is the eps-floor to three figures** (0.02 x 18/90 = 0.00400). Measured/floor = **1.00**: the cleanest extinction this project has recorded. **Both pre-registered kill criteria fired and the run continued anyway:** 25M wanted `RewardShare/AirTouch` > 0.02 and it read **0.0018** (11x below); 50M wanted `Touch/Hit Force` > 620 and it read **609.7**. **THE DECISIVE MEASUREMENT: the critic has priced the takeoff, under something close to a randomized trial.** `Critic/TD Delta Jump` -0.070 -> **-0.2249** against `NoJump` -0.0199, the gap widening monotonically. Read it carefully. (a) The metric is `gamma*V(s') - V(s)`, **not** a full TD residual -- `Train.cpp:139` omits the step reward -- so it is the critic's opinion of where the jump leads, not realized return. (b) At V ~ 1.5 and gamma 0.99, `gamma*V - V` is **-0.015 even with no value change**, and `NoJump` reads -0.0199, i.e. essentially that drift. The excess attributable to jumping is therefore **-0.205 standardized = -0.81 touch-units** at `GAE/Returns STD` 3.939. (c) **The comparison is near-causal**: the policy's own P(jump) is 0.0004 against an eps-floor of 0.0040, so ~91% of sampled jumps are floor-assigned, and the floor mixes uniformly over valid actions **independent of state** -- the fourth external patch doubles as random treatment assignment on the jump action. `Critic/V Grounded` 1.495 vs `V Airborne` 1.258 (0.93 raw) points the same way but is a **marginal, non-causal** comparison, confounded by which states are airborne (launch recovery, walls); it corroborates the direction, it does not independently measure the price. **AirTouch pays 0.37 against that 0.81.** Raw mean 0.000461/step / `Ball Touch Ratio` 0.0411 = 0.0112 per contact step = 0.0224 touch-units per touch; airborne touches are 6.1% of touches (In Air 0.181 x `Touch/Rate Airborne` 0.0138 against 0.819 x `Rate Grounded` 0.0471), so an airborne touch averages 0.184 x 2.0 = **0.37**. Even a textbook air dribble (ball z 800, 1.0 s aloft) pays 2.0 x min(0.571, 0.391) = **0.78 against a cost of 0.81.** **`airTouch = 2.0` was set 15-20% BELOW the indifference point**, which is exactly the signature of a behaviour that appears and decays rather than establishing or vanishing -- p10touch, p11 at 42-56M, and here not at all. **Third consecutive inert term.** RewardShare of the term the run was built around: p10 StrongTouch **0.037**, p11 SaveBoost **0.016**, p12 AirTouch **0.008**. The budget framework converts a *maximal* payout; the policy responds to *realized mass* = budget x mean realized value x rate, and nothing in the framework estimates the rate. **The possession equilibrium, with the ledger.** Per episode at 250M (390 steps), from `Rewards/<class>` x weight, reconciling to `Average Step Reward` 0.0714 within 4.5%: SpeedToBall **14.64 (55.0%)**, FaceBall 5.09 (19.1%), TouchEdge 2.76 (10.4%), TouchGoalAccel 2.47 (9.3%), SaveBoost 1.19 (4.5%), PickupBoost 0.41 (1.5%), AirTouch 0.36 (1.35%), Air 0.20 (0.8%); net ~26.6. **84.5% of the ledger pays for being near, pointed at and in contact with the ball; 9.3% pays for what the ball does.** SpeedToBall + FaceBall has held 0.61-0.88 of reward mass in **every run this project has ever done** (p8ref 0.761, p10 0.876, p11 0.778, p12 0.606; only the p9rel farm displaced it) and has **never once been reduced**. **Why the touches are soft, and it is not a bug.** `TouchGoalAccel` is LINEAR in goal-directed dv, and the total goal-directed dv required to score is fixed by the length of the field, so a linear term is **indifferent between one 2000 uu/s strike and five 400 uu/s pokes** -- it pays the same. The other 84% is not indifferent: a poke leaves the ball inside re-contact range and a strike does not. Measured: 11.3 contacts per episode, 1.41 steps per contact sequence (so **not** a carry farm -- p9rel was 1.98), `Touch/Had Flipped` **0.0008**, i.e. the bot never flips into the ball. The guide's own words are "a slight push that barely changes the velocity of the ball will give almost no reward", which describes a **convex** function of dv; this project implemented it linear. **gamma bounds what the goal term can do.** At `gaeGamma` 0.99 and 15 steps/s the value horizon is 1/(1-g) = 100 steps = 6.7 s, so a goal is worth 10 x 0.99^390 ~ 0.2 at the start of a 26 s episode and 7.4 two seconds out: **Goal shapes the finish and nothing else**, and its 14.9% of |reward mass| is by construction zero-mean variance. Consistent: `Critic/V All` 3.32 -> 1.46 standardized while `GAE/Returns STD` 2.11 -> 3.94, i.e. raw V 7.0 -> 5.75 against much noisier returns. **Real gains, and they are the best this project has produced.** `Rating/1v1` **7.4 -> 108.1** (p11 6.1, p10 25.6); `Player/Ball Touch Ratio` 0.0248 -> **0.0411**; `Touch/Edge Rate` 0.0208 -> 0.0291; `Game/Goal Speed` 1653 -> **1947**; `Episode/Late/Ball Dist` 817; `Surface/Wrong Contact Rate` 0.0065; `Action/Handbrake` 0.140 -> 0.015; `Speed/Above Throttle Cap Share` 0.130 -> 0.389. **First run in which the bot plays a recognisable game of Rocket League**: it approaches, keeps possession, and scores about once every 26 s. Rating caveat stands -- it is against this run's own version pool, so it measures self-improvement, not absolute skill. **Nothing after ~175M was worth having.** `Policy Entropy` 0.309 (100M) -> 0.167 (150M) -> 0.146 (250M); `Rating/1v1` flat and noisy from 175M (117, 117, 91, 86, 98, 108). PPO stayed healthy the whole way (clip 0.068-0.149, KL 0.0074, `Policy Relative Entropy Loss` 1.01 -> 0.12, Critic Loss flat, `Obs/Non-Finite Rate` 0.000): **the run did not break, it converged.** The extra 150M bought a doubling of self-play rating between 100M and 175M and then nothing. **Latent farm shipped but never fired:** `AirTouchReward` pays on every contact STEP, not on the rising edge, unlike `TouchEdgeReward` which exists for exactly that reason. It stayed inert so nothing happened, but **it cannot be raised until it is edge-gated** -- at the budget the break-even analysis calls for, an air carry at ceiling height would pay ~200 touch-units per second |
| 2026-08-19 | p11boost | Boost economy (`SaveBoost` 1.5 sqrt-of-tank, `PickupBoost` 0.5) + `strongTouch` 1.0 -> 3.0 re-derived from p10touch's `Touch/Strong Value` = 0.104. Two changes, declared as such. First attempt died at 29.8M to a NaN (see CLAUDE.md, fifth/sixth patches); rerun fresh to 97.7M | Does paying for boost fix the air game that p10touch found and could not sustain? | **FAILED, and the reason reframes the whole air problem.** Scorecard: **(1) `Player/Boost` >25 -- FAILED, 9.4** (from 7.4; the 25M kill criterion of >15 also failed at 6.4 and the run should have been stopped there). **(2) `Touch/Above 450` >0.15 -- FAILED, 0.081**, peaking 0.094 at 54M then declining. **(3) StrongTouch share 0.08-0.15 AND hit force stops falling -- HALF: share 0.105 in range, but `Touch/Hit Force` 878 -> 551, still falling.** That was the pre-registered tail-chasing failure and it fired. **(4) floor jumping <0.015 -- correct, 0.0044.** **The air game emerged and decayed AGAIN, and this time it is dated:** at 42-56M `Jump When Grounded Upright` 0.0058 -> **0.0116**, `Takeoff Was Jump` 0.0095 -> **0.069**, `In Air Ratio` 0.151 -> **0.257** -- then all three decayed back (0.0044 / 0.032 / 0.192). Second time this has happened (p10touch was the first). **Why the boost economy could not work: there is nothing to save boost FOR.** `Action/Boost When Grounded` **0.719 and rising** against `Action/Boost When Airborne` **0.287 and falling** (from 0.562). The bot spends boost on the ground because boosting there feeds `SpeedToBall`, which holds **57%** of reward mass against SaveBoost's **1.6%** -- spending is worth 36x saving, measured. SaveBoost asks the policy to hoard a resource with no privileged use. **Boost was never the constraint on air play; it was downstream of one.** Nothing in the stack paid more for an air touch than a ground touch, so a ground bot was genuinely optimal and the policy was playing it correctly. **The poke farm:** the rising edge stopped the CARRY farm (steps per contact 1.22, healthy) but not repeated brief weak contacts -- `RewardShare/TouchEdge` doubled 0.033 -> 0.068 while mean hit force fell to 551, i.e. **below StrongTouch's own 555.6 floor, so the average touch earned exactly zero from it.** Touching was paid; touching usefully was not. Real gains: `Player/Velocity Alignment` 0.542 -> **0.718**; touch edge rate 0.0077 -> 0.021; `Rating/1v1` -9.4 -> 6.1. `Obs/Non-Finite Rate` 0.000 throughout, so the NaN guard's counter stayed clean. **Process note: the metrics CSV had both runs concatenated** -- `--fresh` archived checkpoints but not the CSV -- and the first trend read off it showed a policy mysteriously resetting a third of the way through. Caught by an entropy jump 0.60 -> 0.71, which does not happen. Fixed; the files are now split |
| 2026-08-18 | p10touch | **One variable vs p9rel:** flat per-step `TouchBallReward` (1.0) -> `StrongTouchReward` (1.0, |delta ball vel|, zero below 555.6 uu/s) + `TouchEdgeReward` (0.25, rising edge). Obs, dense budgets, spawns, mask, LR, entropy, tsPerItr all frozen | Kill the dribble farm and re-establish a sane ledger | **FARM DEAD, BEST RUN YET, AND MY JUMP METRIC WAS WRONG.** Scorecard: **(1) steps per contact sequence <1.3 -- CORRECT, 1.211** (p9rel 1.98, p8ref 1.16). **(2) `Episode/Mean Steps` <1500 -- FAILED: 1734 and still RISING** (823 -> 1191 -> 1734). Episode length is NOT purely downstream of the farm and needs its own cap. **(3) alignment >0.60 -- CORRECT and best ever: 0.7465** (p9rel 0.528, p8ref 0.593). The relative obs finally shows its value now that carrying the ball no longer pays. **(4) jump stays <0.008 -- technically correct (0.00790 Upright) BUT THE METRIC WAS HIDING THE ANSWER.** `upright` is `rotMat.up.z > 0.7`, so a car driving on a WALL is grounded-and-not-upright and its jumps were being logged to a bucket named 'Inverted' that I read as upside-down recovery and never quoted. That bucket reads **0.0615 = 15.4x the eps-floor and rising** (0.0135 -> 0.0601), with `Player/Takeoff Was Jump` **0.025 -> 0.264** and `Action/Jump When Airborne` 0.0152 -> 0.0726. **Jumping is not extinct; FLOOR jumping is.** Renamed to `Action/Jump When Grounded Tilted` with `Player/Grounded Tilted Ratio` published alongside it. **(5) `Touch/Strong Value` = 0.104**, against the 0.3-0.4 predicted -- so StrongTouch earned 0.104/touch against TouchEdge's 0.25, arriving paid 2.4x more than connecting, and `RewardShare/StrongTouch` sat at **0.037**, effectively inert. Budget re-derived to 3.0 per the pre-registered rule (move the budget, not the curve). **The air game is real and emerging on its own:** `Touch/Above 450` **0.051 -> 0.1095** (p9rel 0.00137, p8ref 0.035); `Player/Touch Height` 162 -> **220**; `Touch/Rate Airborne` 0.0046 -> 0.0154; `In Air Ratio` 0.24. Watched in RocketSimVis: air-dribbling off the wall when it has boost, never jumping from the floor -- which matches the Upright/Tilted split exactly. **The binding constraint is now boost: `Player/Boost` 7.3 out of 100 all run**, and nothing in the stack has ever paid for it. Aerial play is boost-gated in a way ground play is not. Secondary: `Rating/1v1` 3.5 -> **25.6**; `Game/Goal Speed` 1120 -> 1533; touch edge rate 0.0077 -> 0.021; PPO healthy throughout (entropy 0.753 -> 0.354, clip 0.107-0.136). Dense terms still hold 0.877 of reward mass, which is the early-stage shape and this bot is past it |
| 2026-08-18 | p9rel | **One variable vs p8ref:** `TrainConfig::obs` Default -> `RelativeObs` (car-frame `dirToBall` unit vector, distance, offset and closing velocity per body; 89 -> 109 floats). Rewards, spawns, mask, LR, entropy, tsPerItr, `REFERENCE_EPISODE_STEPS` all frozen | Test whether representation cost is what keeps the bot playing a 2D game | **THE AERIAL HYPOTHESIS IS FALSIFIED, AND THE OBS UNLOCKED A DRIBBLE FARM.** Pre-registered scorecard: **(1) alignment >0.65 -- FAILED** at 98M (0.528), though it PEAKED at **0.649 at 18M vs p8ref's 0.486 at the same step count**, so the obs plainly worked before the farm buried it; carrying the ball drives closing speed to ~0, which is what the metric measures. **(2) touch ratio >0.018 -- passed 7x (0.128) and is meaningless.** **(3) jump stays <0.008 -- CORRECT: 0.00387, which is 0.97x the eps-floor**, even more extinct than p8ref's 1.11x. **(4) `Touch/Above 450` >0.05 -- FAILED, and INVERTED: 0.0525 -> 0.00137, a 38x collapse**, with `Touch Height` 166 -> 125 and `Touch/Above 200` 0.176 -> 0.015. **Conclusion: representation cost is NOT what kept the bot 2D. That was my hypothesis, it was labelled as one, and it is dead.** The air-touch reward is now the only untried lever. **The farm, measured:** steps per contact sequence (`Ball Touch Ratio` / `Touch/Edge Rate`) **1.16 -> 1.98**; `Player/Ball Touch Ratio` **0.129**, i.e. the car is in contact with the ball on 13% of ALL steps; `RewardShare/Touch` 0.145 -> **0.741**, three quarters of the budget. Causal story is clean: the relative obs made the bot competent enough to CARRY the ball, and a per-step touch reward pays for carrying. **This project had already solved this and the port removed the solution** -- `TouchEdgeReward` existed for exactly this, and roadmap D4 says 'no dribble/possession reward terms, ever'. Traded away for fidelity to the reference. In fairness the guide predicts it precisely: *'The default touch part of EventReward is not very good once your bot can touch the ball... ball touches can easily be farmed by constantly pushing the ball'*. **So this is the early->middle stage gate arriving on schedule: the early-stage stack succeeded and exhausted itself.** **Structural finding: episode length is now an unbounded free variable.** `Episode/Mean Steps` 812 -> **2956 = 197 seconds**. `NoTouchCondition(12s)` can never fire on a bot that is always touching, and the bot cannot score, so nothing ends an episode. Consequence beyond bookkeeping: **RandomState resets happen 17x less often than designed**, so the spawn diversity the whole reproduction rests on is largely gone and the bot lives in self-induced dribble states. `REFERENCE_EPISODE_STEPS = 171` is now 17x stale, so no rate budget means what it says. Left frozen deliberately -- re-deriving it is a reward change in disguise and would break comparability mid-sequence. Secondary: PPO healthy throughout (entropy 0.783 -> 0.368, clip 0.114-0.154, all kill criteria passed); `Player/Speed` 1046 -> 1302; `Episode/Late/Ball Dist` 1834 -> 613; `Rating/1v1` 2.4 -> 13.9; `Touch/Had Flipped` **0.00027**; `Player/Boost` 14.5 |
| 2026-08-18 | p8ref | **Phase C reproduction** per `docs/superpowers/specs/2026-08-18-known-good-baseline-design.md`: Zealan's early-stage stack ported (Touch 1.0 per step, SpeedToBall 17.1, FaceBall 3.42 **SIGNED**, Air 0.513, all touch-units); Goal and WrongSurface deleted; `RandomState(true,true,false)` replaces the curriculum; action mask OFF; LR 3e-4 -> 2e-4; tsPerItr 100k -> 50k; entropyScale 0.01 -> 0.002 | Establish a known-good reference point. 25 experiments and every one measured against another failure | **ALL FOUR PRE-REGISTERED GATES PASSED, and it is the best result this project has produced.** `Player/Ball Touch Ratio` **0.00121 -> 0.01428** at 98M -- better than p1air's 0.0131 at **245M**, so 2.5x the result at 2.5x the pace, from a stack containing nothing this project invented. **Velocity alignment left its null for the first time ever: 0.3745 -> 0.5932** against the 0.3183 planar null (every prior run: flat at 0.30). `Speed Towards Ball` 383 -> 738. `Episode/Late/Ball Dist` 3413 -> **953**. Gates: health at 10M PASS (entropy 0.776 -> 0.49, clip 0.073-0.13 inside [0.02,0.25], `Policy Relative Entropy Loss` 0.076 vs the <=0.1 target); 25M alignment >0.35 PASS (0.486 at 18M); 50M alignment >0.45 + touch >=0.004 PASS (0.564 / 0.0106, 2.6x the gate); 100M touch >=0.008 PASS (0.0143, 1.8x). **The Early/Late collapse did not just close, it INVERTED** -- Late 0.01501 vs Early 0.001417, i.e. Late is now 10.6x HIGHER. Read that carefully: `Episode/Mean Steps` went **226 -> 1137**, so 'Late (4s+)' now spans 4-75 s and Early is 1/75th of the episode starting 3902 uu out. The bot closes and stays closed; the bucket comparison is no longer measuring what it measured at 171 steps. **JUMPING IS EXTINCT, and for the first time the extinction is quantified rather than inferred.** `Action/Jump When Grounded Upright` 0.1797 -> **0.00444**. The eps-floor supplies `0.02 * 18/90 = 0.00400` of that, so the **policy's own P(jump) is 0.00045** -- four in ten thousand. Measured/floor = 1.11. Every jump observed in the visualiser is forced exploration, not a decision, which is exactly why they accomplish nothing. `Player/In Air Ratio` 0.721 -> 0.120, `Takeoff Was Jump` 0.188 -> 0.0070, `Touch/Had Flipped` 0.570 -> **0.0026**. **Why: the ledger says air costs ~6x what it pays.** Per-step earnings reconcile to `Average Step Reward` 0.0578 within 1.4% -- SpeedToBall 0.0317 (share 0.545), FaceBall 0.0114 (0.243), Touch 0.0135 (0.206), Air **0.00036** (0.0064). Going airborne forfeits FaceBall (~0.0114/step, no steering) and halves contact (`Touch/Rate Airborne` 0.0077 vs `Grounded` 0.0152, so ~0.0075/step) against an Air payment of 0.00036/step. **Air would need ~7x to break even, which would put 37% of the budget on floating** -- p1air's `grounded` attractor inverted. The guide's 'if your bot stops jumping, increase the AirReward' is calibrated for a bot whose alternative is sitting still, not one driving at 1170 uu/s with alignment 0.59. **The wall farm is real and the guide predicted it.** Observed in RocketSimVis driving the ball up the wall to reach high balls while staying grounded; `Touch/Above 450` 0.035, `Player/Touch Height` 149. This is the guide's named 'plat wall-shot' failure, whose stated fix is the air-touch reward `min(air_time_frac, height_frac)`. Secondary: `Rating/1v1` 4.8 -> 22.7 rising; `Game/Goal Speed` 1028 -> 1295; `Surface/Wrong Contact Rate` 0.039 -> 0.005; `Action/Steer Nonzero` 0.602 -> 0.479 against a 0.5333 unmasked null -- below null and drifting, worth watching but not extinction; `Player/Boost` 9.9 -> 13.5, still dry, and there is no boost term by design. **Open item: `REFERENCE_EPISODE_STEPS` is now wrong by 6.6x** (171 vs a measured 1137). It is deliberately NOT updated, because doing so divides every rate weight by 6.6 and is a reward change in disguise |
| 2026-08-18 | p7approach | Reward stack cut 8 terms -> 5 (Goal 1.0, SpeedToBall 0.50, FaceBall 0.05 RECTIFIED, Touch 0.30, WrongSurface 0.10/s); `REFERENCE_EPISODE_SECONDS` 10 -> 11.4. **Second variable: `entropyScale` 0.004 -> 0.01** | Run B: does a rectified approach term (Zealan's `SpeedTowardBallReward`) beat p6budget's factored Speed^2 + FaceBall? | **UNINTERPRETABLE AS A REWARD TEST, AND THE STACK'S ARGMAX IS THE MEASURED FAILURE.** (1) **PPO never moved.** `Policy Entropy` 0.7089 -> 0.6892 -- flat over 100M steps, ~21 of 90 effective actions, i.e. near-uniform; `SB3 Clip Fraction` 0.0078 -> 0.0038 against a healthy 0.05-0.2; `Mean KL` 6.8e-4; `Policy Relative Entropy Loss` 5.7 -> **268** against docs/metrics.md's stated healthy range of <=0.1. This is the p1probe-d/-g failure exactly, and the RUNLOG already said **'Keep 0.002'**. entropyScale 0.01 was shipped anyway, in the same run as a full stack rewrite. No conclusion about the reward stack is available from this run. (2) **`Player/Velocity Alignment` = 0.3135 -> 0.3121 -> 0.3135.** The analytic null for a uniformly random direction is **1/pi = 0.3183** in the ground plane and **0.25** in 3D (p6budget's split: grounded **0.286**, BELOW the planar null; airborne 0.301). **The bot's velocity direction is at chance with respect to the ball, in every run this project has done, while 42% of reward mass pays for exactly that quantity.** The null was never computed, so 0.30 was read as low-but-real rather than as nothing-learned. (3) **The policy converged to jump-and-don't-turn, which is the stack's argmax.** `Action/Steer Nonzero` 0.160 -> **0.087** (third steering extinction in this project); `Jump When Grounded Upright` 0.755 -> 0.878; `Takeoff Was Jump` 0.756 -> 0.865; `Leave Ground Rate` 0.450 = ground dwell **0.15 s**; `In Air Ratio` 0.918 -> 0.929. Mechanism: every dense term is rectified BY DESIGN ('retreating is free', 'facing away pays nothing and costs nothing'), `WrongSurface` fires on 4.6% of steps, and Goal is ~zero-mean noise -- so **no state the bot can enter is ever penalised**, and the only thing that costs anything is turning, because turning scrubs the speed that `max(0, v.dirToBall)` pays for. Rectifying FaceBall removed the one term (rlgym's `FaceBallReward` is SIGNED) that held the nose on target. (4) **Early/Late collapse unfixed since p1age:** Early(0-1s) touch 0.0081 vs Late(4s+) 0.00025 = **32x**; approach speed 931 -> 181; ball distance 2722 -> **4353**, i.e. it ends farther away than it starts. (5) `RewardShare`: SpeedToBall 0.42, **WrongSurface 0.30**, Goal 0.12, FaceBall 0.09, Touch 0.08 -- a tuning penalty is the second-largest term, against the guide's 'reduce or remove tuning rewards'. (6) Goals are ~0.116/episode and 49% of them come from `Scenario/Defend` (`EndedInGoal` **0.87** -- the defender concedes 87% of the time), so the goal-unit budget framework is denominated in a currency the telemetry cannot audit. `Rating/1v1` pinned at -9.79 |
| 2026-08-18 | p6budget | Entire reward stack replaced per `docs/superpowers/specs/2026-08-18-reward-redesign-design.md`: eight goal-referenced budget terms (Goal 1.0, Speed^2 0.30, FaceBall 0.20, FaceBallAxis 0.067, Touch 0.15, CleanLanding 0.10, WrongSurface -0.10/s, HarshSpeedLoss -0.10). Every ball-directed velocity term deleted. gamma/lambda unchanged (run A of two). **Second variable: tsPerItr 100k here vs 250k in p5goalpot** | Run A: does a car-control reward beat a ball-directed one at equal step count? | **FAILS the primary criterion, and the mechanism is measured, not inferred.** Touch ratio **0.00127** at 100.0M vs p5goalpot's **0.00283** at 100.2M -- 2.2x worse (the RUNLOG's 0.0021 for p5goalpot was mid-run). `Touch/Edge Rate` 0.00111 x `Episode/Mean Steps` 171 = **0.19 touch events per episode**, so ~80% of episodes never touch the ball. **The decisive measurement is the nose/velocity split:** rectified nose-to-ball cos ((`FaceBall/Mean Cos` + `FaceBall/Axis Share`)/2) rose **0.338 -> 0.741**, while rectified velocity-to-ball cos (`Speed Towards Ball` / `Player/Speed`) went **0.299 -> 0.300 -- dead flat**. Per-episode ledger, reconstructed from `Rewards/<class>` x weight x `Episode/Mean Steps` (cross-checks against `Average Step Reward` to 0.4%): net **0.116 -> 0.351 gu/ep**, of which FaceBall 0.018 -> **0.158** and FaceBallAxis 0.044 -> 0.060. **The two facing terms are 62% of net earnings and 66% of the whole run's ledger improvement, and they bought exactly zero approach.** **Why: rotation is free, and no term in the stack prefers the ground.** `In Air Ratio` **0.898**, ground dwell 1/0.397 = **2.5 steps = 0.17 s**, `Takeoff Was Jump` 0.885, `Jump When Grounded Upright` **0.815** vs 0.429 uniform prior. Not flip-spam -- `Flip/Diagonal Share` **0.201 -> 0.0066**; it is **jump-and-hang**, ~7 takeoffs/episode at ~1.6 s per stint. `Grounded Speed` 935 vs derived airborne 1122 means **Speed^2 pays 1.44x more per airborne step**. `Surface/Wrong Contact Rate` 0.0205 against In Air 0.898 = the air tax covers **2.3% of airborne steps** (free flight has no chassis contact), collecting -0.023 gu/ep against CleanLanding's **+0.046** -- landing pays 2x what wrong-surface costs, the inverse of D7's asymmetry check, which priced 3 s on the roof, a state the bot never enters (`Landing/Clean Share` 0.989). Velocity alignment is ~0.30 **both grounded (0.286) and airborne (0.301)**: air time is not making it worse, it is uniformly bad. That air time is what *prevents* the heading -> velocity conversion is an **inference, not a measurement** -- run B instruments it. Secondary: Early/Late touch collapse **14.7x** (0.0078 / 0.00053), unchanged from 20x at 5M; `Episode/Late/Ball Dist` **3997** vs Early 2734, it ends farther away than it starts (p5goalpot closed to 1.35x and 1557). Goal production flat -- `Kickoff/EndedInGoal` **0.000** at 100M (p5goalpot 0.169), Strike 0.075 (0.143), NeutralPlay 0.076 (0.144). `Player/Boost` **11.7**/100 (p5goalpot 16.0), so D12's boost open item has fired: the bot is dry and cannot convert heading into velocity with boost either. **Not a PPO problem:** entropy 0.638 -> 0.287 falling under its own steam, Clip Fraction 0.0078 -> 0.044, KL 0.0042, `Clipped Reward Portion` 0 throughout, `Rating/1v1` -9.58 (30M) -> -2.19 (90M) -- improving, not p1-validate's farming signature. **Pre-committed criteria, worked literally:** jump canary 0.815 >> 0.011 PASS (but the criterion is one-sided -- it guarded extinction, not the reverse); `RewardShare/Speed` 0.202 < 0.25 PASS; D8's second trigger does **not** fire (`Speed/Above Throttle Cap Share` +22% while `Touch/Edge Rate` +46%), so the above-floor contingency must **not** be applied; `Surface/Wrong Contact While Flipping` **0.0255** vs the 0.25 the flip tax was priced at -- PASS, 10x cheaper, flipping is free; `Episode/Mean Steps` **171.0**, so `REFERENCE_EPISODE_STEPS = 150` under-states the episode by 14% and every rate budget over-delivers by that much; **`Speed/Max Step Decel` cannot answer D10** -- `Train.cpp:272` uses `report.AddAvg`, so the metric named "Max" is a mean (22.4 uu/s). T = 400 is neither validated nor refuted; only `Speed/Harsh Loss Rate` 0.0067 is known. **Attribution caveat:** tsPerItr differed from p5goalpot (100k vs 250k). Gradient steps are near-identical (997 x 2 x 4 = 7,976 updates vs 400 x 2 x 10 = 8,000, same 25k minibatch), so it is unlikely to carry a 2.2x gap, but it is a second changed variable |
| 2026-08-18 | p5goalpot | VelBallToGoal -> `BallGoalProgressReward` (potential on `Phi = -dist(ball, their goal)`, gamma=1, NOT ZeroSum-wrapped) **and** VelPlayerToBall restored to the p1air form (GroundedReward(VelocityPlayerToBall)). epsilon-floor kept. Fresh run to 100M | p4pbrs proved a car-to-ball potential telescopes to nothing; put the potential on a quantity the bot's own success improves instead | **PARTIAL, and the credit almost certainly does not go to the new term.** Real gains: touch ratio 0.0008 -> **0.0021** and still climbing; **`Episode/Late/Touch Rate` 0.0002 -> 0.0021, closing the Early/Late collapse from 27x (p3strike) to 1.9x** -- the bot recovers within an episode instead of stranding itself; `Speed Towards Ball` 260 -> 461; `Action/Handbrake` **0.753 -> 0.319**, i.e. the indifferent drift bit resolved on its own once contact became frequent enough to reveal its cost; steer healthy at 0.45, throttle-forward 0.93. **But `RewardShare/VelBallToGoal` is 0.0074 -- the new potential carries 0.7% of reward mass and is effectively inert**, while `RewardShare/VelPlayerToBall` climbs 0.223 -> **0.451**, the familiar approach-farm signature. Two variables moved in one run, so attribution is not clean, but the mass numbers point at the restored p1air term, not the potential. **Still 3x worse than p1air at the same step count** (p1air ~0.0057 at 95M vs 0.0021 here), `Jump When Grounded` 0.011 = pinned at the epsilon-floor, `Touch/Had Flipped` 0.419 -> 0.020, `Touch/Above 200` 0.229 -> 0.067. Air game absent |
| 2026-08-18 | p4pbrs | VelPlayerToBall -> `BallProgressReward` (potential-based on `Phi = -dist(car, ball)`, gamma=1, ungated) + 4th external patch: epsilon-floor mixing the policy with uniform over VALID actions (eps 0.02) | Three runs had each lost a control dimension to extinction; and every shaping term in the stack was farmable or penalised the corrective action | **Both mechanisms worked; the run still failed.** epsilon-floor: `Action/Steer Nonzero` **0.0006 -> 0.62**, `Throttle Forward` 0.27 -> 0.84, `Grounded Stationary Ratio` 0.198 -> 0.044, entropy 0.155 -> 0.489. Verified in isolation first -- resuming the EXTINCT p3strike policy moved steer 0.0005 -> 0.0082 with zero retraining. **But `Player/Ball Touch Ratio` FELL 0.0011 -> 0.0007** (p1air was ~0.005 at the same step count). **Root cause: chase-hit-chase is a cycle in `Phi = -dist(car, ball)`, so the shaping telescopes to ~zero over an episode and teaches nothing** -- and in the short run it penalises striking, since a hard hit sends the ball away and lowers the potential. A term that cannot be farmed around a cycle also cannot teach around one. The old term's farmability was what made it informative. `Jump When Grounded` 0.0136 = the epsilon-floor itself, so jumping now has genuinely negative advantage rather than zero support. `Action/Handbrake` ~0.5 at entropy 0.49 is an INDIFFERENT bit, not learned drift -- the reward never distinguishes it because contact is too rare. Stopped by hand at 90M |
| 2026-08-18 | p2height | +`--reward-phase aerial`: TouchHeight 15 on, Grounded 0 off; entropy 0.005; same curriculum. Resumed p1air/249425152 | Does paying for contact height, with the ground tax removed, pull the bot off the floor? | **NO, and it cannot.** `RewardShare/TouchHeight` 0.0096 and `Grounded` correctly 0, so the phase switch applied -- but `Action/Jump When Grounded Upright` stayed **0.0000** (max 0.0001 over 302 iterations) and In Air Ratio 0.0506 vs p2low 0.0457. Touch ratio 0.0132 vs 0.0128, touch height 146.7 vs 147.6 -- identical. The height term pays only for the ~54 uu of elevation the bot already reaches on ordinary ground contact; it cannot reward a jump that is never sampled. `Rating/1v1` 68.0 vs p2low 59.7 is **not** a real difference: `policy_versions` was not copied, so each probe built its own 7-version pool and the ratings are against different opponents |
| 2026-08-18 | p2entropy | entropyScale 0.002 -> 0.005 (`--entropy`), otherwise p2low | p2low left P(jump) at zero; test whether a 2.5x entropy bonus reopens exploration | **NO.** `Policy Entropy` 0.0659 vs p2low's 0.0665 -- indistinguishable. At 0.066 the policy is effectively deterministic and a 2.5x bonus does not move it; the bonus acts on the whole distribution, not on one dead action. Every behavioural metric matches p2low to 3 s.f. Flag verified plumbed (`Train.cpp:519`), so this is a real null, not a no-op |
| 2026-08-18 | p2low | +`Bounce` curriculum entry (AerialState at ball z 150-600, car 600-1600), aerial 5->10, ballContact 20->10. Resumed p1air/249425152 | The curriculum had nothing between a ground ball and a 700-1700 uu aerial -- exactly the band the bot fails. Give it practice | **Curriculum applied, effect nil.** `Scenario/Bounce/Share` 0.18, so the entry spawns as designed, and `Scenario/Bounce/EndedInGoal` is 0.95 -- the bot resolves those episodes **without ever leaving the ground**. `Action/Jump When Grounded Upright` **0.0000**, `Player/Takeoff Was Jump` 0.0000, `Leave Ground Rate` 0.0037: the 4.6% air ratio is entirely from being launched, not from jumping. **Practice cannot teach an action the policy never samples.** Prediction made before the run (p2low moves very little) confirmed |
| 2026-08-17 | p1air | **+AirRecovery 0.04** (pays rotMat.up.z while airborne) and **+Grounded 0.05** (flat, wheels-down), on top of p1advnorm | Fix the reward desert: both directional terms are gated to grounded steps, so the 92% of life spent airborne had no directional signal at all, and that is where behaviour rotted | **WORKS, and it is the first real improvement this project has had.** `In Air Ratio` **0.925 -> 0.0443** (21x), `Jump When Grounded` 0.308 -> **0.00025**. **`Episode/Late/Touch Rate` 0.00027 -> 0.00185, a 6.9x rise, and the Early/Late collapse closes from 25.3x to 1.81x** -- the bot no longer depends on the handed-to-it spawn (Early actually FELL 0.0067 -> 0.0033 while Late rose 6.9x). Overall touch ratio **0.00180**, best ever by 1.7x; Speed Towards Ball 332 -> 523; `Episode/Late/Ball Dist` 4306 -> 1990, it closes instead of drifting. **Still climbing steeply at the 40M cutoff** (late touch 0.00118 -> 0.00190 over the final 8M, no plateau). Caveats: `RewardShare/VelPlayerToBall` reads 0.482 but that is mean \|r*w\| and the term is SIGNED (`dirToBall.Dot(normVel)`), so circling generates big +/- values that cancel -- high absolute mass, ~zero net, the same trap as velBallToGoal. Real risk to watch: `Grounded` pays 0.05/step **unconditionally**, a potential do-nothing attractor (share 0.142). `AirRecovery` share is now 0.0033 -- it bootstrapped the bot out of the air and is near-inert, which is the intended shape. Two variables in one run, so attribution between the terms is untested |
| 2026-08-17 | p1age | none -- resumed p1advnorm's 40048384 with new `Episode/*` (metrics bucketed by steps since reset) and `Flip/*` instrumentation | Test three things seen watching p1advnorm in RocketSimVis: confident off the spawn then never re-orients; always diagonal flips; jump-then-delay-the-flip | **ALL THREE CONFIRMED, and the first one reframes the project.** Touch rate by episode age: **Early (0-1s) 0.00674, Mid (1-4s) 0.00097, Late (4s+) 0.00027 -- a 25x Early/Late collapse.** Approach speed 959 -> 475 -> 192. Ball distance *grows* 3031 -> 2770 -> 4557. Not a proximity artifact: Early is FARTHER from the ball than Mid (3031 vs 2770) and still touches 6.9x more. **The approach skill exists; it is spent once per spawn and never recovered.** The project's headline "touch ratio 0.001, flat forever" is a blend of a competent first second and a broken remainder. `Flip/Diagonal Share` **0.969** and `Flip/Delay Seconds` **0.580** (a reflexive double-jump is ~0.1s; RL's flip window is 1.25s) -- the stall is deliberate and it maximizes air time |
| 2026-08-17 | p1advnorm | advantage standardization ON, **from step 0** (fresh init), otherwise exactly p1-validate's config | p1probe-j applied the patch to an already-converged policy, which only sharpens what it believes. Test it properly, predicting `Action/Jump When Grounded Upright` stays near the 0.429 prior | **PREDICTION FALSIFIED, and it reframes the whole problem.** Jump rate starts at **0.428 -- exactly the uniform prior -- and climbs monotonically to 0.96**: 0.43 (2.5M) -> 0.55 -> 0.72 -> 0.88 -> 0.85 -> 0.95 -> 0.96 -> 0.91 (37.6M). The policy *learns* to jump; "frozen early preference" is dead. **But `Player/In Air Ratio` is 0.886 at jump rate 0.43 and 0.888-0.923 at jump rate 0.96, and `Leave Ground Rate` is 0.324 -> 0.332 across the same span -- both flat.** Air stint ~15 decision steps vs ground dwell ~3, so **even a uniform-random policy is airborne ~88%**. The air time is NOT caused by the flip-spam; that causality was backwards all session. Otherwise the best run this project has produced: `Rating/1v1` **+5.00 flat** at 10/20/30M vs baseline -2.54 *falling* (cross-run ratings are not directly comparable -- each run has its own version pool -- but degrading vs not degrading is), touch ratio 0.00106 vs 0.00098, Speed Towards Ball 331 vs 313. Patch reduced but did not eliminate the update decay: clip fraction 0.0168 -> 0.0029 |
| 2026-08-17 | p1pay | none -- resumed 116075520 with new `Touch/*` and `Pay/*` metrics that recompute StrongTouchReward exactly and express every term as earnings per step | Test whether StrongTouch (weight 50, scaled by impact speed) is what pays for flipping | **STRONGTOUCH FALSIFIED, and it retracts p1probe-j.** Airborne hits are not harder: `HitForce` **1385 airborne vs 1462 grounded**, per-touch `StrongValue` 0.493 vs 0.526 -- slightly *weaker* in the air, so the premise was simply wrong. The real asymmetry is frequency: `Touch/Rate` **0.00114 airborne vs 0.00039 grounded = 2.93x**. But summed per step, **grounded pays 1.7x more**: 0.0569 (VelPlayerToBall 0.0453 + StrongTouch 0.0097 + Touch 0.0019) vs airborne 0.0337. **The reward does not pay for flipping.** Which means p1probe-j's "jumping has positive advantage" was wrong: across its 99 iterations corr(Policy Entropy, jump rate) = **-0.77**, and the jump-rate rise tracks the entropy collapse (0.683->0.533), i.e. the policy concentrated on a preference it already held rather than discovering jumping was good |
| 2026-08-17 | p1probe-j | **advantage standardization** in PPOLearner.cpp (3rd local patch to external/); resumed 116075520 for 10M steps | Advantages were fed to the clipped objective unnormalized, so step size scaled with \|advantage\| -- which decays as the critic improves. Test whether that is why the policy ignores a clear value signal | **Mechanism confirmed, conclusion inverted.** Updates got much bigger: Clip Fraction 0.004 -> 0.015 (~4x), Policy Update Magnitude 0.10 -> 0.21, and entropy finally fell on its own 0.683 -> 0.533. **But the jump rate went UP, 0.731 -> 0.835**, and `Takeoff Was Jump` 0.717 -> 0.822. A stronger gradient moved the policy FURTHER toward jumping. So jumping has genuinely positive advantage under the current reward -- the policy is optimizing correctly and **the reward is what pays for flipping**. **RETRACTED by p1pay** -- see the row above. corr(entropy, jump rate) = -0.77 across this run, so the jump-rate rise is concentration on an existing preference as entropy collapsed, not evidence of positive advantage. What survives is the mechanical half: the advantage decay is real and is what "stall at 40M" was, and the patch demonstrably fixes it. Patch kept, but it was tested wrong -- resuming a policy that spent 117M steps baking in a preference only sharpens that preference. It needs a from-scratch run |
| 2026-08-17 | p1critic | none -- resumed 116075520 with new `Critic/*` metrics (direct `InferCritic` calls from the step callback) | Split "the reward favours air time" from "the gradient never reaches the jump decision" | `V(grounded)` 0.0604 vs `V(airborne)` 0.0243; `V after NoJump` 0.1735 vs `V after Jump` 0.0244. Read at the time as "the critic knows jumping is bad". **p1probe-j then showed that reading was confounded** -- the bot is grounded only 8% of the time and those states correlate with things going well, so V(ground) > V(air) is about state visitation, not about the jump action's value. Kept as instrumentation; do not read these as causal |
| 2026-08-17 | p1meas / p1meas2 | none -- resumed p1-validate's 116075520 checkpoint for ~1.1M steps with new `Action/*` instrumentation | Five probes had been aimed without ever measuring the DECISION, only the resulting state. "In Air Ratio 0.91" is equally consistent with a policy that jumps constantly and one that never jumps but keeps getting launched | **THE FLIP-SPAM IS DELIBERATE AND ABOVE CHANCE.** `Action/Jump When Grounded` **0.747** vs a uniform-prior 0.429 (18 of the 42 actions a grounded car may pick press jump) = **1.74x prior**. `Player/Takeoff Was Jump` **0.744**, so air time is self-inflicted, not spawns/bumps/ramps. `Player/Leave Ground Rate` 0.426 -> mean ground dwell **2.3 steps = 0.16 s**. Not recovery: `Grounded Upright Ratio` 0.862 and jump rate is the SAME upright (0.746) as inverted (0.698). Also `Boost When Grounded` 0.687 vs 0.357 prior = 1.92x. Confounds ruled out: obs contains `isOnGround` (DefaultObs.cpp) and the policy does condition on it (0.75 grounded vs 0.38 airborne) |
| 2026-08-17 | p1probe-i | entropyScale 0.002 -> 0.0005 | p1probe-h killed the reward-mass axis; test whether the policy is simply parked at the entropy-regularization fixed point (entropy bonus 1.4e-3 vs \|Policy Loss\| ~2e-3) | **FALSIFIED, and informative.** Entropy collapsed as intended -- 0.729 -> **0.330** over 26M, ~4.4 effective actions vs ~23, the most deterministic policy this project has made -- and behaviour did not move: In Air Ratio 0.916 (base 0.918), Phase/Recover 0.852 (base 0.862), shares unchanged. **Flip-spam is not exploration noise; a committed policy commits to flipping.** Caveat: Policy Entropy is a batch mean over 92%-airborne states, so 0.330 mostly describes the airborne policy. Reverted to 0.002 |
| 2026-08-17 | p1probe-h | velBallToGoal 0.5 -> 0 | Handoff's leading hypothesis: 44.3% of reward mass, zero-sum, policy-independent = a variance pump burying the advantage | **FALSIFIED.** Removing 44.3% of all reward mass moved `Returns STD` 21.3 -> 20.5 (**4%**) and left the advantage decay identical (-39% base vs -40% probe over 30M). Entropy fell *less* than baseline. **Reason: rewards are standardized** -- `GAE.cpp:52` divides every reward by returnStd -- so absolute mass share cannot set the noise floor, and `RewardShare` (mean \|r*w\|) is a farm detector, not a learnability measure. Made things worse too: mass reflowed to VelPlayerToBall 0.14 -> 0.40 and PickupBoost 0.13 -> 0.22, re-establishing the approach farm; Rating -9.79 at 20M vs -2.54. Reverted. **Do not probe reward weights again without a new mechanism** |
| 2026-08-17 | p1-validate | winning config (entropy 0.002, LR 3e-4, epochs 2, rebalanced rewards) | Confirm the fixed config keeps improving past 30M | **IT DOES NOT.** Stopped by hand at 117M/150M once the answer was unambiguous. Learns until ~40M then stalls completely: touch ratio 0.00089 -> 0.00104 by 50M then dead flat for 65M steps; In Air Ratio *rises* 0.913 -> 0.925; entropy bottoms at 0.650 (36M) and drifts back **up** to 0.700; KL decays 0.00125 -> 0.0005; every RewardShare static to 3 significant figures across 115M steps. **Rating/1v1 falls monotonically: -2.54 (10M) -> -5.00 (50M) -> -7.50 (80M+)** -- reward flat while rating drops is the farming signature from docs/metrics.md. `GAE/Avg Advantage` decays 0.15 -> 0.088 against a `Returns STD` of 21.4. PPO health is no longer the binding constraint; the reward function is. See `.claude/handoffs/2026-08-17-p1-reward-diagnosis.md` |
| 2026-08-17 | p1probe-g | entropyScale 0.002 -> 0.018 (full upstream defaults) | Is upstream's entropy scale affordable now that LR/epochs are fixed? | NO: entropy pinned again at 0.783->0.785, KL -> 0, StrongTouch share flat. 0.018 is still too high for this action space. **Keep 0.002** |
| 2026-08-17 | p1probe-f | policyLR/criticLR 1.5e-4 -> 3e-4, epochs 1 -> 2 (upstream defaults) | E showed the pin was breakable; give the optimizer enough updates to exploit it | **BREAKTHROUGH.** Entropy 0.772->0.654 (first run where it really falls), KL 0.0055 (10x prior runs), Policy Update Magnitude 3x, touch ratio 0.0008->0.0011, StrongTouch share 0.050->0.078. First run where behavior moves. **This is the chosen config** |
| 2026-08-17 | p1probe-e | entropyScale 0.018 -> 0.002 | Test the hypothesis that entropy loss is drowning the policy gradient | Pin partially breaks: entropy 0.773->0.721 (vs frozen 0.79). Confirms hypothesis; LR/epochs still limiting |
| 2026-08-17 | p1probe-d | entropyScale 0.035 -> 0.018 (upstream default) | "Policy Relative Entropy Loss" hit -22 (entropy term 22x policy gradient) -- suspect entropy is drowning learning | Barely moved: entropy 0.774->0.789, KL still ~0. Necessary but nowhere near sufficient |
| 2026-08-17 | p1probe-c | VelBallToGoal 2 -> 0.5, 100M budget | Longer run to see if 30M was simply too short | Killed at ~42M: every metric flat as a board (In Air 0.896->0.898, touch 0.0007->0.0008). Not a horizon problem -> **stopped tuning rewards and looked at PPO health metrics, which is what found the real bug** |
| 2026-08-17 | p1probe-b | + GroundedReward gate on VelPlayerToBall and FaceBall | Flip-spam farms velocity shaping; pay approach money only on wheels | Gate works: VP2B share 0.39->0.06. But VelBallToGoal absorbed it (share 0.67 -- mostly passive ball motion = zero-sum noise). Behavior still frozen at 30M (In Air 0.90, touch 0.0007) -> 30M too short to discriminate; next probe longer + VB2G cut |
| 2026-08-17 | p1probe-a | Rewards: VP2B 3->0.5, +Touch 5 (new term), VB2G 4->2, Goal 30->100, FaceBall 0.1->0.05 | Baseline paid 67% to VP2B, ~5% to outcomes -- rebalance toward outcomes | Shares moved as computed (VP2B 0.39, Goal 0.11, outcomes ~15%) but behavior unchanged at 30M: touch 0.0008, In Air 0.90 |
| 2026-08-17 | deploy-probe (as reward baseline) | old weights: VP2B 3, ST 50, VB2G 4, Goal 30, PB 5, FB 0.1 | First RewardShare telemetry | FARMING: VP2B share 0.67 flat, outcomes ~5%, In Air 0.89, touch 0.001 flat, Phase/Recover 0.84 -- flip-spam toward ball, never learns to drive |
| 2026-08-17 | (live match) | deploy-probe @30M, deterministic, CPU | First-ever live RLBot v5 match; parity check (plan Task 13) | Deployment VERIFIED: model loaded, pads mapped with no warnings, no console errors. Bot flip-spams and lands upside down in-game -- matches training telemetry exactly (In Air Ratio ~0.90 all run, touch ratio ~0.001), so behavior is faithful, just undertrained + a farming-shaped policy. Nexto probe: v4 bridge moot -- NectoFamily v5 port (libs/opponents/NectoFamily) runs natively, 2/2 agents spawned. Gotchas: RLBotServer cannot auto-launch RL under GE-Proton ("Could not find Proton installation") -- launch RL manually with `%command% -rlbot RLBot_ControllerURL=127.0.0.1:23233 RLBot_PacketSendRate=120 -nomovie`; HIVE_MODEL must be an absolute path (run.sh re-checks it from another cwd) |
| 2026-08-17 | deploy-probe | defaults, 128 games, 30M steps | Throwaway checkpoint for the first live RLBot match (plan Task 13) | Trained ~6 min; `verify` PASSED (4/4); live match not yet run |
| 2026-08-17 | throughput-* | 1v1 obs (89 wide), tsPerItr 100k; games swept 64-320 | Re-measure steps/sec at 1v1 width (Phase 0) | steps/sec: 64g=71.4k, 128g=80.6k, 192g=79.4k, 256g=78.1k, 320g=74.6k -- 128 games wins; old ~120k/s figure was multi-size (up to 6 cars/arena, more player-steps per sim step) |

## Corrections

**p1air `grounded` weight is not a single value.** It was changed by hand during
the run, 0.05 -> 0.04 -> 0.02, and the row above records only the 0.05 it
started with. p1air is therefore a valid baseline for *behaviour* (it produced
the viable ground bot at 245M) but **not** a reward-weight reference point:
nothing in it can be attributed to a specific value of `grounded`, and no later
run can be compared against it on the ground-bias axis. The committed config
carries the final 0.02.

Consequence for the p2 probes: they are attributed against **each other**, not
against p1air. p2low is the baseline for the sequence.

## The p2 sequence in one line

`Action/Jump When Grounded Upright` in p1air: **0.4904 at step 0 -> below 0.01 by
~17.5M -> 0.0000 from ~30M onward**, and flat zero for the remaining 215M steps.
In Air Ratio fell 0.90 -> 0.06 over the same window. Touch ratio then climbed
0.0009 -> 0.0131 across 245M steps *entirely without jumping*.

The ground bias did not discourage jumping, it **extinguished** it, inside the
first 20M steps. Everything after that -- p1air's remaining 225M steps and all
three p2 probes -- is a policy optimizing over an action space with the jump
actions effectively deleted. No reward weight and no curriculum entry can
recover that, because PPO's gradient is proportional to the probability of the
action taken, and that probability is zero.

**The window is the first ~15M steps.** Any future run has to keep the jump
action alive through it.

## Standing lesson from p4pbrs

Potential-based shaping is only useful if the potential is on a quantity the
agent's own success *improves*. `Phi = -dist(car, ball)` fails that test: the
bot's best action (striking the ball) moves the ball away and lowers the
potential, and chase-hit-chase closes the cycle so the whole term sums to zero.

The no-farm guarantee and the teaching signal are the same property viewed from
opposite sides. Removing farmability without replacing the signal leaves only
the outcome terms, and at a touch ratio of ~0.001 those are far too sparse to
bootstrap anything.

The epsilon-floor is orthogonal to all of this and should be kept regardless of
which reward direction wins: it is the only thing that has ever reversed an
extinction, and it is verified to do so on an already-dead policy.

## Standing lesson from p10touch: a split metric hides the other half

`Action/Jump When Grounded Upright` was quoted as "jumping is extinct" across
p8ref, p9rel and p10touch. It only ever measured jumps from a **wheels-down**
car. Its sibling bucket was named "Inverted", read as upside-down recovery, and
never cited -- and it was carrying all the wall jumping, at 15x the eps-floor
and rising.

The null discipline from p7approach says compute the chance value. This adds
the other half: **when a metric is split by a condition, quote every branch and
publish the branch's own denominator**, or the split silently becomes a filter.
`Player/Grounded Tilted Ratio` now ships next to the jump rates for that reason.

## PRE-REGISTERED: p19pool (seeded from `main-p18entropy/3075056128`, 2026-08-23)

**One concept -- "the opponent spans a real skill range" -- three coupled
changes, declared as such.** Reward stack, obs, spawns, mask, LR, gamma,
entropy and `tsPerItr` all frozen at p18 values.

1. **The version pool is curated, not rolling.** It held 32 versions at 5M
   spacing: a 160M window which the ladder measures as spanning ~10 Elo, i.e.
   near-clones. It now holds **5: p12goal, p15manual, p16, p17 and p18, spanning
   478 Elo.** `Learner.cpp:536` draws the opponent **uniformly from the pool per
   iteration**, so composition is the lever -- injecting 4 ancestors among 32
   clones would have picked them 2.2% of iterations.
2. `trainAgainstOldChance` **0.2 -> 0.5**. 80% of training was against a current
   copy of the same network, where a k=1 term has zero expected advantage.
3. `tsPerVersion` **5M -> 200M**, so a 400M run adds ~2 versions instead of 80
   and the curated pool is not re-diluted mid-run.

**Why now.** The 483M p18entropy run showed `Average Step Reward` flat to +0.7%,
so PPO has converged on its objective, and 57.6% of reward mass
(`Goal` + `ShotOnTarget` + `Save` + 0.8x`TouchGoalAccel`) has zero expected
population advantage against a copy of itself. This is the cheapest available
test of whether that symmetry is what the plateau is made of.

**Bonus: `Rating/1v1` becomes anchored for the first time.** Pool ratings are
seeded from the 2026-08-23 ladder's measured Elo gaps below p18's 394
(p12goal -84, p15manual +90, p16 +363, p17 +376, p18 +394), and with
`tsPerVersion` at 200M the anchors barely move during the run.

**Predictions, written before the run.**

1. **`Rating/1v1` rises above 420 by +200M.** It has oscillated about 394-424
   for 483M steps against a pool it could not exploit.
2. **The zero-sum-term behaviour metrics move for the first time since p16.**
   `Shot/Saved Share` and `Save/Converted` moved 0.6% and 1.5% across the whole
   483M; anything beyond their oscillation bands (4.9% and 2.0%) is signal.
3. **`SB3 Clip Fraction` rises above 0.040** (0.0342-0.0375 for all of p18) --
   a genuinely different opponent should produce advantages worth a larger step.
4. **The ladder gains more than +42 Elo over p16 in 400M steps**, against p17
   and p18 together managing exactly that in ~800M.

**Kill criteria.** Stop at 100M if `Policy Entropy` has not fallen or
`Rating/1v1` is below 380; stop at 200M if no behaviour metric has left its p18
oscillation band. **The step budget is 3,475,000,000 cumulative (+400M).**

**What a null result means, and it is not nothing.** If a pool spanning 478 Elo
at a 50% encounter rate moves nothing, then opponent symmetry is NOT what the
plateau is made of, and item 11 -- the leading hypothesis since p17 -- is
falsified rather than deferred. That would point the restart at the reward
stack and the state distribution instead.

## PRE-REGISTERED: p18entropy (seeded from `main-p17/2591724288`, 2026-08-21)

**A discriminator, not a reward run. One variable.** The reward stack is
byte-identical to p17 -- save term included, budgets untouched -- and the only
change is `--entropy-target 0 --entropy 0.002`, which disables the controller
and pins `entropyScale` to a fixed value ~7x below the 0.0144 it had drifted to.
0.002 is not arbitrary: it is the value CLAUDE.md records as having produced
"the only breakthrough" this project has had.

**The question it answers.** p17 proved a converged policy stays converged. It
did not say WHY. Either (a) `entropyTarget = 0.40` is pinning the policy at
5.98 effective actions of 90 and forbidding the sharpening that better shot
selection requires, in which case every reward verdict since the controller was
introduced was drawn against a policy that could not act on it -- or (b) this is
a genuine local optimum and the lever is the observation, the network or
opponent diversity, and no reward work should be done until that is fixed.

Deliberately keeping the save term in. Removing it at the same time would
confound the one thing being measured.

**Baselines, p17 final 20% (from wandb `s2fdddcr`):** `Policy Entropy` 0.3974,
`Entropy Scale` 0.0144, `SB3 Clip Fraction` 0.0354, `Mean KL` 0.0038,
`Shot/Distance` 4009.2, `Shot/Saved Share` 0.289, `Save/Converted` 0.793,
`Rating/1v1` 425.8, `Action/Steer Nonzero` to be read at start.

**Predictions:**

1. **`Policy Entropy` falls below 0.30 by 25M** (from 0.397). PRIMARY, and it
   is a mechanism check rather than a result: if entropy does not move once the
   controller is off, the controller was never what pinned the policy and
   hypothesis (a) is dead on the spot.
2. **`SB3 Clip Fraction` rises above 0.045** (from 0.0354, the value p16 ran out
   of road at and p17 never left).
3. **At least one behavioural metric moves more than 5% over 50M**, against
   p17's largest move of ~1% over 300M. Nominated in advance so this cannot be
   fitted after the fact: `Shot/Distance` and `Shot/Saved Share`.

**Kill criteria:**

- **AT ANY TIME:** `Obs/Non-Finite Rate` > 0, or any NaN in the PPO metrics.
- **AT ANY TIME:** `Policy Entropy` < 0.05. That is near-determinism and the
  extinction regime this project has lost three control dimensions to.
- **AT ANY TIME:** `Action/Steer Nonzero` at the 0.02 eps-floor. The fourth
  external patch protects against extinction but does not prevent it being
  approached, and a falling entropy run is exactly when to watch for it.

**Not a rating test.** 50M is far too short for `Rating/1v1` to mean anything,
and the pool is inherited from p17. Rating is recorded, not predicted.

## p18entropy OUTCOME: the entropy fix was real, and it was not the answer

**Adopt the entropy change; abandon the hypothesis it was testing.**

Two things are true at once and they must not be collapsed.

**1. The pathology was real.** `Policy Relative Entropy Loss` in p17 swung
between +940 and -42 against a documented healthy ~0.1 -- the entropy bonus
drowning the policy gradient. With the controller off it reads **0.37 to 0.92**.
Entropy fell 0.389 -> 0.285, i.e. **5.76 -> 3.61 effective actions of 90**, and
`SB3 Clip Fraction` rose 0.0361 -> 0.0438 (+21.2%), the biggest effective steps
since p16's mid-life. `entropyTarget = 0.40` was set from a single external data
point a peer called "a bit low", and it was costing real gradient.

**2. Fixing it moved no behaviour.** `Shot/Distance` +0.2% over 50M.
`Save/Converted` 0.797 -> 0.796. `Shot/Saved Share` -4.1%, which merely
continues a drift already visible across p17's 300M and is not a response.

**And entropy is asymptoting, so "it needs longer" is weak.** Per-decile drops
run -0.031, -0.022, -0.019, -0.006, -0.006, -0.005, -0.003, -0.007, -0.004: a
7x deceleration settling near 0.27-0.28. The policy found a new, lower
equilibrium and stopped. The transient is mostly spent, not mostly ahead.

### The diagnosis this points to, and it is not the reward function

The plateau is most likely a **self-play fixed point**, not a policy limit:

- `trainAgainstOldChance` is 0.2, so **80% of training is against a current
  copy of the same network**, and the other 20% is against 32 snapshots spanning
  only the last ~160M steps -- policies nearly identical to the current one.
- **~46% of reward mass sits in zero-sum terms**: `TouchGoalAccel` 0.31 at
  k=0.8, `ShotOnTarget` 0.11 at k=1.0, `Save` 0.044 at k=1.0. Against a copy of
  yourself a purely differential term has **zero expected advantage** by
  construction, and there is no asymmetry to exploit.
- `Rating/1v1` is measured against that same pool, so the entire apparatus is
  self-referential. A policy saturated against copies of itself looks exactly
  like this: every metric flat, rating oscillating about a fixed point.

This is the p4pbrs lesson at the level of the whole stack rather than one term:
the no-farm guarantee and the teaching signal are the same property. See
`.scratch/run-backlog/issues/08-zerosum-k1-teaching-signal.md` and the new
`11-opponent-diversity.md`.

**Carry forward:** `entropyTarget 0` with `entropyScale 0.002` is now the
default worth keeping -- it is strictly better on every PPO health metric and
costs nothing.

**That carry-forward was never written into `Config.h`, and 2026-08-23 is what
it cost.** The flags were CLI-only, the resume omitted them, and the controller
came back on with nobody watching. `CONFIG_HISTORY.json` recorded it correctly
and no one read the file. **Command-line overrides do not survive a resume**;
anything meant to persist belongs in `Config.h`.

## p18entropy EXTENDED (483M): the objective is maximised, and 58% of it is unmeasurable

The accidental 433M extension answers a question the 50M probe could not, and
it inverts the standing framing of the plateau.

**What was thought:** the reward is being optimised, but it buys the wrong
behaviour -- a misspecification problem, fixable with better terms.

**What 483M shows:** `Average Step Reward` moved **+0.7% net against a 2.6%
oscillation**. The reward is not being optimised any more either. PPO has
converged on its objective, at two different entropy levels, with every health
metric constant to three significant figures. There is no residual gradient to
harvest, so **no reward-term experiment run against this configuration can
produce a readable result** -- which is what p17 and the 50M p18entropy probe
each discovered separately and neither could prove.

### The mechanism, from the k-arithmetic rather than from intuition

A term wrapped at opponent scale k pays the actor `+S` and the opponent `-kS`.
Self-play at 1v1 samples both roles from one network, so the population nets
`S(1-k)` per event. At k=1 the expected advantage is exactly zero. Goals are
inherently k=1 in 1v1 -- a goal for is a goal against.

| Term | Share of reward mass | k | Population-expected |
|---|---|---|---|
| `TouchGoalAccel` | 30.7% | 0.8 | 6.1% |
| `Goal` | 17.4% | 1.0 (structural) | 0 |
| `ShotOnTarget` | 11.2% | 1.0 | 0 |
| `Save` | 4.4% | 1.0 | 0 |
| **subtotal** | **63.7%** | | **6.1%** |

**57.6% of the budget has zero expected advantage against a copy of itself**,
and `trainAgainstOldChance = 0.2` means 80% of training is exactly that. The
remaining ~42% -- `AirTouch` 13.3%, `SpeedToBall` 10.3%, `FlipSpeed` 6.6%, and
the small dense terms -- is a **solo-play objective**: it measures how the car
moves, not whether it wins.

That predicts precisely what the 483M shows. The only metrics with residual
movement are `RewardMass/AirTouch` (+5.9%, 5.4 sigma), `Touch/Above 450`
(+10.6%), `Player/Touch Height` (+2.6%) and `Phase/AirDribble` (+4.6%). **The
bot is still improving on the largest term that is not cancelled by symmetry,
and it is improving at style rather than at winning.**

`AirTouch` being that term is not comfortable: issue 05 has a red test saying it
already pays **1.33x the entire finishing block** per episode.

### Why no number here can say whether that is good or bad

`Rating/1v1` is an Elo against a rolling 32-version, ~160M-step pool, and each
new version inherits the current rating on entry. It is a random walk with no
anchor. Measured on this run: 41 samples, per-sample sigma 6.77, so the
random-walk sigma over 40 updates is **42.8** -- the observed 424 -> 394 drift
is inside chance and means nothing in either direction.

**This is now the binding constraint on the whole project.** Every open reward
ticket asks "did this make the bot better", and there is no instrument in the
repo that can answer it. See `.scratch/run-backlog/PLAN-post-p18.md`.

### Process repairs made 2026-08-23

- **wandb fork.** The resume started a second run (`ciygwkg4`) alongside the
  original (`hp16vami`) instead of continuing it, and the label's CSV lost its
  first 496 iterations. Both repaired: `scripts/merge_wandb_runs.py` appends a
  fork's history onto the run it should have continued, refuses unless the two
  are contiguous in `Total Timesteps`, restores missing CSV rows and re-points
  the ownership sidecar and every checkpoint `run_id`. `hp16vami` now holds all
  4799 iterations, 2592M -> 3075M. The fork is renamed, not deleted.
- **`metric_receiver.py` is no longer silent on the happy path.** It prints
  which wandb run it resumed, and warns explicitly when a label with existing
  history is handed no run id -- the exact fork signature, previously silent.

## p17 OUTCOME: the policy was already finished before the run started

**Read this before designing another reward term.** p17 is the cleanest
evidence this project has that the current bottleneck is not the reward
function.

**Nothing moved.** Not "the new term underperformed" -- across 300M steps the
biggest behavioural change in the entire metric set was `Flip/Neutral Share`
0.0093 -> 0.0196, both negligible. `Shot/Distance` 4007.4 -> 4009.2 (**0.0%**),
`Player/Velocity Alignment` 0.1%, `Shot/Toward Net Rate` 0.2%, `Touch/Above 200`
0.0%.

**And PPO was healthy the whole time.** `SB3 Clip Fraction` 0.0354, `Mean KL`
0.0038, `Policy Update Magnitude` 0.0696, `GAE/Avg Advantage` 0.153 -- constant
to three significant figures for 300M steps. This is not a stalled learner. It
is a converged one, updating forever and going nowhere.

**The entropy controller was engaged 100% of the run**, holding `Policy
Entropy` at 0.397 = **5.98 effective actions of 90**, with `Policy Relative
Entropy Loss` swinging between +940 and -42 against a documented healthy range
of ~0.1 (the entropy bonus drowning the policy gradient).

**But that is NOT sufficient on its own, and the comparison matters.** p16 ran
with the same drowning (`Policy Relative Entropy Loss` -184, +29, +38, -11) and
still gained 104 rating points. What separates them is the clip fraction:

| | clip fraction | `Rating/1v1` |
|---|---|---|
| p16 early (1104-1461M) | 0.050 | 321 -> 345 |
| p16 mid (1580-1927M) | 0.051 -> 0.035 | 369 -> 409 |
| p16 late (2163M) | 0.036 | 425 |
| **p17, all 300M** | **0.035, flat** | **431 -> 426** |

p17 spent its whole life at the value p16 ran out of road at. **p16 had already
finished learning before p17 was seeded**, and p17 confirmed it for 85 minutes.

**Consequences for how the last several runs are read.** Any reward conclusion
drawn from a policy pinned at a fixed entropy is suspect, because improving
shot selection REQUIRES becoming more selective, i.e. lower entropy in those
states, and the controller forbids that globally. `entropyTarget = 0.40` was
set from a single external data point a peer called "a bit low"; it is now the
prime suspect and has never been tested against.

**Operator observation, recorded because a metric set is not a bot.** Watching
the run, the save behaviour was unchanged -- which matches the telemetry
exactly -- but a few small stylistic improvements were visible that had not
been there before. The instruments cover shots, saves, touches, speeds and
action rates; they cover positioning, rotation and decision quality **not at
all**, so "nothing moved" is a statement about what is measured. Attribution is
also ambiguous: 300M extra steps is 300M extra steps whatever the reward says.

**What survived and is worth keeping:**

- The instruments. The original complaint is now a number: the average
  on-target shot is taken from **4,009 uu** out with 1.59 s of flight, and it
  is flat to 0.0%. `Shot/Time` was computed by `ProjectShot` all along and no
  reward or metric had ever read it.
- `Save/Converted`'s measured null of **0.789**, which is not 0.5 and is not
  analytic. See `docs/metrics.md`.
- `CONFIG.json` / `CONFIG_HISTORY.json`, which exist because p16's live budgets
  had to be back-solved from telemetry.
- The wandb guard: p16's `o73axl44` is intact and p17 opened cleanly as
  `s2fdddcr`.

**Still open, deliberately not fixed in p17:** `airTouch` at 55 pays 64.5 per
episode against the finishing block's 48.5, the one failing test in the suite.

## PRE-REGISTERED: p17 (seeded from `main-p16/2281639168`, 2026-08-21)

**One concept: stop paying for the attempt, pay for the outcome.** Three
coupled changes, declared as not-one-variable, with separable instruments.

The complaint that started it, from watching the bot rather than the metrics:
it throws possessions away. It fires low-percentage ground shots at the net
from distances where a save is near-certain, and it clears in ways that hand
the ball straight back. Neither failure is a "who is closer to the ball"
failure -- in both the bot HAS possession and IS closer -- so the reward has to
be keyed on the consequence of the touch, not the position before it.

**1. New `SaveReward`, signed, zero-sum at k = 1.0.** On a touch rising edge,
the change in threat at the player's OWN net, through the same `ProjectShot`
and the same `MISS_SCALE` (892.755) `ShotOnTarget` uses:
`exp(-miss_before/S) - exp(-miss_after/S)`. Removing a threat is a save;
creating one is a botched clearance; one expression prices both. Paired with
`ShotOnTarget` this gives the semantics the run is named for: **an on-target
shot is paid if and only if it is not saved.**

Deliberately NOT graded by how imminent the incoming shot was. That pays more
for the harder save and therefore penalises the shooter MORE for taking the
better shot. The expected value already carries the grading, since P(saved) is
high for a hopeless shot and low for a good one.

**2. `shotOnTargetOpponentScale` 0.8 -> 1.0.** At 1v1 a term wrapped at
opponentScale k pays the shooter `+S` and the defender `-kS`, so the
POPULATION nets `S(1-k)` per event -- and self-play samples both roles from one
policy. At 0.8 the stack was paying `+0.2S` for an on-target shot **existing**,
whoever took it and whatever became of it: a direct subsidy for exactly the
shot-spam being removed. `saveOpponentScale` is 1.0 for the same reason, and
at any k < 1 the save term funds a "let them shoot so I can save it" farm.

`touchGoalAccelOpponentScale` stays at 0.8. Same leak, but it prices ball
movement rather than shot selection and has ~1B steps behind it. **Filed as a
p18 candidate, not fixed here.**

**3. `shotOnTarget` 22 -> 32.** Forced, not chosen. The save budget is capped
at `shotOnTarget * E[Shot/Strength]` so that a saved on-target shot is never
net-negative for the shooter -- without that cap the bot learns that a savable
shot is not worth taking, which prices out rebound pressure, where consecutive
saves progressively pull a defender out of position. At the live value of 22
the cap is 11.5 and the save term realizes 0.043, i.e. inert. 32 is the
ceiling: at 35 a shot on target out-pays a strong touch and the ordering guard
in `test_rewards.cpp` fires.

### The live p16 config was not what anyone thought, and had to be back-solved

**This is the most important thing in this entry.** p16 ran with at least four
budgets that appear in neither HEAD nor the working tree, and nobody knew.
`RewardShare` is `m_i w_i` normalized, so with the policy frozen the ratio of
probe share to p16's final share is proportional to `w_probe / w_live`. Eight
of twelve terms reproduce within 2%, which is what validates the other four:

| term | assumed | **actually live in p16** |
|---|---|---|
| `goal` | 25 | **34** |
| `shotOnTarget` | 35 | **22** |
| `airTouch` | 35 | **55** |
| `air` | 2.88 | **1.9** |

Building p17 on the assumed values would have made it a **five**-variable run.
The baseline is set from the measurement instead. `CONFIG.json` and
`CONFIG_HISTORY.json` now ship (full config per run folder, plus an
append-only delta log keyed to the step count each change began at, accurate
to within `tsPerSave` = 1M), so this can never be necessary again.

### Calibration probe

`main-p17cal`, resumed from `main-p16/2281639168` with `--lr 0
--entropy-target 0`, 5.0M steps. `Policy Update Magnitude` and
`Mean KL Divergence` both **exactly 0**, so the policy could not move and the
measured per-term means are honest. Budgets solved by
`scripts/solve_budgets.py --targets p17save` in PARTIAL mode: one free term,
twelve held. A full re-solve would claw back `TouchGoalAccel` and `Goal`, the
two terms p16 got right.

Solver returned **28.05** for a 0.10 target share. The cap binds at
`32 * 0.5247 = 16.79`, so `save = 16.5` and the **realized share is ~0.061,
not the 0.10 target**. Recorded rather than papered over: it is above the
~0.05 floor below which p5goalpot's potential was inert, and the cap wins
because the alternative teaches the bot not to shoot.

**Baselines, p17cal probe (tail 10), all under the frozen p16 policy:**
`Shot/Saved Share` 0.3131, `Shot/Time` 1.573 s, `Shot/Distance` 3932 uu,
`Save/Converted` 0.7888, `Save/Threat Faced Rate` 0.1582,
`OwnHalf/Touch Rate` 0.0166, `Player/Ball Touch Ratio` 0.0149,
`Shot/Strength` 0.5247, `Rating/1v1` 432.2, `Episode/Mean Steps` ~375.

**`Save/Converted`'s null is 0.7888 and is NOT analytic.** A ball aimed at a
1786-uu mouth is deflected off target by almost any contact, so chance is high,
not 0.5. A run reading 0.75 against an assumed 0.5 would have been scored a
triumph while nothing had been learned. See `docs/metrics.md`.

**Predictions:**

1. **`Shot/Saved Share` falls below 0.25** (from 0.3131). PRIMARY. This is the
   quantity the whole run is aimed at: the shooter now pays for a shot that
   gets saved, so shot selection must improve or the term did nothing.
2. **`Shot/Distance` below 3400 uu and `Shot/Time` below 1.35 s** (from 3932 /
   1.573). This is the operator's actual complaint, instrumented for the first
   time -- `ProjectShot` had computed `time` all along and nothing read it.
3. **`Save/Converted` rises above 0.85** (from 0.7888). The bot clears
   decisively instead of deflecting, which is the positive side of the term.
4. **`Rating/1v1` > 460 at 300M** (from 432.2), against the pool inherited from
   p16 so the comparison is valid for roughly the first 160M steps.
5. **`OwnHalf/Touch Rate` holds above 0.014** (from 0.0166). Not a success
   metric -- the GUARD on the term's negative side.

**Kill criteria:**

- **10M:** `Obs/Non-Finite Rate` = 0; `SB3 Clip Fraction` in [0.02, 0.25];
  `Mean KL Divergence` < 0.03; and `RewardShare/Save` within 3x of its solved
  0.061, i.e. in [0.020, 0.183].
- **25M:** `Shot/Saved Share` falling rather than rising. Direction, not level.
- **AT ANY TIME:** `RewardShare/Save` > 0.25 -- the save farm. k = 1.0 should
  make it unreachable, which is exactly why breaching it would mean the
  zero-sum arithmetic is wrong rather than the budget.
- **AT ANY TIME:** `OwnHalf/Touch Rate` < 0.010 against `Player/Ball Touch
  Ratio`. The signed term's failure mode is a bot that refuses to touch the
  ball near its own net rather than risk creating a threat.
- **AT ANY TIME:** `Episode/Mean Steps` > 500 **while `RewardMass/Goal`
  falls.** Carried from p16, but note that better defending on BOTH sides of
  self-play legitimately lengthens episodes, so this only fires if the goal
  mass moves with it.

**Known violation, inherited and deliberately not fixed.** At the live
`airTouch` of 55 the air block pays **64.5 per episode against the finishing
block's 48.5**, breaking a guard written specifically so "get it high" would
never be the stack's loudest opinion again (`test_rewards.cpp:873`, the one
failing test). Cutting `AirTouch` is a retarget and would confound p17, so it
is carried. It belongs with the p18 air question.

**Also corrected:** `ASSUMED_SHOT_STRENGTH` in the ledger tests, 0.25 -> the
measured 0.5247. p16 published `Shot/Strength` for exactly this purpose. It
more than doubles the finishing block's shot contribution and makes every
"stay below finishing" guard strictly harder to pass.

**Frozen:** gamma 0.99, policyLR/criticLR 2e-4, `tsPerItr` 100k, obs Relative,
spawn Random, `maskActions` false, the action parser, `entropyTarget` 0.40,
`infiniteBoostChance` 0.1, `touchGoalAccel` 45 at exponent 2 and k = 0.8,
`goal` 34, `airTouch` 55, `air` 1.9, and all six external patches.

**Seeding.** `main-p16/2281639168` copied with `RUNNING_STATS.json`'s `run_id`
DELETED. Verified: policy weights hash-identical, p16's own `run_id`
(`o73axl44`) intact, no `metrics/main-p17.wandb-id` so the receiver opens a
fresh run. `policy_versions` copied (32 versions) so `Rating/1v1` carries over.
`return_stat` deliberately NOT reset, for the reason p16 documented.

**Stop at 300M**, primary read at 100M, and do not interrupt at 100M.

## PRE-REGISTERED: p16 (seeded from `main-p15manual/1044744064`, 2026-08-20)

**Two changes, one concept: stop paying for possession, start paying for the
shot.** Declared as not-one-variable, with separable instruments.

p15 ran 1044.7M steps under a config that changed at least fifteen times and
was never logged. p16 draws a line under that: the stack is frozen from here
and every subsequent change gets its own row.

**Seeding.** `checkpoints/main-p15manual/1044744064` copied to
`checkpoints/main-p16/1044744064` with `RUNNING_STATS.json`'s `run_id`
DELETED, so wandb starts a new run instead of renaming p15's (`8l6ewpl5`).
Verified: policy weights hash-identical to p15, p15's own `run_id` intact, and
a dry run opened a fresh id rather than adopting p15's. `policy_versions` was
copied too, so the 32-version opponent pool and `Rating/1v1` 316.5 carry over
-- which makes the rating comparable across the p15/p16 boundary for the first
~160M steps, until the pool rotates out.

**`return_stat` was deliberately NOT reset** (count 2.87M, std 50.9). It is a
whole-run cumulative Welford and it describes a reward stack that no longer
exists, so it is stale by construction. Resetting it would hand the critic a
50x target rescale for no benefit, because the advantage-standardization patch
now makes the policy step independent of return scale. It is a known
limitation, not an oversight: only a fresh run gets an honest one.

**1. `ShotOnTarget` gains a strength factor.** `budget x
clamp(dv_goalward/3611, 0, 1) x exp(-miss/892.755)`, budget 12 -> 35. The term
shipped as a plateau in BOTH placement and force, so a ball rolling goalward on
the car's hood paid exactly what a strike paid, on every contact rising edge --
measured 4.16 touch-units per contact sequence, one sequence every ~50 steps,
against a goal worth 25 once. Walking the ball to the line was strictly optimal
and slower was strictly better, and the bot learned precisely that. Keyed on
delta-v rather than ball speed because a carried ball travels at the CAR's
speed (1309 uu/s), so a speed factor would still have paid a dribble ~40%.
Linear, not convex: TouchGoalAccel already prices power convexly and doubling
up would re-create the blast-it failure. Placement stays a plateau, so corner
shots are untouched.

**2. `airTouchHeightExponent` 2 -> 1, `airTouch` 20 -> 12.** At the measured
touch height of 191 the term paid 0.175 against a takeoff cost of 0.68 --
**3.9x BELOW break-even** -- and the convexity made the collapse
self-reinforcing, because a falling touch height cut the payment
quadratically. At exponent 1 it pays 1.12, or 1.65x break-even. This is the
third time this project has found and lost the air game at a margin near 1.0
(p12, p14, p15). The guard test that should have caught it asserted break-even
at a ball height of 800 -- a height the bot reached when the test was written
and does not reach now -- so it passed while the term was underwater in play.
It now asserts at the MEASURED height.

**Also carried, not variables:** `REFERENCE_EPISODE_SECONDS` 41.5 -> 26.1 with
the four rate budgets rescaled to hold their per-step weights to within 0.7%
(episodes shortened to 392 steps under gamma 0.99, so the old constant was
silently delivering every rate budget at 1.59x its declared integral); and
`Shot/Strength` / `Shot/Ball Speed` published so the one guessed constant in
the budget arithmetic (`ASSUMED_SHOT_STRENGTH = 0.25`) can be replaced by a
measurement.

**Baseline, p15manual over its last 10M steps:** `Rating/1v1` 317.4,
`Episode/Mean Steps` 404.6, `Touch/Hit Force` 708, `Touch/Goal Accel Raw`
0.1064, `Player/Touch Height` 190.4, `Touch/Above 450` 0.0459,
`Phase/AirDribble` 0.0107, `Phase/GroundDribble` 0.0564, `Phase/Aerial` 0.0354,
`Shot/On Target Share` 0.569, `Shot/Miss Distance` 486,
`AirTouch/Direction Factor` 0.781, `AirTouch/Backward Share` 0.171,
`Player/Ball Touch Ratio` 0.0306, `Touch/Edge Rate` 0.0215, ledger 110.7
touch-units/episode, goals/episode 1.038, V 14.2, continuation ceiling 17.1,
`goal/ceiling` 1.46x. Realized shares: ShotOnTarget 32.8%, TouchGoalAccel
26.6%, Goal 23.4%, AirTouch 1.9%.

**Predictions:**

1. **`Player/Touch Height` > 260 at 100M** (from 190.4), and
   `Touch/Above 450` > 0.09 (from 0.0459). PRIMARY for change 2. If the air
   game does not come back at 1.65x break-even, the problem is not pricing and
   no further air budget will fix it.
2. **`RewardMass/ShotOnTarget` share falls to 0.05-0.15** (from 0.328).
   Outside that band the budget of 35 was mis-set, in a direction
   `Shot/Strength` will name.
3. **`Phase/GroundDribble` falls below 0.040** (from 0.0564) while
   `Touch/Hit Force` rises above 850 (from 708). The dribble stops being an
   end in itself and becomes a setup.
4. **`Shot/On Target Share` holds above 0.50** (from 0.569). The strength
   factor must not cost accuracy -- it only removes payment for touches that
   were never shots.
5. **`Rating/1v1` > 340 at 100M** (from 317.4), against a pool inherited from
   p15 so the comparison is valid for that span.

**Kill criteria:**

- **10M:** `Obs/Non-Finite Rate` = 0; `SB3 Clip Fraction` in [0.02, 0.25];
  `Mean KL Divergence` < 0.03; and `RewardMass/ShotOnTarget` share within 3x
  of its 0.10 target. The last one is the lesson of three runs that shipped
  their headline term at the wrong mass and found out in the post-mortem.
- **25M:** `Player/Touch Height` rising rather than falling. Direction, not
  level -- 25M is too early for the level.
- **AT ANY TIME:** `RewardMass/AirTouch` share > 0.25 (the air-carry farm that
  held p15 for 400M steps at 0.50), or `AirTouch/Backward Share` > 0.30
  against its 0.5 chance value (the direction factor has stopped biting).
- **AT ANY TIME:** `Episode/Mean Steps` rising above 500 while
  `RewardMass/Goal` share falls (the possession farm returning in any form).

**The standing drift to watch, which is maintenance rather than a kill.**
`goal/ceiling` is 1.46x and falls as the bot improves, because the ceiling is
`Average Step Reward x 100` and that rises with skill. Cutting the dribble farm
should push it back toward 1.75x. **When it drops below 1.3x, raise `goal` or
cut a dense term** -- below 1.0x, continuing beats scoring and the bot will
correctly decline open nets, which is exactly what it did at 0.89x.

**Frozen:** gamma 0.99, policyLR/criticLR 2e-4, tsPerItr 50k, obs Relative,
spawn Random, `maskActions` false, the action parser, `infiniteBoostChance`
0.1, `goal` 25, `touchGoalAccel` 45 at exponent 2, `airTouchDirectionExponent`
1, `entropyTarget` 0.40, and all six external patches (verify with
`scripts/apply_external_patches.py --check`).

## PRE-REGISTERED: p13strike (not yet run)

**Two concepts, declared as such, with separable instruments.** Budgets are
SOLVED, not chosen -- `scripts/solve_budgets.py` against a target reward-share
vector, calibrated on a 2M-step probe resumed from `p12goal/250006016` with the
entropy controller off so the policy could not move.

**1. Target-entropy controller** (5th and 6th external patches). `entropyScale`
becomes a controlled variable, adjusted each iteration in log-space toward
`entropyTarget = 0.40` -- SAC's automatic temperature tuning (Haarnoja et al.
2018) applied to PPO's entropy bonus. A FIXED coefficient cannot hold an entropy
floor, because the bonus is proportional to H and therefore weakens exactly as H
falls: p12 ran `Policy Relative Entropy Loss` 1.01 -> 0.12 while entropy fell
0.71 -> 0.146. Multiplied by log(90) that is an EFFECTIVE ACTION COUNT of 1.9 of
90 -- near-deterministic. 0.40 is 6.0 effective actions; p7approach's 0.69, the
run where "no reward conclusion is available from a policy that cannot move",
was 22.3. The only framework-comparable external number this project has is a
GigaLearn run reporting 0.4786 at 197M, which a peer called "a bit low".

**2. Pay for what the ball does.** `TouchGoalAccel` goes convex (`sign(x)|x|^2`,
exponent a config field) and the whole budget vector is re-solved. Targets:
TouchGoalAccel 0.364, SpeedToBall 0.220, Goal 0.200 (anchor, weight frozen at
10.0), TouchEdge 0.065, SaveBoost 0.050, FaceBall 0.045, AirTouch 0.030,
PickupBoost 0.020, Air 0.006. **The proximity block goes 0.688 -> 0.330 and the
ball block 0.247 -> 0.564.** That inversion is the run. Two supporting fixes
that are not variables: `AirTouchReward` becomes a RISING EDGE (mandatory at the
new budget -- the p12 per-contact-step form would pay ~170 touch-units/sec for a
ceiling-height air carry), and `REFERENCE_EPISODE_SECONDS` 11.4 -> 26.0, which
cancels out of a share solve and only makes the budget fields read honestly.

**Why convex.** A linear touch term is indifferent to CONCENTRATION: the
goal-directed delta-v needed to score is fixed by the length of the field, so
five 400 uu/s pokes pay exactly what one 2000 uu/s strike pays, and the other
84% of the stack broke that tie toward pokes. At exponent 2 an 80 kph strike is
worth 16x a 20 kph poke rather than 4x. It is preferred to StrongTouch's hard
555.6 uu/s floor because a floor has no gradient below it -- at the 80 kph a
"strong touch" ought to mean, a floor would read identically zero today, since
the mean touch is 15.2 kph.

**Why AirTouch 12.89.** p12 measured the price of a takeoff: `Critic/TD Delta
Jump` -0.2249 against `NoJump` -0.0199, and `gamma*V - V` is -0.015 at V ~ 1.5
before anything happens, so the excess is -0.205 standardized = **-0.81
touch-units** at `Returns STD` 3.939. ~91% of sampled jumps come from the
exploration floor, which mixes uniformly over valid actions independently of
state, so that comparison is close to a randomized trial. At budget 2.0 a
realistic aerial (ball z 800, 1.0 s aloft, min = 0.391) paid 0.78 against 0.81 --
**4% below break-even**, which is the exact signature of a behaviour that appears
and decays rather than establishing or vanishing. At 12.89 it pays 5.04.

**Calibration, and what it caught.** The analytic solve off p12's summary
statistics was accurate to 5.6% for TouchGoalAccel (the one unknown was E[x^2],
estimated as 2E[|x|]^2 on a heavy-tailed assumption) but **1.63x wrong for
AirTouch** -- air contacts are more clustered than the average contact, so
edge-gating cost more mass than the overall 1.41 steps-per-sequence implied. A
verification probe with the final budgets returned every non-anchor term within
10% of target (TouchGoalAccel 0.3603/0.364, SpeedToBall 0.2381/0.220, AirTouch
0.0277/0.030). The residual is the anchor's own sampling noise -- goals are rare,
`RewardShare/Goal` read 0.179 against 0.200 over 2M steps, and every other term
is uniformly ~8% high as a consequence. Not chased further; that would be
fitting noise.

**Attempt 1 discarded at 8.4M steps (2026-08-19), controller windup.** The
entropy controller shipped without anti-windup and it is broken on a fresh
init. A new policy starts at `Policy Entropy` ~0.98, far ABOVE the 0.40 target,
so the controller correctly wants no bonus -- but it integrates that large
one-sided error for the whole transient. Measured: `Entropy Scale` 0.002 ->
0.000835 (1.7M) -> 0.000161 (5.0M) -> **0.000055 (8.4M) and still falling**, on
course for its floor by ~13M. Recovery afterwards would have needed 100M+ steps
at gain 0.05, so the run would have been an `entropyScale ~ 0` run: LESS
exploration than p12's fixed coefficient, which is the exact opposite of the
intent, and it would have looked fine at the 10M gate. Fixed by engaging the
controller only once entropy first REACHES the target, raising the gain 0.05 ->
0.15, and making the floor a fraction of the engagement scale rather than an
absolute 1e-5. `Entropy Controller Engaged` is now published so the handover is
readable off the graph. **Lesson, and it is the run-log's own rule turned on
itself: a controlled variable needs its controller's transient predicted in
writing before the run, exactly like any other prediction.**

**Attempt 2 discarded at 48.7M steps (2026-08-19), and the error was the
calibration, not the design.** The budgets were solved against a probe RESUMED
FROM p12goal's converged checkpoint -- a bot touching the ball 11.3 times per
episode -- and then the run was started FRESH. Under a random policy the five
EVENT terms have almost no events to pay for, so they collapse and the four RATE
terms absorb everything. Measured against target: TouchGoalAccel **0.05x**,
TouchEdge **0.02x**, AirTouch 0.15x, Goal 0.17x; Air **19.5x**, SaveBoost
**6.6x**, FaceBall 3.5x, SpeedToBall 1.4x. Air 0.117 + SaveBoost 0.330 +
FaceBall 0.156 = **60% of reward mass collectable without ever touching the
ball**, which is p1air's do-nothing attractor rebuilt, and the policy found it
correctly: `Player/In Air Ratio` **0.898** (p6budget's jump-and-hang to three
digits), `Jump When Grounded Upright` **0.454** against a 0.20 null,
`Player/Boost` **46.3** against 6-7 in every other run, `Episode/Mean Steps`
**185** = the 12 s no-touch timeout, `Touch/Edge Rate` **0.0003** against
p8ref's 0.0085 at the same step count, and `Player/Velocity Alignment`
**0.3469 = 1.09x the 0.3183 planar null**, i.e. the bot never learned to drive
at the ball at all. **The 10M share gate fired at ITERATION ONE and was not
read** -- first-20-iteration shares were already 69x and 28x off target.

**The general lesson, which outlasts this run: a target-share solve is only
valid when the calibration policy IS the run policy.** Event-term share scales
with event RATE, and the event rate is the thing a run is trying to change, so
no single static share vector is correct at both ends of a fresh run. Share
targeting is the right instrument for a RESUMED run and the wrong one for a
cold start; a cold start has to be calibrated on a cold policy, and then only
its early shares mean anything. `scripts/solve_budgets.py` now refuses a solve
whose probe checkpoint does not match the run's starting checkpoint.

**Attempt 3 therefore RESUMES `p12goal/250006016`,** which makes the existing
calibration exactly valid from step 0 and tests the actual question -- what a
bot that already reaches and controls the ball does once striking is what pays.
The original objection to resuming (p12's policy sits at entropy 0.146 = 1.9
effective actions and cannot respond) is precisely what the entropy controller
removes: resuming BELOW target engages it on iteration 1. Predictions below are
already written as deltas from p12's endpoint, so they carry over unchanged.

**AMENDMENT filed at 11M steps of attempt 3, BEFORE the 25M gate it changes, so
that it is an amendment and not a retro-fit.**

`Critic/TD Delta Jump` was pre-registered as the primary air readout because it
is entropy-robust. It is. **It is not SELECTIVITY-robust, and I did not
anticipate that.** The metric averages `gamma*V(s') - V(s)` over ALL
grounded-upright decision points, and ~91% of the jumps in it are eps-floor
forced, i.e. uniformly random with respect to state. Once a policy learns to
jump SELECTIVELY -- only when a jump is actually useful -- V(s) rises to include
that option value while the forced random jumps still land in states where
jumping is wrong. The gap therefore WIDENS as deliberate jumping gets better,
which is the opposite of what the gate assumed.

Measured at 11M: `Critic/TD Delta Jump` -0.2249 -> **-0.2677** (gate wanted
> -0.15) while `Touch/Had Jumped` went 0.0564 -> **0.5268** and
`Action/Jump When Grounded Upright` went 0.0039 -> **0.0172 = 4.3x the 0.0040
eps-floor**. p12's read exactly 1.00x the floor -- total extinction. **Floor
jumping has come off the floor for the first time since p1air, and the gate
metric says it got worse.**

So the 25M air gate is **replaced, not waived**: `Touch/Had Jumped` > 0.25 and
`Action/Jump When Grounded Upright` > 2x the eps-floor. `Critic/TD Delta Jump`
stays published and stays interesting, but as a measure of what a RANDOM jump
costs, which is a different question from whether a chosen one pays.

**Flip finding at 11M, with its null, and it reframes the whole air problem.**
`DefaultAction`'s 18 jump actions are pitch x roll x boost = 3 x 3 x 2 (yaw is
skipped when jump is set), and RocketSim computes dodge direction as
`(-pitch, yaw + roll)` (`Car.cpp:688`), so roll supplies the sideways component
and all eight dodge directions are representable. Only pitch == 0 AND roll == 0
gives a non-dodging jump: **2 of 18, so the uniform null for
`Flip/Neutral Share` is 0.111.** Measured: **0.856 = 7.7x the null.**

**The bot is not failing to DISCOVER the flip. It is actively selecting the
non-dodging jump, 7.7x above chance.** That kills the obvious remedy: Zealan's
"add more jump actions to a discrete action parser" is aimed at a policy that
does not sample the action, and 88.9% of our jump actions already dodge. Adding
more cannot help something the policy is rejecting on merit.

The merit is real, too. A neutral jump preserves BOTH the flip and full air
control; a dodge commits ~1.25 s of rotation. Against a ball whose arrival you
cannot yet time precisely, the adjustable option dominates -- and dodges only
start paying once timing is good, while timing only improves by dodging. That is
a bootstrap problem, not an incentive problem, and it is exactly what the
`strike` curriculum entry was written for ("ball at jump height, car already
rolling at it with pace, so the only open decisions are when to leave the ground
and whether to flip"). It has been disabled since p8ref's reproduction.
p2low's null -- curriculum cannot teach an action the policy never samples --
does not apply any more: both of its premises (no exploration floor, P(jump) =
0.0000) are now false.

**The flip ledger, derived from RocketSim's own constants, so p14 does not have
to argue about it.** A forward flip adds **exactly +500 uu/s**
(`FLIP_INITIAL_VEL_SCALE = 500`, and the speed multiplier
`((maxSpeedScaleX - 1) * forwardSpeedRatio) + 1` is exactly 1.0 for a forward
dodge because `FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1`). **There is NO
flip-specific bonus in the ball collision** -- `BALL_CAR_EXTRA_IMPULSE_FACTOR_
CURVE` (`Arena.cpp:315-326`) keys purely off `relSpeed = |ballVel - carVel|`, so
a flip helps hit force only by raising relative speed at contact.

That makes it arithmetic. A contact at ~700 uu/s relative gives extra impulse
700 x 0.639 = 447; with the flip, 1200 x 0.611 = 733, i.e. **+64% ball speed**.
Squared by the convex term that is **2.69x the touch reward**: at the measured
`Touch/Goal Accel Value` 0.0195 and budget 62.14, **1.21 -> 3.26 touch-units per
contact**, plus ~0.12 from the extra closing speed feeding SpeedToBall. So

    flip pays iff P(connect | flip) / P(connect | no flip) > 1.35/3.40 = 0.40

**A flip need only connect 40% as often as a normal approach to be worth
taking.** Under p12's LINEAR term the same comparison was 1.64x and the
break-even ratio 0.66, so convexity has already moved this a long way.

**And the cheapest version is in the bot's hands already:** it jumps into the
ball on 53% of touches and then spends the second jump on a NEUTRAL double jump
(`Flip/Delay Seconds` 0.243). That is the same action slot -- the only
difference from a forward dodge is `pitch = -1` -- and it is trading +500
forward for +292 up (`JUMP_IMMEDIATE_FORCE`) at a touch height of 146, where the
vertical is worth almost nothing.

**Conclusion for p14: the incentive is not the problem and must not be
"fixed".** The payoff is conditional on connecting, and the exploration floor
samples dodges uniformly across states, so the good moments are a small fraction
of flip samples, the MEAN advantage is ~0, and PPO -- which sees the mean --
suppresses it. The policy has correctly learned that flipping at a random
instant is bad. The lever is a state distribution concentrated on the moments
where flipping is right, i.e. the `strike` scenario, not another reward term.
Identified costs (`FLIP_PITCHLOCK_TIME` 1.0 s of no pitch control, ~0.04
touch-units of forfeited FaceBall; and losing the jump so a miss recovers
slowly) are nowhere near large enough to explain a 7.7x preference for neutral.

**Speed-flips: `FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9` against forward's 1.0,
and the side component DOES scale with `forwardSpeedRatio`** -- which is the
mechanic. Representable here (diagonal pitch/roll combinations are 8 of the 18
jump actions), but converting it to forward speed needs an air-roll correction
mid-dodge. Not designed for.

**Baseline, p12goal at 250M:** `Touch/Hit Force` 422, `Touch/Above 450` 0.037,
`Player/Touch Height` 155, `Critic/TD Delta Jump` -0.2249, `Touch/Had Flipped`
0.0008, `Touch/Edge Rate` 0.0291, `Episode/Mean Steps` 390, alignment 0.636,
`Policy Entropy` 0.146.

**Predictions:**

1. **`Touch/Hit Force` > 700 at 100M** (from 422), reversing a three-run decline
   of 878 -> 551 -> 422. PRIMARY. **Note the confound runs the RIGHT way:**
   higher entropy means noisier control and a hard touch needs a precise
   approach, so raising entropy pushes this metric DOWN. A pass is therefore
   strong evidence; only a failure is ambiguous.
2. **`Critic/TD Delta Jump` rises above -0.10** (from -0.2249). This REPLACES
   `Jump When Grounded Upright` as the primary air readout, because raising
   entropy moves every action-rate metric mechanically toward its null and the
   nulls in `docs/metrics.md` assume a fixed exploration level. TD Delta Jump is
   entropy-robust and in fact improves with entropy: ~100x more jump samples
   fixes the critic's coverage of post-jump states, which is the one weakness in
   the -0.81 estimate.
3. **`Touch/Edge Rate` falls below 0.022** (from 0.0291) while steps per contact
   sequence (`Ball Touch Ratio` / `Edge Rate`) stays under 1.6. Fewer and harder,
   not a carry.
4. **`Touch/Above 450` > 0.10 and `Player/Touch Height` > 200** (from 0.037/155).
5. **`Policy Entropy` holds in [0.35, 0.45]** with `Entropy Scale` settling
   rather than railing at a bound.

**Kill criteria:**

- **10M:** `Obs/Non-Finite Rate` = 0; `SB3 Clip Fraction` in [0.02, 0.25];
  `Entropy Scale` strictly inside (1e-5, 0.2); `Policy Entropy` moving toward
  0.40 rather than away; **and every `RewardShare/*` within 2x of its target.**
  That last one is new and is the whole point of the solver -- three consecutive
  runs shipped their headline term under 4% of reward mass and nobody found out
  until the post-mortem.
- **25M:** `Critic/TD Delta Jump` > -0.15. If the price of a takeoff has not
  moved with AirTouch at 12.89 AND the proximity block halved, then air is not a
  pricing problem and no further air budget will fix it -- which would send p14
  at the action parser (the guide's own remedy: "add more jump actions... doubling
  the jump actions seems to be enough to eliminate the need for air rewards")
  rather than at the reward.
- **50M:** `Touch/Hit Force` > 550, i.e. genuinely above p11's and not merely
  falling more slowly.
- **AT ANY TIME, the do-nothing ceilings**, added after attempt 2 measured what
  they look like: `RewardShare/Air` > 0.05 or `RewardShare/SaveBoost` > 0.15.
  Both terms are collectable while doing nothing -- floating, and not spending --
  so both inflate the moment event rates fall, and together they are the
  attractor that killed attempt 2.
- **AT ANY TIME, the two farm ceilings.** `RewardShare/AirTouch` > 0.35 is the
  float/air-carry farm. `RewardShare/TouchGoalAccel` > 0.60 **while
  `Episode/Mean Steps` is rising** is the SHOT-SPAM farm -- blasting the ball
  goalward without scoring, which convexity invites and which is already
  rational, since scoring ends the episode and forfeits the rest of the stream.

**Pre-committed tiebreak, filed BEFORE the run so it is a plan and not an
excuse.** If prediction 1 fails, the run does not get to blame entropy. It
triggers exactly one rerun at `--entropy-target 0 --entropy 0.002` with an
identical reward stack, ~32 minutes, to separate the two changes.

**Frozen:** gamma 0.99, policyLR/criticLR 2e-4, tsPerItr 50k, obs Relative,
spawn Random, `maskActions` false, the action parser, `goal` 10.0,
`infiniteBoostChance` 0. LR 1e-4 and tsPerItr 100k are both guide-prescribed for
this stage and both now wrong; they are p14, deliberately not p13.

**Stop at 100M** and extend only if `Rating/1v1` is still climbing. p12 ran to
250M and bought a rating doubling between 100M and 175M and then nothing.
`Rating/1v1` is measured against this run's OWN version pool, so it is not
comparable to p12's 108.

## p12goal: the pre-registration as filed (run 2026-08-19, outcome in the table above)

**Three changes, one concept: the bot has never been told the net exists.**
Declared as not-one-variable, with separable metrics.

1. **`StrongTouch` -> `TouchGoalAccel`** (budget 3.0 carried over unchanged, so
   only what it MEASURES moves). Signed change in goal-directed ball speed at
   the moment of contact. Touch-gated on purpose: the continuous
   `VelocityBallToGoalReward` is known-bad here (p1probe-b measured it absorbing
   67% of reward mass as passive ball noise; p1probe-h found removing it changed
   nothing).
2. **`Goal` 10.0**, symmetric. Deliberately not huge -- see the budget comment.
3. **`AirTouch` 2.0**: `min(airTimeFrac, heightFrac)` per touch. A wall shot
   scores exactly zero because a car on a wall is `isOnGround`.

**Baseline, p11boost at 97.7M:** `Touch/Hit Force` 551, `Touch/Above 450` 0.081,
`Player/Touch Height` 194, `Jump When Grounded Upright` 0.0044, `Player/Boost`
9.4, `Episode/Mean Steps` 1667 and rising, alignment 0.718, edge rate 0.0206.

**Predictions:**

1. **`Touch/Hit Force` stops falling and rises above 700.** Direction is the
   only thing that separates a useful touch from a poke, and this is the metric
   that has fallen in both of the last two runs.
2. **`Episode/Mean Steps` FALLS below 1200.** Goals are the only thing that can
   end an episode, and nothing else in this run changes that. If episodes keep
   growing, the bot still is not scoring and the goal term is inert -- which
   would be the clearest possible signal that a hard cap is needed.
3. **`Touch/Above 450` rises above 0.15** and `Player/Touch Height` above 260.
   This is what AirTouch is for, and unlike p11 it pays for a behaviour the bot
   already performs 8% of the time rather than one it must discover.
4. **`Player/Boost` rises above 15 WITHOUT the boost budgets changing.** This is
   the real test of the p11 diagnosis: if boost was only unattractive because
   there was nothing to spend it on, giving air touches value should move boost
   on its own. **If boost stays flat while `Touch/Above 450` rises, the p11
   diagnosis was wrong.**
5. **Floor jumping finally moves: `Jump When Grounded Upright` above 0.02**
   (4.6x the eps-floor, vs 1.1x now). AirTouch cannot be collected from a wall,
   so leaving the floor is the only way to earn it.

**Kill criteria:**

- **10M:** entropy falling, clip in [0.02, 0.25], `Obs/Non-Finite Rate` = 0.
- **25M:** `RewardShare/AirTouch` > 0.02 and `RewardShare/TouchGoalAccel` > 0.03.
  Either below and the new terms are inert like SaveBoost was, which is a budget
  problem and not worth 100M steps to confirm.
- **50M:** `Touch/Hit Force` > 620 (i.e. rising, not falling). If it is still
  falling with a signed goal-directed term in the stack, the poke farm is deeper
  than direction and needs the episode cap plus a rethink.

## Standing lesson from p7approach: write down the null

`Player/Velocity Alignment`, `FaceBall/*`, `Action/Jump When Grounded` and every
other rectified-cosine or action-rate metric has a **computable chance value**,
and a measurement is meaningless until it is compared against one.

- rectified cosine, uniform direction in the ground plane: `E[max(0,cos)] = 1/pi = 0.3183`
- rectified cosine, uniform direction in 3D: `E[max(0,cos)] = 0.25`
- `Action/Jump When Grounded`: 18 of the 42 actions a grounded car may pick
  press jump, so the uniform prior is `18/42 = 0.4286`

This project spent eight runs reading `Velocity Alignment ~ 0.30` as a low but
real number. It is the null. Nothing had been learned. The jump-rate prior was
computed correctly and used well (it is what made the p1air extinction
visible); the alignment null was never computed at all.

**Every behavioural metric added from here on ships with its null value in
`docs/metrics.md`, and no run conclusion may cite a metric whose null is
unknown.**
