# Hivemind Bot - Extracted Code Comments & Documentation

This document contains all design notes, derivations, empirical measurements, and code documentation extracted from the source files in `bot/src/`.

---


# Config.h

// Everything tunable lives here. Defaults are sized for the training machine
// (Ryzen 3600, 6 cores; RTX 2060, 6 GB VRAM).

// Relative weights for how often each scenario spawns. They control what the
// policy practises, not which model runs -- there is one model. Re-derive
// them from telemetry at each phase gate.

// Ordinary play. Keep this dominant.

// Car spawned next to the ball, so contact is nearly free -- mostly
// training on a situation the bot already gets for free at this point.

// Cars end up airborne whether or not we spawn them there, and landing
// on wheels is a prerequisite for everything else.

// The jump-flip strike: ball at jump height, car already rolling at it with
// pace, so the only open decisions are when to leave the ground and
// whether to flip.

// Ball in the air, cars on the ground: the lesson is "the ball is up, go
// and meet it", which the touch reward pays for directly.

// Full kickoffs mixed into training; the one policy plays its own
// kickoffs, so it has to practise them here.

// Zero until the reward function pays for them. Zero-weight setters are
// dropped entirely, so these cost nothing while disabled.

// --- Reward budgets --------------------------------------------------------
//
// Every reward weight in this project is declared as a BUDGET and converted to
// a per-step weight in exactly one place (Hive::GeneralRewardSpecs). Nothing
// may be written as a bare per-step float.
//
// This exists because p1air's `grounded = 0.05` integrated to 9.0 units per
// episode -- nine goals per episode for holding still on the wheels -- and
// nobody noticed, because nobody wrote down the integral. Declaring the
// integral makes that class of mistake unrepresentable.
//
// THE UNIT IS ONE BALL TOUCH, not a goal. A goal was the unit until
// p7approach and it could not be audited: goals arrive 0.116 times per episode
// and 49% of those are `Scenario/Defend` conceding rather than anyone scoring,
// so the ledger had to be reconstructed by hand in every post-mortem. A touch
// occurs 0.16-2 times per episode and is read straight off `Touch/Edge Rate`
// and `Player/Ball Touch Ratio`.
//
// Restating the previous stack in the new unit is what showed the problem.
// p7approach paid 1.83 touch-units for a whole episode of PERFECT approach;
// the guide's stack pays 20.5. An 11x difference that neither spec noticed,
// because the two were denominated in different currencies.

// tickSkip 8 at RocketSim's 120 Hz.

// MEASURED, not assumed. p6budget's `Episode/Mean Steps` read 171.0 over 100M
// steps and p7approach's read 166.6. noTouchTimeout caps a never-touching bot
// at 180 steps and goals end episodes early; at a touch ratio near zero almost
// every episode runs to the timeout, which is why this sits just under it.
//
// RE-DERIVED for p13strike. p12goal's `Episode/Mean Steps` read 390 (26.0 s)
// and, unlike every previous run, that number is STABLE: goals now end
// episodes, so the length is set by how long the bot takes to score rather
// than by an unbounded no-touch drift. 11.4 s was understating the episode by
// 2.28x, so every rate budget was over-delivering by that much.
//
// Changing it is normally "a reward change in disguise" and that is why it sat
// stale through p9-p12. It is free HERE because p13's budgets are solved
// against a target REWARD SHARE vector (scripts/solve_budgets.py), and the
// conversion constant cancels out of that solve: the shares are what is being
// held fixed, so the constant only decides what number gets written in the
// budget field. Taking the honest value now means the field finally reads as
// what it claims -- touch-units earned over one real episode.

// Touch-units earned by holding a behaviour perfectly for one reference
// episode, converted to the per-step weight that earns it.

// Touch-units of cost for one second of a condition, converted to per-step.

// All values are touch-units. This is the early-stage stack from Zealan's
// RLGym-PPO-Guide (making_a_good_bot.md) in the guide's own proportions --
// touch 50, speed-to-ball 5, face-ball 1, air 0.15 -- divided through by the
// touch weight so that a touch is 1.0, then expressed as episode integrals.
//
// Nothing here is this project's invention, deliberately. See
// docs/superpowers/specs/2026-08-18-known-good-baseline-design.md.

// --- Event: earned per occurrence ---

// THE UNIT: one full-power ball touch, 1.0 by definition. Scales with hit
// force (|delta ball velocity|) and pays exactly ZERO below RLGymCPP's
// 20 kph floor, which is 555.6 uu/s -- an order of magnitude above the tens
// of uu/s a dribble carry imparts. Saturates at 130 kph = 3611 uu/s.
//
// Realized values will be well under 1.0: p1pay measured typical hit force
// around 1400 uu/s, which scores 0.39. That is fine and expected -- the
// budget is what scales it, and `Touch/Strong Value` is what measures it.
//
// This replaces p9rel's flat per-step touch, which the bot farmed by
// carrying the ball for 13% of all steps. It is the guide's middle-stage
// prescription verbatim: "scale the reward with the strength of the
// touch... a slight push that barely changes the velocity of the ball will
// give almost no reward, but a strong shot will give lots of reward."
//
// PROVISIONAL. Per roadmap D6 (no magic numbers without measurement), this
// stands until `Touch/Hit Force` says what a realized touch is actually
// worth; the first run that reports it is what sets the final value.
// THE UNIT: one maximal goal-directed strike, 1.0 by definition, saturating
// at the same 130 kph (3611 uu/s) StrongTouch used -- so the currency is
// unchanged and every earlier budget still reads the same.
//
// Replaces `strongTouch`, which paid for force in ANY direction. p11
// measured that failing: `Touch/Hit Force` 878 -> 551 while
// `RewardShare/TouchEdge` doubled. StrongTouch's floor is 555.6, so the
// average touch ended the run earning zero from it and the poke farm was
// paid entirely by the flat per-contact term. Direction is what separates a
// useful touch from any touch. See TouchGoalAccelReward.
//
// CONVEX for p13strike, and this is the run's thesis. p12 ran this term
// LINEAR in goal-directed dv, and a linear term is INDIFFERENT to
// concentration: the total goal-directed dv needed to score is fixed by the
// length of the field, so five 400 uu/s pokes and one 2000 uu/s strike pay
// exactly the same. Every other term in the stack broke that tie toward the
// pokes, because a poke leaves the ball inside re-contact range. Measured
// result: `Touch/Hit Force` 878 (p10) -> 551 (p11) -> 422 (p12), i.e. the
// average touch fell BELOW RLGymCPP's own 20 kph "weak touch" floor of
// 555.6 uu/s.
//
// r = sign(x) * |x|^touchAccelExponent breaks the tie. At exponent 2 an
// 80 kph strike is worth 16x a 20 kph poke (4^2) rather than 4x. It is the
// guide's own prescription -- "a slight push that barely changes the
// velocity of the ball will give almost no reward" describes a CONVEX
// function, and this project implemented it linear.
//
// A convex term is also a SOFT floor whose effective threshold rises by
// itself, which is why it is preferred to StrongTouch's hard 555.6 cutoff.
// A hard floor at the 80 kph the eventual target wants would be identically
// zero today (mean hit force is 15.2 kph) with no gradient anywhere, which
// is p11's inert-term failure amplified 4x.
//
// NOTE ON THE UNIT. Saturation is unchanged at 130 kph, so 1.0 is still one
// maximal goal-directed strike and every earlier budget still reads the
// same. But a REALIZED touch now earns far less of that 1.0 than it did
// under the linear form, so the BUDGET NUMBER IS LARGE AND THAT IS CORRECT:
// it is denominated in maximal strikes, and the run collects fractions of
// one. `Touch/Goal Accel Value` reports what a realized touch actually
// earns, exactly as `Touch/Strong Value` did for StrongTouch.
//
// SOLVED, not chosen. Target reward share 0.364, solved against p12goal's
// measured shares by scripts/solve_budgets.py.
//
// Why the number is so much larger than p12's 3.0, spelled out, because a
// budget this size looks like a mistake otherwise:
//   * p12 measured `RewardShare/TouchGoalAccel` 0.098 at weight 3.0, which
//     back-solves to E[|x|] = 0.0705 per contact step -- the average touch
//     moved the ball 255 uu/s = 9.2 kph toward the net.
//   * Squaring that collapses the realized value ~7x: assuming the
//     heavy-tailed exponential (E[x^2] = 2 E[|x|]^2) gives 0.00994.
//   * Holding the same share therefore needs ~19.6x the weight.
// 58.86 buys the SAME fraction of reward mass that 3.0 bought. It is not a
// promotion; it is the same share through a steeper curve.
//
// PROVISIONAL on that E[x^2] assumption, which is the one quantity that
// could not be derived from p12's summary statistics -- shares carry only
// the first moment. The calibration probe measures it directly:
//   scripts/solve_budgets.py --csv bot/build/metrics/main-p13cal.csv
//
// MEASURED. The probe (2M steps resumed from p12goal/250006016, entropy
// controller off so the policy could not move) returned share 0.3486
// against the 0.364 target, so the exponential assumption was accurate
// to 5.6% and the budget needed only a 1.056x correction. 58.86 -> 62.14.

// The convexity. 2 is shipped for p13; 3 (64x rather than 16x for 80 kph
// over 20 kph) is where this should end up, but only once mean hit force
// clears ~40 kph. The learnability constraint is the gradient available at
// the CURRENT operating point relative to the target one: (x_now/x_target)
// ^(p-1). At today's 15.2 kph against 80 kph that is 19% for p=2, 3.6% for
// p=3 and 0.7% for p=4. Move it on the measurement, not on a schedule.

// Opponent penalty scale for TouchGoalAccel.
// Unlike GoalReward which is 100% zero-sum (+1 scored, -1 conceded),
// this penalizes opponent goal-directed touches at 50% (0.5).
// This discourages allowing opponent attacks without making conceded
// touches overwhelmingly punitive.

// Fraction of TouchGoalAccel reward shared with teammates in multi-car modes.

// --- The scoreboard ------------------------------------------------------

// A goal, +1 scored and -1 conceded (already zero-sum).
//
// DELIBERATELY NOT HUGE, and this is the one place the obvious intuition is
// wrong. Zealan: "Don't give massive goal rewards! This *feels* like it
// makes sense because goals are the most important thing in the game.
// However... adding massive goal rewards early on in training simply slows
// down learning and decreases exploration. A giant goal reward will drown
// out every other reward you have." He reports training a bot to a high
// level with no goal reward at all.
//
// The mechanism here: in self-play a goal is +1 for one car and -1 for the
// other, so its MEAN is zero and its entire contribution is variance the
// critic cannot predict. Scaling it up scales the noise, not the signal.
// What teaches direction is the dense `touchGoalAccel` term; this one
// exists to break ties between behaviours the shaping rates equally, and to
// end episodes.
//
// 10.0 measured out at `RewardShare/Goal` 0.149 in p12, which is a sane
// place for a zero-mean term, so it is FROZEN for p13 and used as the
// ANCHOR of the budget solve: every other budget is scaled to hit its
// target share against this one's fixed mass. Freezing one weight is what
// makes the solve well-posed instead of scale-degenerate.
//
// Worth knowing what it can and cannot do: at gaeGamma 0.99 and 15 steps/s
// the value horizon is 1/(1-g) = 100 steps = 6.7 s, so a goal is discounted
// to 10 * 0.99^390 ~ 0.2 at the start of a 26 s episode and to 7.4 two
// seconds out. This term shapes the FINISH and nothing else; what teaches
// direction over a whole possession is touchGoalAccel.

// Arriving at the ball, worth a quarter of a full-power hit. Rising edge,
// so carrying pays this exactly once (see TouchEdgeReward). Kept small and
// non-zero because the bot still needs a signal for REACHING the ball --
// deleting it entirely would leave nothing paying for contact at all until
// the bot can already hit hard.
//
// CUT for p13strike (target share 0.065, from 0.082). The guide: "decrease
// the TouchBallReward a lot so that it is no longer the bot's top
// priority." The number barely moves but its standing does: against
// touchGoalAccel it goes from 1:1.2 to 1:5.4, so arriving at the ball stops
// being competitive with doing something useful once there.

// --- Rate: earned by holding the behaviour for one reference episode ---

// CUT for p13strike: target share 0.22, down from a measured 0.441. This is
// the first time in this project's history that the approach budget has
// been reduced. `RewardShare/SpeedToBall` + `FaceBall` has held 0.61-0.88
// of all reward mass in EVERY run (p8ref 0.761, p10 0.876, p11 0.778, p12
// 0.606) and no run has ever tested lowering it. 84.5% of p12's realized
// ledger paid for being near, pointed at and touching the ball; 9.3% paid
// for what the ball actually did.
//
// The guide's middle-stage instruction is explicit that the ball-to-goal
// term should be "a fair bit stronger than SpeedTowardBallReward" once the
// bot can hit the ball. It can: 11.3 contacts per episode.
//
// The number LOOKS like a small cut (17.1 -> 14.5) only because
// REFERENCE_EPISODE_SECONDS moved 11.4 -> 26.0 in the same commit. The
// per-step weight, which is what the policy sees, goes 0.1000 -> 0.0372.
//
// This is also the biggest single lever on the air game, and that is not a
// side effect. Most of the measured -0.81 touch-unit price of a takeoff is
// forfeited dense income; cutting the dense income cuts the price.

// Kept at the guide's speedToBall/5 ratio through the cut (14.51/5 = 2.90
// would be the strict ratio; 1.585 is what the share solve returns, because
// FaceBall's realized value per step is much closer to its maximum than
// SpeedToBall's is, so equal budgets do not buy equal mass). SIGNED, so
// driving backwards at the ball is punished rather than merely unpaid --
// see the pairing note in Rewards.h. Target share 0.045.

// --- The boost economy -------------------------------------------------
//
// p10touch's bot found air dribbling off the wall and could only sustain it
// when it happened to have boost, running at `Player/Boost` **7.3 out of
// 100** all run. Aerial play is boost-gated in a way ground play is not,
// and nothing in the stack had ever paid for boost. This is the guide's
// middle-stage prescription and the most direct lever on the air game that
// does not involve paying for height directly.

// sqrt(boost/100) per step. The sqrt is the point: it makes boost worth
// more the less you have, which is simply true in Rocket League -- 0 to 50
// is worth more than 50 to 100.
//
// Kept small because a per-step reward for a HOLDABLE state is a do-nothing
// attractor, which is what p1air's flat `grounded = 0.05` was. The guide:
// "If your bot is hogging boost and is afraid to use it, decrease the
// reward."
//
// Target share 0.05, up from a measured 0.035 -- essentially held, not
// promoted. p11 established that a boost economy cannot create air play on
// its own; nothing here revisits that.

// sqrt(new) - sqrt(old) on pickup, i.e. the increment of saveBoost's own
// potential, so a full grab from empty pays 1.0 x this budget. Being the
// derivative of a sqrt, it pays disproportionately for topping up when low,
// which is most of what the guide wants from small-pad behaviour.
//
// Target share 0.02, i.e. held at p12's measured 0.015.

// --- Air play ------------------------------------------------------------

// Per air touch: min(airTimeFrac, heightFrac). A maximal one -- 1.75 s
// airborne, ball at the ceiling -- is worth 2.0, i.e. two thirds of a
// maximal goal-directed strike. A realistic air dribble touch at z~1000
// after 1.2 s aloft scores min(0.686, 0.489) = 0.489, so ~0.98 touch-units:
// clearly worth more than the 0.25 for merely arriving, without dwarfing a
// good strike.
//
// A wall shot scores exactly ZERO: a car on a wall is `isOnGround`, so its
// airTime is 0 and the min collapses. That is the guide's anti-farm device
// and this bot is the exact case it was written for.
//
// RAISED 2.0 -> 7.89, and this is a DERIVED number, not a bigger guess.
// p12 measured what leaving the ground actually costs: `Critic/TD Delta
// Jump` -0.2249 against `NoJump` -0.0199, and since gamma*V - V is -0.015
// at V ~ 1.5 before anything happens, the excess attributable to jumping is
// -0.205 standardized = -0.81 TOUCH-UNITS at `GAE/Returns STD` 3.939. That
// comparison is close to causal: the policy's own P(jump) is 4e-4 against
// the exploration floor's 4.0e-3, so ~91% of sampled jumps are
// floor-assigned, and the floor mixes uniformly over valid actions
// INDEPENDENTLY OF STATE -- the fourth external patch doubles as random
// treatment assignment on the jump action.
//
// At budget 2.0 a realistic air dribble touch (ball z 800 -> heightFrac
// 0.391; 1.0 s aloft -> airTimeFrac 0.571; min = 0.391) paid 0.78 against
// that 0.81. It was set 4% BELOW break-even, which is exactly the signature
// of a behaviour that appears and decays instead of establishing or
// vanishing -- p10touch, p11 at 42-56M, and p12 not at all.
//
// 7.89 pays that same touch 3.08 touch-units, a 3.8x margin over cost. The
// over-payment is deliberate: two runs have already found this behaviour
// and lost it at margins near 1.0. The starting reward share is only ~0.030
// and that is accepted on purpose -- what decides a jump is the per-EVENT
// advantage, not the aggregate mass. The aggregate is guarded from the
// other side instead: `RewardShare/AirTouch` above 0.35 at any point is a
// kill criterion, because a term that becomes the run's argmax is p9rel's
// dribble farm wearing a different hat.
//
// Paired with the RISING-EDGE fix in AirTouchReward, which is not optional
// at this budget: the p12 form paid per contact STEP, so an air carry at
// ceiling height would have earned ~170 touch-units per second.
//
// MEASURED, and this is the one term the analytic estimate got wrong. The
// probe returned share 0.0186 against the 0.030 target, needing a 1.63x
// correction to 12.89 -- so air CONTACTS are more clustered than the
// average contact, and edge-gating cost the term more mass than the
// overall 1.41 steps-per-sequence figure implied. Exactly the sort of thing
// the budget framework used to miss by an order of magnitude and now
// catches for the price of a two-minute probe.
//
// At 12.89 a realistic aerial pays 5.04 against a cost of 0.81 -- a 6.2x
// margin, larger than intended and accepted: the share is still only 0.030,
// and what decides a jump is the per-event advantage.
//
// p14: 12.89 -> 112.5. The budget rises 8.7x while the TARGET SHARE only
// doubles (0.030 -> 0.060), because squaring heightFrac collapses the
// realized value ~4.4x. Solved on a 2M probe resumed from
// p13strike/350001792, which is the checkpoint p14 starts from.

// How steeply the air-touch reward scales with ball height. 1.0 is the
// guide's linear form, which p13 measured being collected by ordinary
// jump-touches at z ~350. 2.0 keeps paying those -- a jump to reach a high
// ball is a real aerial, just a small one -- while making a genuine one
// worth several times more. See AirTouchReward.

// 0.15/50 per step over 171 steps: 2.5% of the dense budget. Measured to be
// ~50x too small to pay for the traction and contact a jump costs (p8ref
// ledger), so it does NOT keep jumping alive and is not expected to.
// Raising it enough to break even would put ~37% of the budget on floating,
// which is p1air's do-nothing attractor inverted. Air is a problem for an
// air-TOUCH term, not for this one.
//
// HELD at p12's measured share of 0.006 rather than given a target, and
// that is deliberate: this is the float attractor's own term and the lever
// this project has already found to be the wrong one twice. Per-step weight
// goes 0.00300 -> 0.00224. The budget NUMBER rises only because
// REFERENCE_EPISODE_SECONDS was corrected 11.4 -> 26.0.

// Forward closing velocity gained from dodging/flipping towards the ball
// (normalized by 500 uu/s impulse). Provides the discovery gradient for
// ground flipping and speed-flips to traverse the pitch and preserve boost.

// Play a fraction of games against saved old versions.

// Chance per iteration that it trains against an old version.

// How often to snapshot the current policy into the version pool.
// GigaLearn's own default (25M) suits multi-billion-step runs but barely
// engages in a short comparison run; 5M gives a usable pool quickly.
// Raise it towards 25M for long runs.

// Skill tracking is what makes two runs comparable, so it is worth
// enabling even without trainAgainstOldVersions.

// Evaluation arenas compete with training for CPU; keep well under the
// core count.

// Iterations between evaluation runs.

// Players per team the observation reserves space for. Changing this
// changes the observation width, which invalidates every existing
// checkpoint -- treat it as permanent once a run you care about starts.

// Where episodes start from. `Random` is RLGymCPP's RandomState with
// (randBallSpeed, randCarSpeed, carsOnGround) = (true, true, false), which
// is what Zealan's guide specifies and what every reference bot starts on:
// fully random positions and velocities, cars airborne on half of spawns.
//
// `Curriculum` is this project's 10-entry scenario mix. It has never been
// validated against anything, and it is the leading suspect for the
// Early/Late collapse that has stood since p1age -- Early(0-1s) touch rate
// 0.0081 against Late(4s+) 0.00025, a 32x gap, with the bot ending FARTHER
// from the ball than it started. A spawn distribution that hands the bot a
// favourable first second teaches the first second.
//
// The curriculum code stays in the tree; phase B measures it against
// Random rather than assuming it.

// Whether the action parser masks actions by situation. See Actions.h: the
// mask more than doubles the grounded jump prior (42.9% vs 20%) relative to
// every reference implementation. Must match at deployment.

// Which observation the policy sees. `Relative` is `Default` plus car-frame
// relative geometry for the ball and every other car (see RelativeObs.h);
// `Default` is RLGymCPP's DefaultObsPadded, which p8ref ran on.
//
// Changing this changes the observation WIDTH, so it invalidates every
// existing checkpoint. Must match at deployment -- a width mismatch throws
// at load, but a same-width layout mismatch would not, so `verify` checks
// it explicitly.

// Fraction of episodes spawned with a full tank that never drains. See
// InfiniteBoostState: a policy cannot learn the value of a resource it
// never has, and p10touch spent its whole run at 7.3 boost out of 100 while
// trying to air dribble.
//
// ZERO for p11boost, deliberately. p11's primary metric is `Player/Boost`
// and its first prediction is that boost rises above 25; infinite-boost
// episodes would inflate exactly that number and make the run
// uninterpretable. Turned on in p12 once the boost economy has been
// measured on its own. Settable at runtime with `--infinite-boost`.

// Fraction of reward shared between teammates on zero-sum terms (0.0 for
// pure individual in 1v1, 0.3-1.0 in NvN).

// --- PPO ---------------------------------------------------------------
// Guide, learner_settings.md: "values of 50_000 are good for early
// learning, but once the bot is actually hitting the ball, it should be
// increased to 100_000". Halving it from 100k also doubles the number of
// policy updates per unit of experience, which is what an under-updating
// run needs.

// Minibatch is the main VRAM knob (6 GB available).

// Upstream default; guide recommends 2 or 3.

// NOT the guide's 0.01. GigaLearn's entropy is NORMALIZED to [0,1] (divided
// by log(numActions)), so this scale is not comparable to rlgym-ppo's, and
// porting the number verbatim ports a value this project has measured
// breaking three times.
//
// Evidence: 0.035 and 0.018 pinned entropy near-uniform (p1probe-d, -g);
// 0.01 pinned it again in p7approach (`Policy Entropy` 0.709 -> 0.689 flat
// over 100M steps, `SB3 Clip Fraction` 0.0038 against a healthy 0.05-0.2,
// `Policy Relative Entropy Loss` 268 against a documented healthy <=0.1);
// 0.002 produced this project's only breakthrough (p1probe-f) and the
// RUNLOG says "Keep 0.002".
//
// Exploration does not depend on this alone: the fourth external patch
// floors every VALID action at 0.02/N probability, which is the only thing
// that has ever reversed an action extinction.

// --- Target-entropy controller (5th external patch) ---------------------
//
// Above 0, `entropyScale` stops being a constant and becomes a CONTROLLED
// variable: the learner adjusts it each iteration to hold measured
// `Policy Entropy` at this target. See PPOLearnerConfig.h for the
// derivation (SAC's automatic temperature tuning, Haarnoja et al. 2018).
//
// Why the fixed coefficient could not work: the entropy bonus is
// proportional to H, so as H falls the bonus weakens, which is
// self-accelerating. p12goal at scale 0.002 ran `Policy Relative Entropy
// Loss` 1.01 -> 0.12 while entropy fell 0.71 -> 0.146. The bonus faded
// exactly when it was needed.
//
// 0.40 is not a guess. GigaLearn's entropy is normalized by
// log(numActions), so multiplying by log(90) gives an EFFECTIVE ACTION
// COUNT: p12's 0.146 is 1.9 of 90 actions (near-deterministic), 0.40
// is 6.0, and p7approach's 0.69 -- the run where "no reward conclusion is
// available from a policy that cannot move" -- is 22.3. 6 effective actions
// is a policy with real preferences and genuine exploration, comfortably
// clear of both failure modes. It also matches the only
// framework-comparable external reference this project has: a GigaLearn run
// reporting 0.4786 at 197M steps, which a peer called "a bit low".
//
// NOTE: this makes every ACTION-RATE metric non-comparable with p12 and
// earlier, because raising entropy moves them mechanically toward their
// nulls. Read outcome metrics (`Touch/Hit Force`, `Critic/TD Delta Jump`,
// `RewardShare/*`) across this boundary, not `Action/*`. Settable with
// `--entropy-target`; 0 restores the old fixed-scale behaviour.

// Per-iteration multiplicative gain, in log space. 0.15, raised from an
// initial 0.05 that could not move the scale far enough inside one run --
// p12 showed 0.002 does not hold 0.40, so the controller has to be able to
// multiply it by 10x or more, and at 0.05 that took 46M steps.
//
// The controller does not engage until entropy first REACHES the target. Do
// not remove that: without it, the fresh-init transient (entropy starts
// near 0.98, far above target) integrates the scale down 36x within 8M
// steps and it never recovers. Measured on p13strike's first, discarded
// attempt.

// RAISED for p14: 0.99 -> 0.995, i.e. the value horizon 1/(1-gamma) goes
// 100 steps (6.7 s) -> 200 steps (13.3 s), and the half-life 4.6 s -> 9.2
// s.
//
// This is the term that makes TIME cost something, and p13 showed nothing
// else does. `Episode/Mean Steps` went 378 -> 672 = 44.8 s, and at gamma
// 0.99 a goal 20 s away is worth 10 * 0.99^300 = 0.49 -- effectively
// invisible, so shortening a 45 s cycle earns the policy nothing. At 0.995
// the same goal is worth 2.23, and at 0.997 it is 4.07.
//
// The bot's slow cycle is NOT aimless: `Player/Speed Sustained` is flat at
// 1287 while `Speed Towards Ball` fell 866 -> 660 and `Action/Handbrake`
// tripled, which is a bot driving full speed on wide arcs and trying to
// tighten them. It loops because it cannot drift or speed-flip, and it
// accepts the loop because convex TouchGoalAccel pays for AIM quadratically
// -- trading closing speed for a better-aimed hit is rational under the
// reward as written. Raising gamma is what puts a price on the time that
// trade costs, without paying for any mechanic directly.
//
// One step, not two. 0.997 gives the roadmap's ~15 s half-life and is the
// eventual target, but p13 already pushed `Critic Loss` 0.051 -> 0.203 and
// `GAE/Returns STD` 3.94 -> 7.14; a longer horizon makes the critic's job
// harder again, so this moves once and gets measured.

// Guide, learner_settings.md: "Bot that can't score yet: 2e-4".

// --- Bookkeeping -------------------------------------------------------

// Distinguishes runs. Without it, a second run resumes from the first
// one's checkpoints, which silently invalidates any comparison between
// them.

// -1 uses the clock

// Stream to RocketSimVis instead of training fast


# Verify.cpp

// 1. The checkpoint loads under the compiled-in ModelShape. A shape
//    mismatch throws here instead of silently misplaying in a match.

// 2. Deterministic inference is actually deterministic, and the policy is
//    not degenerate (always the same action regardless of state).

// Random controls step the arena into varied states.

// 3. Deployment env vars, if set, agree with compiled training values.


# Verify.h

// Checkpoint/deployment parity check: loads a checkpoint the way the RLBot
// client does and asserts it infers sanely. Returns 0 on pass, 1 on failure.


# env/Actions.cpp

*No substantial comments in this file.*


# env/Actions.h

// RLGymCPP's DefaultAction with the situational mask removed.
//
// WHY THIS EXISTS. `DefaultAction::GetActionMask` restricts a grounded car to
// 42 of the 90 actions, of which 18 press jump -- a 42.9% jump prior. Python
// RLGym's `LookupAction`, which every reference bot and Zealan's guide use,
// applies no mask at all, so its grounded jump prior is 18/90 = 20%.
//
// That factor of 2.1 is not cosmetic. Ground dwell is 1/p_jump decision steps
// and an air stint is ~15, so a UNIFORM policy is airborne 87% masked against
// 75% unmasked -- and p1advnorm measured `Player/In Air Ratio` 0.886 at a jump
// rate of 0.43, matching the masked prediction. This project has spent eight
// runs fighting air time that its own action mask doubles the prior of.
//
// The mask is not a bug. It stops the policy spending capacity on actions that
// do nothing (air-roll while grounded), which is real. It is simply an
// undocumented divergence from every implementation this project compares
// itself against, so the reproduction removes it and phase B measures whether
// putting it back helps.

// The ONLY place an action parser is constructed. Training, deployment,
// `verify`, `eval` and `spectate` all route through here, because a parser
// mismatch between training and deployment does not crash -- the bot loads,
// plays, and is quietly worse. Same reasoning as ModelShape.


# env/Curriculum.cpp

// Child setters are process-lifetime, matching upstream ownership.


# env/Curriculum.h

// Drop-in replacement for RLGC::CombinedState that remembers which child it
// last picked, so episode outcomes can be attributed to the scenario that
// spawned them. One instance per arena (CreateEnv makes one per env), so the
// last-picked name needs no locking.

// Drops weight<=0 entries, deleting their setters; throws if none remain.

// Empty until the first reset.

// Names of the entries that survived the zero-weight filter. Metrics
// report a zero sample for scenarios not picked this step; without that,
// rare scenarios' Share averages are biased upward.


# env/Env.cpp

// (randBallSpeed, randCarSpeed, carsOnGround) = (true, true, false), which is
// the RandomState configuration Zealan's guide specifies: random positions and
// velocities for both the ball and the cars, with cars spawning airborne on
// half of resets so they learn to land.

// Wraps rather than replaces, so the spawn distribution is unchanged and
// the only difference is whether the tank drains.

// CurriculumState drops zero-weight entries and remembers which scenario
// each reset came from, which is what the Scenario/* metrics report.


# env/Env.h

// Build the environment for game `index`. This is the function handed to
// GigaLearn's Learner; it is called once per game at startup.

// The spawn distribution training resets to. Exposed so the spectator shows the
// same distribution the policy is actually practising -- watching a different
// one is how you conclude a bot handles situations it has never seen.

// The scenario mix, when TrainConfig::spawn is Curriculum.


# env/Obs.cpp

// Build a full-size arena so padding is exercised at its maximum, which
// also validates maxPlayersPerTeam is large enough for the intended cars.


# env/Obs.h

// Which observation layout the policy sees. Lives here rather than in
// TrainConfig so the RLBot client can name it without pulling in the whole
// training config; both sides need it, because it determines the input layer.

// DefaultObsPadded plus car-frame relative geometry. See RelativeObs.h.

// RLGymCPP's DefaultObsPadded: absolute world-frame everything. Every run
// up to and including p8ref used this.

// Two observations, selected by ObsMode.
//
// `Default` is RLGymCPP's DefaultObsPadded: absolute, world-frame everything,
// with teammate/opponent slots zero-padded to maxPlayersPerTeam and shuffled
// each step, and orange mirrored so both teams share one policy. This is what
// every run up to and including p8ref used.
//
// `Relative` is that plus car-frame relative geometry for the ball and every
// other car -- see RelativeObs.h for why.
//
// Padding and shuffling also mean a slot cannot be told human from bot from
// empty except by content, so humans need no special handling anywhere.

// Create the observation builder. Caller owns the result. Both the mode and
// maxPlayersPerTeam must match between training and deployment: together they
// determine the network's input layer.

// Non-finite observation values seen since the last call, and the total
// checked. Reset by reading.
//
// EXISTS BECAUSE p11boost died at 29.8M steps with a CUDA device-side assert
// ("probability tensor contains either inf, nan or element < 0"). Everything
// upstream was finite -- entropy 0.538, KL 0.0053, reward 0.0624, GAE/Returns
// STD 2.113, Critic/V All 2.72 -- and the only GAE outputs that went NaN were
// the two that depend on critic value predictions. A NaN in an observation
// produces exactly that signature and nothing else in the telemetry would show
// it. This turns a crash into a number.

// Measure the observation width by building a throwaway arena and asking the
// builder how many floats it produces, rather than computing it by hand (which
// would silently drift if the layout changes).
// Requires RocketSim::Init() to have been called first.


# env/PlayPhase.cpp

// Flip Y so negative always means "towards our own goal", whichever team we are.

// Ordered most specific first.

// Checked before Defend so a tumbling car is not counted as shadowing.


# env/PlayPhase.h

// Play phases are metrics labels only, never routing: they describe what the
// single policy is doing at a given moment, for wandb, not which model runs.

// Per-phase counters for training metrics.


# env/RelativeObs.cpp

// Express a world-frame vector in the car's own basis. RotMat's rows are the
// forward/right/up unit vectors, so this is the change of basis the network
// would otherwise have to learn from the raw orientation vectors.

// The block whose absence defined p8ref: where the target is and how it is
// moving, from the car's point of view.
//
// Ten floats: a unit direction (scale-free, and literally the argument of both
// dense reward terms), a separate distance, the full relative offset, and the
// relative velocity. Direction and offset are redundant with each other by
// construction; the unit vector is kept because it stays well-conditioned at
// every distance, while the offset carries the magnitude the unit vector drops.

// Absolute block, byte-for-byte what DefaultObs::AddPlayerToObs emits, so the
// only difference between this builder and the old one is what is ADDED.

// --- Global, team frame (unchanged from DefaultObsPadded) ---------------

// --- Self ---------------------------------------------------------------

// --- Ball, in the car's frame -------------------------------------------

// --- Other cars: absolute block then relative block ----------------------
// Both are emitted per slot so that padding zeroes an entire car rather
// than leaving a relative block that reads as "a car exactly on top of me".

// Shuffled for the same reason DefaultObsPadded shuffles: a fixed slot
// order teaches slot identity rather than what is in the slot.

// A single NaN here reaches the critic, and from there the advantage
// tensor, and from there every weight in the network -- with nothing in the
// telemetry naming the observation as the source. Replacing it with zero
// costs one corrupted sample; letting it through cost p11boost 29.8M steps.


# env/RelativeObs.h

// Implemented in Obs.cpp; see ObsHealth there.

// DefaultObsPadded plus car-frame relative geometry for the ball and every
// other car.
//
// WHY. The p8ref observation was entirely absolute, world-frame. To compute
// `v . dirToBall` -- the quantity 55% of the reward mass pays for -- the
// network had to subtract two absolute 3-vectors, normalise, and dot with a
// third; and to ACT on the answer it had to further rotate the result into its
// own frame using the forward and up vectors. That is four separate 3-vectors
// and a change of basis, learned from scratch, before a single useful decision
// can be made.
//
// Ground driving is nearly planar and tolerates that. Aerial control is far
// more demanding of it, and p8ref's bot resolved the tension by playing an
// almost purely 2D game: `Player/Touch Height` 149, `Touch/Above 450` 0.035,
// high balls reached by driving up the wall rather than by leaving the ground.
// Handing the network the relative geometry directly is the cheapest available
// test of whether representation cost is what shapes that preference.
//
// This is not novel. rlgym-tools ships `relative_physics()`
// (`(target.position - origin.position) @ rot`) for exactly this, and Necto,
// Nexto and current community observations all carry relative features.
// Zealan's guide: "I've found you can get moderately better results if you add
// car-relative positions and velocities".
//
// STATELESS, deliberately. Action stacking (the other half of the modern obs)
// needs per-car history and therefore an episode reset, and `ObsBuilder::Reset`
// is called by EnvSet during training but by NOTHING on the RLBot deployment
// path. A stateful builder would train and deploy differently with no symptom
// -- the bot would load, play, and be quietly worse. That is its own change,
// with its own reset plumbing and its own run.

// Car-frame positions are not field-aligned, so a single scalar is used for
// all three components rather than DefaultObs's per-axis coefficients.
// BACK_WALL_Y puts a cross-field separation near 1 and the arena diagonal
// near 2.6.

// Number of floats one relative block contributes. Exposed for tests.


# env/Rewards.cpp

// Budgets become per-step weights HERE and nowhere else. That single
// conversion site is the point of the budget system: p1air's do-nothing
// attractor was a per-step float whose episode integral nobody computed.
//
// The UNIT is one ball touch. A goal was the unit until p7approach, and it
// could not be audited: goals arrive 0.116 times per episode, and 49% of
// those are `Scenario/Defend` conceding rather than anyone scoring. A touch
// occurs 0.16-2 times per episode and is read directly off
// `Touch/Edge Rate` and `Player/Ball Touch Ratio`, so the ledger can be
// checked against telemetry instead of reconstructed by hand.

// THE UNIT: a maximal goal-directed strike. Signed, so putting the ball
// toward your own net costs. Touch-gated, so it measures only the ball
// motion this car caused -- the continuous VelocityBallToGoal form is
// known-bad here (p1probe-b: 67% of reward mass as passive ball noise).
// Wrapped in ZeroSumReward with opponentScale (default 0.5) so opponent
// goal touches discourage the bot at 50% penalty.

// The scoreboard. Already zero-sum: +1 scored, -1 conceded. Moderate on
// purpose -- see the budget comment; a huge goal reward scales variance,
// not signal, and this is also the only thing that ends an episode.

// Arriving at the ball, on the RISING EDGE so carrying pays once.

// SIGNED. Facing away from the ball returns a negative value and is
// punished. RLGymCPP's FaceBallReward unmodified, matching rlgym, and
// the only term in the stack that charges for anything. See the pairing
// note on SpeedToBallReward.

// Pays for being airborne. Measured ~50x too small to cover what a jump
// costs in traction and contact, and deliberately left that way: see
// the note on RewardBudget::air.
// Per step, on the boost LEVEL: discourages wasting it.

// Per pickup, on the boost INCREMENT: encourages collecting it. Small
// pads carry a guaranteed baseline floor so routing over them stays attractive.

// Forward flip closing acceleration towards the ball. Boost-neutral,
// supporting both ground flips and speed-flips to traverse and conserve boost.

// Pays for touching the ball high AFTER real air time. The min() makes a
// wall shot worth exactly zero, which is the farm this bot already runs.

// Pays for being airborne at all. Measured ~50x too small to cover what
// a jump costs, and left that way on purpose: AirTouch is the term that
// pays for air now, and it pays for PRODUCTIVE air.


# env/Rewards.h

// ============================================================================
// Custom rewards
// ============================================================================
//
// There is exactly one custom reward left. Everything else in the stack comes
// from RLGymCPP's CommonRewards, unmodified, because the point of this stack is
// to reproduce a configuration that is known to work elsewhere before this
// project adds anything of its own. See
// docs/superpowers/specs/2026-08-18-known-good-baseline-design.md.

// One payment per contact SEQUENCE, not per step of contact.
//
// RESTORED after p9rel. This class existed before the phase-C port, the port
// dropped it for fidelity to the reference's flat `EventReward(touch=1)`, and
// p9rel produced exactly the failure it was written to prevent: once the
// relative observation made the bot competent enough to CARRY the ball, a
// per-step touch reward paid for carrying. Steps per contact sequence went
// 1.16 -> 1.98, contact occurred on 13% of all steps, and `RewardShare/Touch`
// reached 0.741 -- three quarters of the budget spent on a dribble.
//
// The guide predicts this transition rather than contradicting it: "The default
// touch part of EventReward is not very good once your bot can touch the ball.
// This is because ball touches can easily be farmed by constantly pushing the
// ball." It is also roadmap decision D4 ("no dribble/possession reward terms,
// ever -- the flick-bot local optimum") arriving through the back door.
//
// A rising edge makes carrying the ball worth exactly one touch, so the term
// pays for ARRIVING at the ball and nothing else.

// A null prev means the episode just reset, so a touch on this step is
// a genuine new contact rather than the continuation of one.

// max(0, v . dirToBall) / CAR_MAX_SPEED -- how fast we are closing on the ball.
//
// This is Zealan's SpeedTowardBallReward (RLGym-PPO-Guide, rewards.md), and it
// is RLGymCPP's VelocityPlayerToBallReward RECTIFIED at zero. The guide is
// explicit about why: "Many good behaviors require moving away from the ball,
// so I highly recommend you don't punish moving away."
//
// The rectification also removes a measurement trap. p1air's RUNLOG row records
// RewardShare 0.482 for the SIGNED form against a near-zero NET, because
// circling generates large +/- values that sum away; the share metric is
// mean|r*w| and cannot tell a farm from a cancellation. Rectified, the mass is
// the signal.
//
// NOTE ON THE PAIRING. Rectifying the velocity term is only safe because
// FaceBall is SIGNED. p7approach rectified both, which left a stack where no
// state the bot could enter was ever penalised -- and the argmax of such a
// stack is "carry speed in a straight line and never turn", since turning is
// the only action that costs speed. That is what p7approach converged toward:
// `Action/Steer Nonzero` 0.160 -> 0.087 while `Jump When Grounded Upright`
// went 0.755 -> 0.878. Do not rectify FaceBall without replacing the term that
// charges for pointing the wrong way.

// Change in the ball's GOAL-DIRECTED speed caused by this touch, signed.
//
// Replaces StrongTouchReward, which paid for hit force in any direction. p11
// measured why that fails: `Touch/Hit Force` fell 878 -> 551 over the run while
// `RewardShare/TouchEdge` doubled, i.e. the bot converged on many brief, weak
// contacts. StrongTouch's floor is 555.6 uu/s, so by the end the AVERAGE touch
// earned exactly zero from it and the only touch term still paying was the flat
// per-contact one. The rising edge stopped the carry farm; it did not stop the
// poke farm, because nothing distinguished a useful touch from any touch.
//
// Direction is what distinguishes them. A poke that does not move the ball
// toward the opponent's net scores ~0; a strike toward it scores highly; and
// putting the ball toward your OWN net is negative, which no previous term in
// this project has ever expressed.
//
// Touch-gated deliberately. `VelocityBallToGoalReward` is the continuous form
// and it is known-bad here: p1probe-b measured it absorbing 67% of reward mass
// as "mostly passive ball motion = zero-sum noise", and p1probe-h found
// removing it changed nothing. Gating on contact attributes only the ball
// motion this car actually caused. Same construction as Lucy-SKG's
// "Touch Ball-to-Goal Acceleration" and rlgym-tools' AdvancedTouchReward.
//
// Normalized by the same 130 kph (3611 uu/s) that saturates StrongTouch, so the
// unit is unchanged: 1.0 is a maximal goal-directed strike.

// CONVEX for p13strike. p12 ran this linear, and a linear term is
// indifferent to CONCENTRATION: the goal-directed dv needed to score is
// fixed by the length of the field, so five 400 uu/s pokes pay exactly
// what one 2000 uu/s strike pays. Every other term broke that tie toward
// the pokes, and `Touch/Hit Force` fell 878 -> 551 -> 422 across three
// runs. |x|^p with p > 1 breaks it the other way: at p = 2 an 80 kph
// strike is worth 16x a 20 kph poke rather than 4x.
//
// A power law rather than StrongTouch's hard floor, deliberately. A floor
// has NO gradient below it; at the 80 kph a "strong touch" ought to mean
// it would read identically zero today, since the mean touch is 15.2 kph.
// |x|^p keeps a gradient everywhere (d/dx = p|x|^(p-1)) while making the
// effective threshold rise on its own as the bot gets stronger.

// One direction for both samples: we want the change in speed toward
// the net, not a change that includes the ball having moved.

// Saturation stays at 130 kph so 1.0 is still one maximal goal-directed
// strike and every earlier budget reads in the same unit. Sign is kept
// outside the power so putting the ball toward your OWN net still
// costs, and costs convexly too.

// Pays for touching the ball high AFTER genuinely being in the air.
//
// `min(airTimeFrac, heightFrac)` is the guide's form, and the min is the whole
// design. Paying for height alone produces what the guide names the "lame plat
// wall-shot" -- and this bot already does exactly that, reaching high balls by
// driving up the wall. A car on a wall is `isOnGround`, so its `airTime` is 0
// and the min makes that worth nothing. To score here it has to leave a surface
// and stay off it.
//
// Only reachable behaviour is being paid for: `Touch/Above 450` is already
// 0.081, so this is not asking the policy to discover something new. It is
// paying for something it does occasionally and then argues itself out of --
// air play emerged and decayed twice (p10touch, and p11 at 42-56M).

// CONVEX IN HEIGHT for p14. p13 measured the failure: at ball height ~350
// with 0.9 s aloft, min(0.52, 0.171) = 0.171, so a plain jump-touch
// collected the term outright. `RewardShare/AirTouch` duly rose to 0.047,
// ABOVE its 0.030 target, while `Touch/Above 450` FELL 0.037 -> 0.015. The
// term was paying for exactly the behaviour that replaced aerials.
//
// The fix is not a height floor. A jump taken to reach a high ball IS an
// aerial and should be paid -- it is the same skill, smaller. What was
// missing is that height must pay DISPROPORTIONATELY, so heightFrac is
// raised to a power instead of gated. At exponent 2 a touch at z 800 is
// worth 7.1x one at z 300, against 2.7x linear, and the gradient at the
// current operating point is still 38% of the target one -- the same
// derivation that set TouchGoalAccel's exponent.

// A rough ceiling on a reasonable aerial, from the guide. Longer air times
// are not worth more: this pays for reaching the ball, not for floating.

// RISING EDGE, matching TouchEdgeReward, and NOT optional at p13's
// budget. The p12 form paid on every contact STEP, so an air carry at
// ceiling height would have earned ~170 touch-units per second -- the
// p9rel dribble farm, relocated to the air. It stayed harmless in p12
// only because the budget was too small for anything to happen.

// The min() with air time is untouched: it is what makes a wall shot
// worth exactly zero (a car on a wall is isOnGround, so airTime is 0),
// and that is still the farm this bot is closest to.

// Rewards the forward closing velocity gained from dodging/flipping towards the
// ball. Boost-neutral: does not check or punish boost, allowing speed-flips to
// emerge naturally while providing the gradient needed for ground flip discovery.

// Trigger only on the rising edge of a dodge (flip initiation)

// Inactive if already at supersonic speed before the flip

// Measure velocity gain in the direction of the ball

// Normalized against the 500 uu/s impulse of a standard forward dodge

// Rewards collecting boost, with an elevated base floor for small pads (12 boost)
// so clipping pads into general ground routes remains attractive even when partly full.

// Big boost pads replenish up to 100 (delta > 25); small pads grant 12

// Big boost pickup: reward scaled by boost replenished (up to 1.0 for 0 -> 100)

// Small pad pickup: guaranteed base floor + diminishing potential gain

// ============================================================================
// Diagnostic constants
// ============================================================================
// Not reward weights. Thresholds for metrics in Train.cpp, kept here so the
// derivations stay next to the physics they come from.

// Throttle-only top speed: DRIVE_SPEED_TORQUE_FACTOR_CURVE reaches zero here,
// so any car can hold this indefinitely with no boost and no skill.

// Speed that can be lost in one decision step without a collision. RL brakes at
// roughly 3500 uu/s^2, which over a 1/15 s step is 233 uu/s, so 400 is clear of
// any input-driven deceleration and only a collision reaches it. That 3500 is
// EMPIRICAL, not a RocketSim constant.

// Returns heap-allocated rewards; GigaLearn's EnvSet takes ownership, so the
// caller does not free them.

// One reward term with a stable metric name, since Reward::GetName() is
// useless for that on GCC (near-mangled typeid strings).

// metric label, e.g. "SpeedToBall"

// The order of specs matches the order of envSet->rewards[arena] and
// envSet->state.lastRewards[arena]; the reward-share metrics index by it.

// Materializes the specs, same order.


# env/StateSetters.cpp

// Sign convention: +1 means "this team attacks +Y" (blue), -1 means "attacks
// -Y" (orange). Multiply any Y coordinate written from blue's perspective by
// this to mirror it for orange.

// Rejection-sample inside the unit sphere so directions are uniform.
// Normalising a uniform cube biases towards the corners.

// Point a car at a world-space target, with optional roll/pitch noise.

// Place a car flat on the ground at (x, y) facing a target.

// ----------------------------------------------------------------------------

// Ball falling rather than rising: a rising ball climbs out of the strike
// band while the car closes, which turns half the spawns into a scenario
// that cannot be solved by the skill being taught.

// Aim at the ball's ground position, not the ball: pointing the car up
// at an airborne ball would spawn it nose-high off the floor.

// Spawn on the ground a workable distance from the ball so the policy
// has to actually drive-and-jump rather than start already underneath.

// The first car gets the ball; everyone else is scattered so they do not
// spawn inside it.

// Just under the ball, moving with it -- the state you are in
// immediately after a successful pop.

// No second jump: force air-roll control

// Below the ball, rising towards it, flip already spent. Reaching
// the ball's underside with the wheels is what grants the reset.

// Flip is gone until a reset restores it

// Drive towards the opponent's goal, ball balanced on the roof.

// Ball parked out of the way so the episode is about the cars.

// Lay the cars out along a shared axis, blue on one side, orange the other,
// all pointed at each other and moving fast.

// Offset each extra car perpendicular to the axis so they do not stack.

// Pick a team to be under pressure; mirror everything about that choice.

// Defended goal is at defSign * BACK_WALL_Y

// Aim the ball at the defended goal, roughly.

// Goal side of the ball.

// Attacker following the ball in.

// Airborne, tumbling, pointed nowhere useful, and far from the ball.

// Keep the ball on or near the deck. A high ball here would just recreate
// the problem this setter exists to solve.

// Place the car on a random bearing around the ball, facing it.

// Lead a moving ball slightly rather than aiming where it is now,
// so the spawn is a playable position and not an instant miss.

// Everyone else starts at a normal distance, so this does not turn
// into a scrum around the ball.

// Bias each team towards its own half so the layout resembles real play
// rather than a scramble.

// Written back on EVERY reset, not just the infinite ones. The mutator
// config belongs to the arena and outlives the episode, so skipping the
// restore would leave an arena permanently infinite after its first
// infinite episode -- and nothing downstream would look wrong.


# env/StateSetters.h

// Wraps any spawner and, on a fraction of episodes, gives both cars a full tank
// that never drains (RocketSim's `boostUsedPerSecond = 0`).
//
// WHY. p10touch's bot found air dribbling off the wall and could only sustain
// it when it happened to have boost, running at `Player/Boost` **7.3 out of
// 100** for the whole run. A policy cannot learn the value of a resource it
// never has: with a near-empty tank, every aerial it attempts fails for a
// reason that has nothing to do with the aerial. This gives it a supply of
// episodes where the boost constraint is simply absent, so the behaviour can be
// discovered first and the economy learned second.
//
// The observation carries `boost / 100`, so the policy can tell an infinite
// episode from a normal one within a step or two of boosting and does not have
// to average the two regimes into one behaviour.
//
// MUST RESTORE. The mutator config is arena state, not episode state, so an
// arena that goes infinite stays infinite for every subsequent episode unless
// the normal rate is written back. That failure would be invisible -- the run
// would simply look like a bot that solved its boost problem. Asserted in
// bot/tests/test_statesetters.cpp.

// Takes ownership of the inner spawner.

// Whether the episode just spawned has infinite boost, for metrics.

// Each setter spawns the arena into the *start* of one situation, so the
// policy gets a dense supply of it instead of waiting for it to occur
// naturally (a flip reset happens roughly never under random play).
//
// Design rules followed throughout:
//   * Always call arena->ResetToRandomKickoff() first, then overwrite what we
//     care about -- it resets boost pad timers, the ball, and every car.
//   * Randomise generously; a fixed scenario teaches memorisation, not skill.
//   * Set state for EVERY car, or leftover cars pollute the episode.
//   * Respect team symmetry: mirror positions by team.

// The jump-flip strike: ball at jump height, car already rolling at it and
// already at pace, so the only open decisions are timing and steering trim.

// Jump-reachable band, above where the bot can reach on wheels.

// Close enough that contact is likely, far enough that the jump has to be
// timed rather than mashed on spawn.

// Full boost on a fraction of spawns, to connect "full tank" with "can hit
// harder".

// Ball off the ground, cars on the ground with boost. Teaches driving-to-takeoff
// and airborne ball contact. Height and distance are one setting, not two,
// because the curriculum instantiates this twice at different heights and
// spawn distance has to track ball height (too far and the ball is back on
// the ground before the car arrives).

// Car and ball both airborne and travelling together, car just under the ball.
// Teaches carrying the ball through the air once contact is established.

// Car airborne under a high ball with its flip already used. Reaching the ball
// underside restores the flip -- that is the reset. Teaches the approach, not
// the follow-up.

// Ball resting on the car roof, both moving forward together on the ground.

// Two cars converging at speed with boost. Teaches bumps and demos, and
// teaches the receiving car to avoid them.

// Ball moving towards one team's goal with a defender positioned behind it.
// Teaches saves and shadow defence.

// Cars tumbling in the air away from the ball. Teaches wave dashes, recoveries
// and landing on wheels -- unglamorous but a large share of real playtime.

// Ball and cars placed in plausible mid-play positions on the ground. This is
// the "everything else" setter and should carry most of the curriculum weight.

// One car spawned right next to the ball, on the ground, already pointed at
// it. Unlike other setters (which assume the bot can already drive), contact
// here is available within a second or two, so the touch reward fires often
// enough to reinforce "hit the ball" before the bot has learned to navigate
// to it. Not a permanent fixture -- weight should come down in favour of
// NeutralPlayState once touch ratio is healthy.

// Near enough that contact is close to unavoidable, far enough that the
// car still has to steer and commit.

// Fraction of spawns where the ball is already rolling rather than still.


# env/Terminal.h

// Custom terminal conditions live here. Currently empty: the env uses
// RLGymCPP's NoTouchCondition and GoalScoreCondition directly.


# eval/Checkpoints.cpp

// What Policy::Load -> GGL::InferUnit actually opens. RUNNING_STATS.json is
// included because loading without it gives an unnormalized observation: the
// bot runs, and is quietly wrong, which is the failure mode this project
// keeps hitting.

// Skips policy_versions (the self-play snapshot pool) for free.

// Digits but wider than 64 bits; not a real checkpoint.


# eval/Checkpoints.h

// Newest loadable checkpoint inside a run folder (checkpoints/main-<label>).
// "Newest" is by step count parsed as a number, not lexicographically, and
// folders missing a required file are skipped (the trainer may still be
// mid-write to the newest one).


# eval/Eval.cpp

// actionDelay: hold the old action for the first actionDelay ticks
// of this window, then apply the fresh one -- the same cadence the
// policy trained with and HivemindBot::update replays.


# eval/Eval.h

// Headless checkpoint-vs-checkpoint matches in RocketSim. This is the
// frozen-reference-pool tool: pit any two checkpoints (current vs a gate
// checkpoint, run A vs run B) without a learner or a game client.

// per game, sim time

// MPPI lookahead ticks for Blue (0 = off)

// MPPI lookahead ticks for Orange (0 = off)

// Number of MPPI rollout candidates


# eval/Spectate.cpp

// Resolve which checkpoint to play. Returns an empty path when following a run
// that has not saved one yet, which is a wait-and-retry, not an error.

// Pin inference to one thread so a spectator doesn't steal CPU from a
// concurrent training run. Set through the environment rather than
// at::set_num_threads(), since libtorch's headers are private to
// GigaLearnCPP; the pools read these on first inference. The 0 flag leaves
// an explicit setting from the caller alone.

// Deployment-side values, so what is watched matches what is trained and
// what is deployed. A divergence here would not crash -- it would just make
// the bot look worse than it is, which is the whole class of bug the
// `verify` subcommand exists to catch.

// Learner's constructor is what normally starts the interpreter; there is
// no Learner here, and RenderSender needs it to import the receiver module.

// Between episodes, not mid-episode: swapping the policy under a car
// mid-play would show a discontinuity that is an artifact of watching,
// not of the bot.

// One GameState reused for the whole episode. Constructing a fresh one
// per step (as RunEval does, where it is harmless) makes deltaTime the
// arena's whole lifetime rather than one step: it would break both the
// renderer's wall-clock pacing and ballTouchedStep, and so the no-touch
// timeout too.

// Replay the training cadence exactly: hold the previous action for
// actionDelay ticks, then apply the fresh one for the rest.

// Also paces to wall-clock.


# eval/Spectate.h

// Where episodes are spawned from.

// Whatever TrainConfig::spawn selects, via the same BuildSpawner() the
// learner uses -- so what you watch is what the policy is practising.
// Watching a distribution the bot never trains on is how you conclude it
// handles situations it has never seen.

// Kickoff to goal, like a real match. Easier to judge as play, but not
// representative -- kickoffs are only ~8% of training resets.

// Watch a checkpoint play, live, in RocketSimVis.
//
// This is deliberately NOT a learner. `train --render` builds a full Learner
// and collects experience; running it alongside a real run wastes CPU and
// competes for the run's checkpoint folder. This loads a checkpoint, plays it
// against itself in one arena at wall-clock speed, and streams the gamestate.
// It never writes anything.

// Exactly one of these. `model` is a specific checkpoint folder; `followRun`
// is a run folder (checkpoints/main-<label>) whose newest checkpoint is
// picked up between episodes, so a live run can be watched improving.

// Training owns the GPU during a run, and one car at 120 ticks/sec does not
// need it.

// Training samples from the action distribution; deterministic shows the
// policy's intent without exploration noise, and is what deployment uses.

// Multiplier on real time. 1.0 is Rocket League speed.

// 0 runs until interrupted.

// MPPI Lookahead search in ticks (0 = disabled)


# main.cpp

// RLBot launches us with these set; the defaults let you run by hand.

// Load models before connecting. RLBot has a connection timeout, and
// loading a policy onto the GPU is slow enough to trip it if done
// lazily on the first packet.

// batchHivemind = true asks RLBot to deliver all our cars in one update()
// call, which is what lets us run a single batched inference pass.

// 0 disables the controller and pins entropyScale to --entropy,
// which is what a calibration probe wants: the probe must not
// move the policy it is measuring.

// Skill tracking is what makes the result readable, so turn it on
// with self-play unless it was already requested.

// Rendering runs the sim at wall-clock speed so you can watch it in
// RocketSimVis. Useful for sanity-checking state setters and
// rewards; useless for actually training.

// The learner silently loads the newest checkpoint in the run folder, so
// reusing a label continues that run instead of starting one. Config.h has
// warned about this since the label was introduced and it still cost a run:
// every threshold in runs/RUNLOG.md is stated as "X at 100M against the
// previous run's X at 100M", and a resumed run has no such baseline.

// Renamed, never deleted. A crashed run's checkpoints are often the
// most interesting thing on disk -- p11boost's last save turned out
// to predate the NaN and verified clean.

// The metrics receiver RELOADS an existing CSV and appends to it,
// deliberately, so that a resumed run keeps one continuous file and
// late-arriving columns (Rating/*) are not lost. That is right for a
// resume and wrong here: leaving it would concatenate two runs into
// one file with no marker, and a trend read off it shows a policy
// that mysteriously resets partway through. It did exactly that once
// already.

// Unbuffered output so logs interleave correctly when RLBot captures them.

// Take a label, not a path: the point of this command is to
// watch a run, and the caller should not have to know how
// checkpoint folders are named.


# policy/Policy.cpp

// GigaLearn writes each sub-model as an uppercased <NAME>.lt file. Checking
// here turns an opaque libtorch failure into an actionable message.

// Shared head feeds the policy, no logits


# policy/Policy.h

// Shape of the policy network. Single source of truth for training and
// deployment: both sides default-construct this struct. Changing it
// invalidates every existing checkpoint.

// obsBuilder and actionParser are borrowed; the caller keeps them alive.

// Throws std::runtime_error on a missing/invalid checkpoint folder.


# policy/RolloutPlanner.cpp

// 1. Base action from policy

// 2. Action variations (perturbations)

// Boost on/off

// Jump on/off

// Handbrake toggles

// Steer adjustments

// Pitch adjustments

// Roll adjustments for recovery

// Throttle variations

// Tactical primitives
// Max forward strike

// Hard brake / reverse

// Air recovery (wheels down)

// 1. Goal events

// 2. Goal-directed ball velocity
// Blue attacks +Y (target Y = +5120), Orange attacks -Y (target Y = -5120)

// uu/s

// 3. Own-goal danger veto
// If ball is heading rapidly toward own net

// 4. Ball contact

// 5. Ball proximity

// 6. Car recovery & landing

// In air: reward upright orientation (rotMat.up.z close to 1)

// Set up our car and opponent in the simulation arena

// Find opponent player if present

// Neutral rest position outside play area if no opponent

// Restore state

// Neutral coasting for opponent

// Selection: Argmax if low temperature, else MPPI softmax blending

// Softmax MPPI weighting

// Discretize button inputs


# policy/RolloutPlanner.h

// 0 means disabled (vanilla policy execution)

// ~1.5 decision steps at 120Hz physics (100ms)

// Number of perturbed action trajectories

// Softmax temperature for MPPI weighting (<= 0.05 selects argmax)

// Physics heuristic weights

// Scored in opponent net

// Conceded into own net

// Goal-directed ball velocity

// Closing distance to ball

// Making contact with ball

// Upright landing / 4 wheels on ground

// Penalty if ball deflected fast toward own net

// Evaluates K candidate perturbations around baseAction in RocketSim and
// returns the optimal action according to MPPI physics heuristics.

// Evaluates a simulated arena state from the perspective of team.

// Generates candidate action variations around baseAction.


# rlbot/HivemindBot.cpp

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared context
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bot
// ---------------------------------------------------------------------------

// The interface marks update() noexcept, so nothing may escape. A throwing
// inference call must degrade to "keep holding the last controls" rather
// than terminate the process mid-match.

// --- Tick accounting -------------------------------------------------

// Clamp: a paused game or a goal replay can produce a huge or
// negative delta, which would otherwise desync the action cadence.

// --- Decide which cars need a fresh action ---------------------------

// A car we were assigned may not be in the packet yet (or at all,
// briefly, during a match restart).

// --- One batched forward pass for every car that needs it ------------

// --- Apply the trained action cadence --------------------------------

// Latency: hold the previous action until actionDelay ticks have
// passed, matching how the policy experienced the world in
// training.

// Repeat: infer again only every tickSkip ticks.

// ControllerState is a flatbuffers struct: build it in one shot.
// Field order is throttle, steer, pitch, yaw, roll, then the
// buttons -- matching RLGC::Action's own ordering.

/*use_item=*/false));

// Log once; a per-tick error would flood output


# rlbot/HivemindBot.h

// One process, one team, every car on it. With `hivemind = true` in bot.toml,
// RLBot v5 assigns all same-team cars sharing our agent_id to a single Bot
// instance (`indices` holds all of them), and every car is inferred in one
// batched forward pass.

// Settings loaded from the environment before the bot connects, since RLBot
// gives a bot no config channel of its own.

// HIVE_MODEL

// Whether the action parser masks by situation. MUST match training: an
// unmasked policy deployed against a masked parser (or the reverse) picks
// from a different action set than it learned on, and nothing about that
// looks like a failure -- the bot loads, plays, and is quietly worse.

// HIVE_MASK_ACTIONS

// Which observation layout to build. MUST match training. A width mismatch
// throws at load; a same-width layout mismatch would not, which is why
// `verify` checks this explicitly rather than relying on the load.

// HIVE_OBS_DEFAULT

// Deterministic play picks the highest-probability action every step. It is
// noticeably stronger than sampling and is what you want in a real match;
// sampling is only useful if you want variety.

// MPPI Lookahead search (0 = disabled, vanilla inference)

// HIVE_LOOKAHEAD_TICKS

// HIVE_ROLLOUT_CANDIDATES

// HIVE_MPPI_TEMPERATURE

// Read settings from environment variables, applying the defaults above.
// Throws std::runtime_error if HIVE_MODEL is unset.

// Process-wide state shared by every HivemindBot instance. RLBot constructs
// bots through a factory that takes no user parameters, so this is how settings
// and the loaded model reach them. The model is loaded once and shared, which
// also means a blue and an orange hivemind in the same process share weights.

// Initialise RocketSim, build the obs/action pipeline, and load the model.
// Call once before connecting. Throws on failure.

// Per-car action repeat state. The policy was trained acting once every
// `tickSkip` ticks with `actionDelay` ticks of latency; replaying that
// cadence at deployment is not optional. Acting every tick instead makes
// the bot behave measurably differently from the one that was trained.

// Freshly inferred, not yet applied

// Currently being held


# rlbot/PacketConvert.cpp

// Match each RLGymCPP pad to the nearest RLBot pad by location. Nearest-match
// rather than assuming a shared sort order, so a change to either side's
// ordering shows up as a loud mismatch instead of scrambled observations.

// Ignore z; pads sit at 70 or 73

// Pads are hundreds of units apart, so a correct match is within a few
// units. 100 uu of slack catches float noise without accepting a
// genuinely different pad.

// --- Ball ---------------------------------------------------------------
// Standard soccar has exactly one ball. If there is none (between goals,
// or in an exotic mode) leave the ball at the origin rather than reading
// out of bounds; the policy will produce something harmless for one tick.

// --- Boost pads ----------------------------------------------------------

// The inverted views are what orange-team observations read from.

// --- Players -------------------------------------------------------------

// demolished_timeout is -1 when the car is alive.

// Flip/jump availability. The observation reads HasFlipOrJump(),
// which RocketSim derives as:
//     isOnGround || (!hasFlipped && !hasDoubleJumped &&
//                    airTimeSinceJump < DOUBLEJUMP_MAX_DELAY)
// RLBot tells us the answer directly, so rather than reconstruct
// RocketSim's internal timers we set these fields to whichever
// values make the derivation produce RLBot's answer.

// The isOnGround term already forces true.

// A dodge or double jump is still available: force true.

// No dodge left: force false.

// Previous action, so the observation's prev-action block matches
// what the policy saw in training.

// Turn "latest touch" into "touched during this step" by watching
// for the touch timestamp changing.

// --- Match state ---------------------------------------------------------


# rlbot/PacketConvert.h

// Translates an RLBot v5 GamePacket into the RLGymCPP GameState the policy was
// trained on. Errors here are silent -- the bot just plays worse than in
// training -- so two traps get explicit handling instead of relying on
// upstream agreement: boost pad order (RLGymCPP's hardcoded table vs RLBot's
// y-then-x order, reconciled via an index map built from FieldInfo) and flip
// availability (RocketSim derives it from internal fields, which we set to
// match RLBot's dodge_timeout ground truth).

// Build the boost pad index map from FieldInfo. Call once, from
// Bot::initialize(). Safe to call with nullptr (falls back to identity
// mapping and logs a warning).

// Convert a packet into a GameState.

// rlgymIndex[i] = index into RLBot's boost_pads array for RLGymCPP pad i.

// player_id -> game_seconds of that player's most recent registered touch,
// used to turn RLBot's "latest touch" into a per-step "touched this step".


# train/Metrics.cpp

*No substantial comments in this file.*


# train/Metrics.h

// Convert per-term sum(|weighted reward|) into fractions of the total.
// Returns an empty vector when the total is zero.


# train/Train.cpp

// private GGL header; see bot/CMakeLists.txt

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
// These track HOW the bot earns reward, not just how much -- reward goes up
// in every run, but a healthy curve can still hide a collapsing touch height.
// Sampled on a fraction of steps, since iterating every player of every game
// every step is a real cost at 128 games.

// Step budget for bounded runs. Set from RunTraining(); 0 disables.

// Set by HandleSigint, checked in StepCallback. wandb's "Stop run" button
// sends SIGINT to this process; a plain Ctrl-C does too. Only async-signal-
// safe work happens in the handler itself -- the actual save/exit runs from
// StepCallback on the main thread.

// (name, weight) per reward term, in spec order -- the same order EnvSet
// stores per-term rewards in. Set once in RunTraining() before the learner
// starts; no rewards are allocated for this.

// Needed to rebuild observations inside the metrics callback; EnvSetConfig does
// not carry it. Set once in RunTraining() before the learner starts.

// Mirrors RewardBudget::touchAccelExponent so the metric reports the same
// curve the reward pays on. A metric that silently disagrees with its
// reward is worse than no metric.

// Decision steps since each arena last reset, for the Episode/* buckets.

// Runs on a fraction of sampled iterations: an extra critic forward pass is not
// free, and this exists to answer a question, not to run forever. Same thread
// as collection (Learner.cpp calls the step callback after envSet->Sync()), so
// touching the models here does not race the workers.

// Flat [N, obsSize] buffer plus, per row, what it is so the values can be
// bucketed after one batched forward pass.

// row is the state the policy chose FROM

// plain wheels-down, for the V(ground) vs V(air) split

// grounded AND upright AND jump legal: the jump choice

// and the policy pressed jump

// ragged obs would corrupt the batch; skip rather than

// guess

// Rows were pushed in (before, after) pairs, so i is the chosen-from
// state and i+1 is where it led.

// The plain split answers (a) directly: is being airborne worth more?

// Save first, then _exit rather than return: unwinding out of a callback
// mid-collection would race the worker threads, and there is nothing left to
// clean up once the checkpoint is on disk.

// GigaLearn's training loop runs until the user presses Q; there is no
// timestep limit and no documented way to break out of it. The step
// callback is the only hook that runs inside the loop with access to the
// learner, so the budget and the SIGINT check are both enforced here.

// --- Episode age --------------------------------------------------------
// Tracked OUTSIDE the sampling gate below, because it has to count every
// step to stay accurate. Buckets behaviour by time since spawn, so an
// approach failure right after spawn is distinguishable from a recovery
// failure later in the episode.

// Terminals are flagged for the step that ended the episode; the
// reset happens after. Count first, then zero, so the final step of
// an episode is still attributed to that episode.

// Re-derives REFERENCE_EPISODE_STEPS, which ships as a working
// figure of 150. Recorded at the terminal step so it is a real
// episode length rather than a running age, and outside the
// sampling gate so no episode is missed.

// Sample roughly a quarter of steps. Averages over an iteration are just
// as accurate and cost a quarter as much.

// Buckets are decision steps at 15 Hz: the first second off the spawn,
// the next three, then everything after.

// --- Play phase distribution -------------------------------------
// Shows what the policy actually spends its time doing. If you
// bump the aerial weight in the curriculum and the Aerial share
// does not move, the setter is not doing what you think it is.

// --- Core behaviour ----------------------------------------------

// --- Surface contact ---------------------------------------------
// The WrongSurface term as a rate. If this does not fall over a
// run, the bot is not learning to land, whatever the reward says.

// A front flip drives the nose into the floor for a few ticks, and
// Player samples the final tick of eight, so the scrape is caught
// ~25% of the time. The design prices that at ~11x cheaper than the
// flip's own speed gain; this is the check on that arithmetic.

// --- Landings
// -------------------------------------------------------

// --- Speed
// ----------------------------------------------------------

// No reward pays for generic speed any more, so this is pure
// diagnostics: it says whether the bot is boosting and flipping or
// coasting on the throttle-only floor.

// The deceleration TAIL, as threshold shares. p6budget shipped
// a single metric called "Speed/Max Step Decel" that was built
// with report.AddAvg, i.e. a MEAN -- it read 22.4 uu/s and
// could neither confirm nor refute the 400 uu/s collision
// threshold it existed to check. Shares at three thresholds
// make the tail readable without a histogram. Hits are excluded
// because a hard shot SHOULD cost speed.

// --- Facing, and the gap that killed p6budget --------------------
// p6budget's headline failure was that its NOSE alignment doubled
// (0.338 -> 0.741) while its VELOCITY alignment never moved off
// 0.300 across 100M steps. That was a DERIVED number -- Speed
// Towards Ball divided by Player/Speed -- and this project has
// retracted two analyses that rested on derivations. Both halves
// are first-class reads now, on the same rectification, so the gap
// can be measured directly.
//
// The ground/air split is the mechanism check: rotation is free in
// the air and costs steering on the ground, so if the nose is
// aimed only while airborne, the facing term is being bought
// without being paid for.

// What FaceBallRectifiedReward actually pays.

// The quantity that has to move for any of this to be working.

// --- Touch edge -------------------------------------------------
// The rate the Touch term actually pays at, as opposed to
// Player/Ball Touch Ratio which counts every step of contact. The
// gap between them is how much carrying is happening.

// Same three quantities, split by how old the episode is. If the
// Early numbers are strong and Mid/Late collapse, the bot can
// approach a ball exactly once per spawn.

// Touch height is the clearest single indicator of whether the bot
// is developing an air game. Watch it more than the reward.

// --- Touch distribution
// ------------------------------------------- Distribution rather
// than the mean touch height (~147 measured): the mean hides the
// thing that actually matters, which is whether ANY touches are
// happening in the jump-only band at all. None of these depend on
// state.prev -- they read the current touch and grounded state, not
// a velocity delta.

// Did it get there with a jump, and did it flip into the
// ball? This is the target skill, stated as a metric.

// What a realized touch is actually WORTH under
// StrongTouchReward, which is the only thing that can turn
// its provisional 1.0 budget into a measured one (roadmap
// D6). Hit force is |delta ball velocity| at contact; the
// reward is 0 below 20 kph and 1.0 at 130 kph. KPHToVel(x)
// is x * 250/9, so those are 555.6 and 3611.1 uu/s -- NOT
// the 183/1192 this comment used to claim. For scale,
// CAR_MAX_SPEED 2300 uu/s is 82.8 kph and the ball caps at
// 6000 uu/s = 216 kph, so 20 kph is a very weak touch.
//
// The gap between `Hit Force` and `Strong Value` is the
// point: a dribble carry produces a large touch RATE at
// near-zero force, so if Strong Value stays flat while
// `Player/Ball Touch Ratio` climbs, the bot is farming
// contact again and the rising-edge term is not enough.

// What a realized touch earns from the term actually in
// the stack. `Strong Value` above is the RETIRED
// StrongTouch curve, kept only so the series stays
// comparable across runs; this is the one that
// reconciles against `RewardShare/TouchGoalAccel`.
// Being convex it falls away far faster than hit force
// does, and the gap between Raw and Value is exactly
// what p13strike buys.

// --- What the policy actually DID --------------------------------
// Everything above is a state statistic: it says where the car
// ended up, not what the policy chose. "In Air Ratio 0.91" is
// consistent with a policy that jumps constantly AND with one that
// never jumps but keeps getting launched, which need opposite
// fixes.
//
// prevAction is the action applied during this step, so it must be
// conditioned on the PREVIOUS state -- that is the state the policy
// saw when it chose. Without prev there is no decision to
// attribute.

// Mirrors DefaultAction::GetActionMask: jump actions are offered
// while a flip/jump remains, and also while turtled (upside down),
// which is how a stuck car rights itself. If jump was not on the
// menu, the step says nothing about whether the policy wants it.

// A car resting upside down on the floor is "grounded" and jump is
// the correct way out of it, so an upright split is needed before
// a high grounded jump rate can be read as a farm rather than as
// recovery. Wheels-down is rotMat.up.z near +1.
//
// BUT: a car driving on a WALL is also grounded with up.z near 0,
// so it lands in the non-upright bucket too. That bucket was named
// "Inverted" and read as upside-down recovery for three runs, while
// p10touch's 0.0615 there -- 15x the eps-floor and rising -- was
// overwhelmingly wall jumping, the one place this bot leaves a
// surface on purpose. The name is now "Tilted" and the denominator
// is published alongside it, because "jumping is extinct" was drawn
// from the Upright bucket alone and was wrong.

// --- Is it just standing there? ----------------------------------
// The old stack had a flat `Grounded` bonus that made standing
// still a risk-free annuity; that term is gone (deleted with
// GroundedBonusReward), but the ratio is still worth watching --
// SpeedToBallReward pays zero for a motionless car, so a nonzero
// stationary share now means the *rest* of the stack isn't moving
// the policy off it either.

// --- Is it actually driving? -------------------------------------
// Added while diagnosing a policy that boosts in straight lines and
// never turns. Everything else here measures resulting STATE; these
// measure the grounded control inputs directly, because "it does
// not steer" and "it steers but cannot hold a line" look identical
// from speed and position alone.
//
// Split by whether boost is even available: DefaultAction masks out
// every boost action at zero boost, so a raw boost rate conflates
// "chose not to boost" with "could not".

// --- What KIND of flip? ------------------------------------------
// DefaultAction's jump entries always have yaw == 0 (jump+yaw
// combinations are skipped when the table is built), so a diagonal
// flip is pitch and roll together; a straight flip is one axis.

// p6budget left 99.1% of jump-presses in neither bucket, i.e.
// single-axis, with no way to tell a front flip from a side
// flip. Watching the bot said side flips; the metrics could not
// confirm it. Roll-only IS a side flip, so split them.

// airTimeSinceJump is the gap between leaving the ground and
// now, so on the step a second jump is pressed in the air it IS
// the flip delay. A deliberate stall shows up as a delay well
// past the ~0.1s a reflexive double-jump would give.

// Where does the air time come from? Of every ground->air
// transition, how many did the policy cause by pressing jump, as
// opposed to driving off a ramp, being bumped, or a curriculum
// spawn. If this is low the flip-spam reading is simply wrong.

// How long a single airborne stint lasts, in seconds. Pairs with
// Leave Ground Rate: together they say whether 91% air time is many
// short hops or a few very long tumbles.

// --- What does a flip actually buy? ------------------------------
// Landing vs sustained split: tests whether a flip is a speed pump
// or just a heading randomizer. Not a clean counterfactual --
// "sustained" cars have had time to accelerate -- but it bounds the
// effect.

// Report each phase as a fraction of sampled player-steps.

// --- Reward shares ------------------------------------------------------
// lastRewards holds each term's raw (unweighted, pre-zero-sum) reward for
// one sampled player per arena; |r * w| across terms approximates where
// the realized reward mass is going. This is the farming detector.

// --- Observation health --------------------------------------------------
// Zero in every healthy run. Non-zero means a NaN or inf reached the
// network's input, which is the one failure that corrupts training without
// appearing anywhere else in this report.

// --- Infinite-boost episodes --------------------------------------------
// Published so `Player/Boost` can never be read without knowing what share
// of episodes had a tank that could not drain.

// --- Scenario outcomes --------------------------------------------------
// Terminal arenas have not been reset yet at callback time, so the
// curriculum's last-picked name still labels the episode that just ended.

// Seed every configured scenario with zero so names not picked
// this step still contribute a sample; otherwise rare scenarios'
// Share averages are biased upward.

// A true share: count per name across all arenas, so a scenario that
// never runs is distinguishable from one that always does.

// --- What does the CRITIC think? ----------------------------------------
// V(grounded) vs V(airborne) says whether the critic favours air time on
// its own, independent of what the reward does. The TD split goes further:
// out of a grounded state, does bootstrapping through the jump action look
// better than not jumping? Note the TD figure omits the immediate reward
// (it's gamma*V(s') - V(s), not the full residual), so read it as "where
// does jumping take me", not as an advantage.

// ---------------------------------------------------------------------------

// RocketSim needs the collision meshes to simulate the arena geometry.
// Without them cars fall through the world, which presents as a bot that
// learns nothing rather than as an obvious error.

// Probe the observation width rather than deriving it. See env/Obs.h.

// Above 0 this turns entropyScale into a CONTROLLED variable rather
// than a constant; the learner then reports `Entropy Scale` and
// `Entropy Target` so the loop is auditable. See the note on
// TrainConfig::entropyTarget for why a fixed coefficient cannot hold
// an entropy floor.

// The policy and shared-head shapes here MUST match ModelShape in
// Config.h, because that is what the RLBot client rebuilds at load time.
// Mismatch means the deployed bot silently loads garbage weights.

// The critic is training-only, so it never has to match the client. Give it
// the same shape as the policy; there is rarely a reason to differ.

// --- Self-play ----------------------------------------------------------
// The learner forces savePolicyVersions on if either of these is enabled,
// since both need the version pool. Setting it explicitly documents the
// dependency rather than relying on that.

// Capture cfg by value: the learner calls this for every game at startup,
// and outliving the caller's stack frame is not worth risking.

// Installed after the Learner (and its embedded Python interpreter) is
// constructed, since Python init can otherwise install its own handler.


# train/Train.h

// Run a training session to completion (or until you stop it).
// Blocks until the learner exits.
