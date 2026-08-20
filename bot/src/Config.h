#pragma once

#include "env/Obs.h"
#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Hive {

// Everything tunable lives here. Defaults are sized for the training machine
// (Ryzen 3600, 6 cores; RTX 2060, 6 GB VRAM).

// Relative weights for how often each scenario spawns. They control what the
// policy practises, not which model runs -- there is one model. Re-derive
// them from telemetry at each phase gate.
struct CurriculumWeights {
	float neutralPlay = 35.f; // Ordinary play. Keep this dominant.

	// Car spawned next to the ball, so contact is nearly free -- mostly
	// training on a situation the bot already gets for free at this point.
	float ballContact = 10.f;

	float defend = 15.f;

	// Cars end up airborne whether or not we spawn them there, and landing
	// on wheels is a prerequisite for everything else.
	float recover = 8.f;

	// The jump-flip strike: ball at jump height, car already rolling at it with
	// pace, so the only open decisions are when to leave the ground and
	// whether to flip.
	float strike = 15.f;

	// Ball in the air, cars on the ground: the lesson is "the ball is up, go
	// and meet it", which the touch reward pays for directly.
	float aerial = 10.f;

	// Full kickoffs mixed into training; the one policy plays its own
	// kickoffs, so it has to practise them here.
	float kickoff = 8.f;

	// Zero until the reward function pays for them. Zero-weight setters are
	// dropped entirely, so these cost nothing while disabled.
	float airDribble = 0.f;
	float flipReset = 0.f;
	float demo = 0.f;
};

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
inline constexpr float STEPS_PER_SECOND = 15.f;

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
inline constexpr float REFERENCE_EPISODE_SECONDS = 26.0f;
inline constexpr float REFERENCE_EPISODE_STEPS =
	STEPS_PER_SECOND * REFERENCE_EPISODE_SECONDS;

// Touch-units earned by holding a behaviour perfectly for one reference
// episode, converted to the per-step weight that earns it.
inline constexpr float RateWeight(float budgetPerEpisode) {
	return budgetPerEpisode / REFERENCE_EPISODE_STEPS;
}

// Touch-units of cost for one second of a condition, converted to per-step.
inline constexpr float PerSecondWeight(float budgetPerSecond) {
	return budgetPerSecond / STEPS_PER_SECOND;
}

// All values are touch-units. This is the early-stage stack from Zealan's
// RLGym-PPO-Guide (making_a_good_bot.md) in the guide's own proportions --
// touch 50, speed-to-ball 5, face-ball 1, air 0.15 -- divided through by the
// touch weight so that a touch is 1.0, then expressed as episode integrals.
//
// Nothing here is this project's invention, deliberately. See
// docs/superpowers/specs/2026-08-18-known-good-baseline-design.md.
struct RewardBudget {
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
	float touchGoalAccel = 62.14f;

	// The convexity. 2 is shipped for p13; 3 (64x rather than 16x for 80 kph
	// over 20 kph) is where this should end up, but only once mean hit force
	// clears ~40 kph. The learnability constraint is the gradient available at
	// the CURRENT operating point relative to the target one: (x_now/x_target)
	// ^(p-1). At today's 15.2 kph against 80 kph that is 19% for p=2, 3.6% for
	// p=3 and 0.7% for p=4. Move it on the measurement, not on a schedule.
	float touchAccelExponent = 2.0f;

	// Opponent penalty scale for TouchGoalAccel.
	// Unlike GoalReward which is 100% zero-sum (+1 scored, -1 conceded),
	// this penalizes opponent goal-directed touches at 50% (0.5).
	// This discourages allowing opponent attacks without making conceded
	// touches overwhelmingly punitive.
	float touchGoalAccelOpponentScale = 0.5f;

	// Fraction of TouchGoalAccel reward shared with teammates in multi-car modes.
	float touchGoalAccelTeamSpirit = 0.0f;

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
	float goal = 40.0f;

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
	float touchEdge = 0.1394f;

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
	float speedToBall = 14.51f;

	// Kept at the guide's speedToBall/5 ratio through the cut (14.51/5 = 2.90
	// would be the strict ratio; 1.585 is what the share solve returns, because
	// FaceBall's realized value per step is much closer to its maximum than
	// SpeedToBall's is, so equal budgets do not buy equal mass). SIGNED, so
	// driving backwards at the ball is punished rather than merely unpaid --
	// see the pairing note in Rewards.h. Target share 0.045.
	float faceBall = 1.585f;

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
	float saveBoost = 0.785f;

	// sqrt(new) - sqrt(old) on pickup, i.e. the increment of saveBoost's own
	// potential, so a full grab from empty pays 1.0 x this budget. Being the
	// derivative of a sqrt, it pays disproportionately for topping up when low,
	// which is most of what the guide wants from small-pad behaviour.
	//
	// Target share 0.02, i.e. held at p12's measured 0.015.
	float pickupBoost = 0.5067f;

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
	float airTouch = 40.f;

	// How steeply the air-touch reward scales with ball height. 1.0 is the
	// guide's linear form, which p13 measured being collected by ordinary
	// jump-touches at z ~350. 2.0 keeps paying those -- a jump to reach a high
	// ball is a real aerial, just a small one -- while making a genuine one
	// worth several times more. See AirTouchReward.
	float airTouchHeightExponent = 2.f;

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
	float air = 0.8736f;

	// Forward closing velocity gained from dodging/flipping towards the ball
	// (normalized by 500 uu/s impulse). Provides the discovery gradient for
	// ground flipping and speed-flips to traverse the pitch and preserve boost.
	float flipSpeed = 0.5f;
};

struct SelfPlayConfig {
	// Play a fraction of games against saved old versions.
	bool trainAgainstOldVersions = true;

	// Chance per iteration that it trains against an old version.
	float trainAgainstOldChance = 0.2f;

	// How often to snapshot the current policy into the version pool.
	// GigaLearn's own default (25M) suits multi-billion-step runs but barely
	// engages in a short comparison run; 5M gives a usable pool quickly.
	// Raise it towards 25M for long runs.
	int64_t tsPerVersion = 5'000'000;

	int maxOldVersions = 32;

	// Skill tracking is what makes two runs comparable, so it is worth
	// enabling even without trainAgainstOldVersions.
	bool trackSkill = true;

	// Evaluation arenas compete with training for CPU; keep well under the
	// core count.
	int skillArenas = 8;

	// Iterations between evaluation runs.
	int skillUpdateInterval = 100;

	float skillSimTime = 45.f;
	float skillMaxSimTime = 240.f;
};

struct TrainConfig {
	// Players per team the observation reserves space for. Changing this
	// changes the observation width, which invalidates every existing
	// checkpoint -- treat it as permanent once a run you care about starts.
	int maxPlayersPerTeam = 1;

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
	enum class SpawnMode { Random, Curriculum };
	SpawnMode spawn = SpawnMode::Random;

	CurriculumWeights curriculum = {};

	// Whether the action parser masks actions by situation. See Actions.h: the
	// mask more than doubles the grounded jump prior (42.9% vs 20%) relative to
	// every reference implementation. Must match at deployment.
	bool maskActions = false;

	// Which observation the policy sees. `Relative` is `Default` plus car-frame
	// relative geometry for the ball and every other car (see RelativeObs.h);
	// `Default` is RLGymCPP's DefaultObsPadded, which p8ref ran on.
	//
	// Changing this changes the observation WIDTH, so it invalidates every
	// existing checkpoint. Must match at deployment -- a width mismatch throws
	// at load, but a same-width layout mismatch would not, so `verify` checks
	// it explicitly.
	ObsMode obs = ObsMode::Relative;

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
	float infiniteBoostChance = 0.1f;

	RewardBudget rewards = {};
	// Fraction of reward shared between teammates on zero-sum terms (0.0 for
	// pure individual in 1v1, 0.3-1.0 in NvN).
	float teamSpirit = 0.0f;
	ModelShape modelShape = {};
	SelfPlayConfig selfPlay = {};

	float noTouchTimeoutSeconds = 12.f;

	int numGames = 128;

	int tickSkip = 8;
	int actionDelay = 7;

	// --- PPO ---------------------------------------------------------------
	// Guide, learner_settings.md: "values of 50_000 are good for early
	// learning, but once the bot is actually hitting the ball, it should be
	// increased to 100_000". Halving it from 100k also doubles the number of
	// policy updates per unit of experience, which is what an under-updating
	// run needs.
	int tsPerItr = 50'000;

	// Minibatch is the main VRAM knob (6 GB available).
	int miniBatchSize = 25'000;

	// Upstream default; guide recommends 2 or 3.
	int epochs = 2;

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
	float entropyScale = 0.002f;

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
	float entropyTarget = 0.40f;

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
	float entropyAdjustRate = 0.15f;

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
	float gaeGamma = 0.99f;

	// Guide, learner_settings.md: "Bot that can't score yet: 2e-4".
	float policyLR = 2e-4f;
	float criticLR = 2e-4f;

	int64_t maxSteps = 0;

	// --- Bookkeeping -------------------------------------------------------
	std::filesystem::path checkpointRoot = "checkpoints";

	// Distinguishes runs. Without it, a second run resumes from the first
	// one's checkpoints, which silently invalidates any comparison between
	// them.
	std::string runLabel = {};

	int64_t tsPerSave = 1'000'000;
	int checkpointsToKeep = 8;
	int64_t randomSeed = -1; // -1 uses the clock

	bool sendMetrics = true;
	std::string wandbProject = "hivemind-rl";
	std::string wandbGroup = "dev";

	bool renderMode = false; // Stream to RocketSimVis instead of training fast
	float renderTimeScale = 1.f;

	bool useGPU = true;

	std::string RunName() const {
		return runLabel.empty() ? std::string("main") : "main-" + runLabel;
	}

	std::filesystem::path CheckpointFolder() const {
		return checkpointRoot / RunName();
	}
};

} // namespace Hive
