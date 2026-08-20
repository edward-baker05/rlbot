#pragma once

#include "../Config.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace Hive {

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
class TouchEdgeReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		// A null prev means the episode just reset, so a touch on this step is
		// a genuine new contact rather than the continuation of one.
		return (player.prev && player.prev->ballTouchedStep) ? 0.f : 1.f;
	}
};

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
class SpeedToBallReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		const float closing = player.vel.Dot(toBall / len);
		return RS_MAX(0.f, closing) / RLGC::CommonValues::CAR_MAX_SPEED;
	}
};

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
class TouchGoalAccelReward : public RLGC::Reward {
public:
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
	explicit TouchGoalAccelReward(float exponent) : exponent(exponent) {}

	float exponent;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep || !state.prev)
			return 0.f;

		const Vec target = (player.team == Team::BLUE)
			? RLGC::CommonValues::ORANGE_GOAL_CENTER
			: RLGC::CommonValues::BLUE_GOAL_CENTER;

		// One direction for both samples: we want the change in speed toward
		// the net, not a change that includes the ball having moved.
		const Vec toGoal = (target - state.ball.pos).Normalized();

		const float now = state.ball.vel.Dot(toGoal);
		const float before = state.prev->ball.vel.Dot(toGoal);

		// Saturation stays at 130 kph so 1.0 is still one maximal goal-directed
		// strike and every earlier budget reads in the same unit. Sign is kept
		// outside the power so putting the ball toward your OWN net still
		// costs, and costs convexly too.
		const float x =
			RS_CLAMP((now - before) / RLGC::Math::KPHToVel(130), -1.f, 1.f);
		return std::copysign(std::pow(std::fabs(x), exponent), x);
	}
};

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
class AirTouchReward : public RLGC::Reward {
public:
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
	explicit AirTouchReward(float heightExponent) : heightExponent(heightExponent) {}

	float heightExponent;

	// A rough ceiling on a reasonable aerial, from the guide. Longer air times
	// are not worth more: this pays for reaching the ball, not for floating.
	static constexpr float MAX_AIR_TIME = 1.75f;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		// RISING EDGE, matching TouchEdgeReward, and NOT optional at p13's
		// budget. The p12 form paid on every contact STEP, so an air carry at
		// ceiling height would have earned ~170 touch-units per second -- the
		// p9rel dribble farm, relocated to the air. It stayed harmless in p12
		// only because the budget was too small for anything to happen.
		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		const float airTimeFrac = RS_MIN(player.airTime, MAX_AIR_TIME) / MAX_AIR_TIME;
		const float heightFrac =
			RS_MAX(0.f, state.ball.pos.z / RLGC::CommonValues::CEILING_Z);

		// The min() with air time is untouched: it is what makes a wall shot
		// worth exactly zero (a car on a wall is isOnGround, so airTime is 0),
		// and that is still the farm this bot is closest to.
		return RS_MIN(airTimeFrac, std::pow(heightFrac, heightExponent));
	}
};

// Rewards the forward closing velocity gained from dodging/flipping towards the
// ball. Boost-neutral: does not check or punish boost, allowing speed-flips to
// emerge naturally while providing the gradient needed for ground flip discovery.
class FlipSpeedReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev || !state.prev)
			return 0.f;

		// Trigger only on the rising edge of a dodge (flip initiation)
		const bool justFlipped = (player.isFlipping || player.hasFlipped) &&
		                         (!player.prev->isFlipping && !player.prev->hasFlipped);
		if (!justFlipped)
			return 0.f;

		// Inactive if already at supersonic speed before the flip
		if (player.prev->vel.Length() >= RLGC::CommonValues::SUPERSONIC_THRESHOLD)
			return 0.f;

		// Measure velocity gain in the direction of the ball
		const Vec toBall = (state.ball.pos - player.pos).Normalized();
		const float closingNow = player.vel.Dot(toBall);
		const float closingPrev = player.prev->vel.Dot(toBall);
		const float deltaClosing = closingNow - closingPrev;

		if (deltaClosing <= 50.f)
			return 0.f;

		// Normalized against the 500 uu/s impulse of a standard forward dodge
		return RS_CLAMP(deltaClosing / 500.f, 0.f, 1.f);
	}
};

// Rewards collecting boost, with an elevated base floor for small pads (12 boost)
// so clipping pads into general ground routes remains attractive even when partly full.
class TieredPickupBoostReward : public RLGC::Reward {
public:
	float smallPadBase = 0.25f;

	TieredPickupBoostReward(float smallPadBase = 0.25f) : smallPadBase(smallPadBase) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev)
			return 0.f;

		const float delta = player.boost - player.prev->boost;
		if (delta <= 0.f)
			return 0.f;

		// Big boost pads replenish up to 100 (delta > 25); small pads grant 12
		const bool isBig = (delta > 25.f);

		if (isBig) {
			// Big boost pickup: reward scaled by boost replenished (up to 1.0 for 0 -> 100)
			return 1.0f * (delta / 100.f);
		} else {
			// Small pad pickup: guaranteed base floor + diminishing potential gain
			const float potGain = std::sqrt(player.boost / 100.f) - std::sqrt(player.prev->boost / 100.f);
			return smallPadBase + potGain;
		}
	}
};


// ============================================================================
// Diagnostic constants
// ============================================================================
// Not reward weights. Thresholds for metrics in Train.cpp, kept here so the
// derivations stay next to the physics they come from.

// Throttle-only top speed: DRIVE_SPEED_TORQUE_FACTOR_CURVE reaches zero here,
// so any car can hold this indefinitely with no boost and no skill.
inline constexpr float THROTTLE_TOP_SPEED = 1410.f;

// Speed that can be lost in one decision step without a collision. RL brakes at
// roughly 3500 uu/s^2, which over a 1/15 s step is 233 uu/s, so 400 is clear of
// any input-driven deceleration and only a collision reaches it. That 3500 is
// EMPIRICAL, not a RocketSim constant.
inline constexpr float HARSH_LOSS_THRESHOLD = 400.f;

// Returns heap-allocated rewards; GigaLearn's EnvSet takes ownership, so the
// caller does not free them.

// One reward term with a stable metric name, since Reward::GetName() is
// useless for that on GCC (near-mangled typeid strings).
struct RewardSpec {
	std::string name;   // metric label, e.g. "SpeedToBall"
	float weight;
	std::function<RLGC::Reward*()> make;
};

// The order of specs matches the order of envSet->rewards[arena] and
// envSet->state.lastRewards[arena]; the reward-share metrics index by it.
std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg);

// Materializes the specs, same order.
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

} // namespace Hive
