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

// Penalty for any part of the car that is not its wheels being against a
// surface.
//
// `worldContact.hasContact` is set only by
// Arena::_BtCallback_OnCarWorldCollision -- the chassis hitbox producing a
// Bullet manifold against world geometry. Wheels are raycast suspension and
// never generate a manifold, so this is exactly "something that is not a wheel
// is touching a surface", and it is correct on walls, the corner curve and the
// ceiling with no plane assumption anywhere.
//
// Gated on !isOnGround, which is defined as 3+ wheels in contact and is
// therefore the in-control discriminator: if the wheels were doing their job
// you would never be inside this penalty. That gate is why there is no
// orientation grading -- being on your side is as useless as being on your
// roof, so grading would only distinguish 45-degrees-wrong from
// 90-degrees-wrong.
//
// The recovery gradient grading would have bought is unnecessary:
// Car::_UpdateAutoFlip makes escaping your roof a single discrete input (jump,
// while chassis-contacting), and the epsilon-floor patch keeps that input
// sampled.
//
// Returns a NEGATIVE value and carries a POSITIVE weight (see the sign
// convention in GeneralRewardSpecs).
class WrongSurfaceReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return (player.worldContact.hasContact && !player.isOnGround) ? -1.f : 0.f;
	}
};

// Reference impact speed for a clean landing: a ~1000 uu aerial returns at
// sqrt(2 * 650 * 1000) = 1140 uu/s. A held single jump leaves the ground at
// ~453 uu/s (JUMP_IMMEDIATE_FORCE 875/3, plus JUMP_ACCEL 4375/3 held for
// JUMP_MAX_TIME 0.2 s, less GRAVITY_Z over the hold) and returns at the same.
inline constexpr float LANDING_REF_IMPACT = 1100.f;

// Pays for arriving back on the wheels, scaled by the fall that was absorbed.
//
// This is what makes going airborne net-POSITIVE rather than merely permitted.
// Without it the only term touching air play is a penalty, in a project that
// has extinguished the jump action three times.
//
// SQUARED, and that is the entire anti-farm argument. Under linear scaling a
// bunny hop pays 0.41 every ~1.6 s (0.24/s) against a real aerial's 1.0 every
// ~3.7 s (0.27/s) -- hopping is competitive, so it is a farm. Squared, the hop
// drops to 0.098/s and the aerial dominates 2.8x. The only way to raise this
// term is to go higher, which costs time and boost that could have gone at the
// ball.
//
// Measured as downward speed rather than |vel|: using speed would pay for
// horizontal pace, double-counting SpeedSquaredReward and biasing toward the
// wall. The accepted cost is that a wall landing has no vertical component and
// scores zero.
class CleanLandingReward : public RLGC::Reward {
public:
	float refImpact;

	explicit CleanLandingReward(float refImpact = LANDING_REF_IMPACT)
		: refImpact(refImpact) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		// EnvSet::ResetArena empties prevGameStates, so a null prev means the
		// episode just reset and any velocity change is a state-setter
		// teleport, not a landing.
		if (!player.prev)
			return 0.f;

		// The landing edge, and only a clean one. Chassis contact on the same
		// step means this was a crash, which WrongSurfaceReward is already
		// charging -- paying here too would let the bot buy its way out of that
		// penalty by crashing faster.
		if (!player.isOnGround || player.prev->isOnGround || player.worldContact.hasContact)
			return 0.f;

		const float impact = RS_MAX(0.f, -player.prev->vel.z);
		const float f = RS_MIN(1.f, impact / refImpact);
		return f * f;
	}
};

// One payment per contact SEQUENCE, not per step of contact.
//
// A per-step touch reward IS a dribble reward: carrying the ball on the nose
// collects it every step, roughly 180 times in an episode. That is the
// flick-bot local optimum (roadmap spec D4) arriving through the back door.
// The rising edge makes carrying the ball worth exactly one touch, so the term
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

// |forward . dirToBall|, shipped alongside upstream's signed FaceBallReward
// because together the two ARE the asymmetric form:
//
//   w+ * max(0,c) + w- * min(0,c)  ==  ((w+ + w-)/2) * c  +  ((w+ - w-)/2) * |c|
//
// Facing away from the ball is sometimes correct (shadow defence, retreating
// for a bounce), so the negative side should be weaker than the positive. But
// implementing that as rectified weights silently ships the second component,
// which pays IDENTICALLY for nose-at-ball and nose-directly-away and pays zero
// for perpendicular -- and which is an annuity, since for a policy with no
// facing preference c is uniform on [-1,1] and E|c| = 0.5.
//
// Split out so it gets its own RewardShare line and its own budget.
// RewardShare reports mean |r*w| and cannot tell a signed term from a
// rectified one, which is exactly how this would have gone unnoticed.
class FaceBallAxisReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		return std::fabs(player.rotMat.forward.Dot(toBall / len));
	}
};

// Throttle-only top speed: DRIVE_SPEED_TORQUE_FACTOR_CURVE reaches zero here,
// so any car can hold this indefinitely with no boost and no skill.
inline constexpr float THROTTLE_TOP_SPEED = 1410.f;

// (|v| / CAR_MAX_SPEED)^2.
//
// Squared, not linear, because linear leaves 1410/2300 = 0.613 of the term's
// maximum as a free annuity for holding throttle in a straight line. Squaring
// cuts that to 0.375 and raises the payoff for boost- and flip-derived speed
// over coasting from 1.63x to 2.67x, while keeping a rising gradient from zero
// so the term still bootstraps.
//
// GENERIC speed, not speed-toward-ball, on purpose. The ball-directed form is
// a PRODUCT of speed and alignment, and its cross term
// (d2R/d|v| dcos = 1/V, nonzero) charges a steering input on both factors at
// once: turning scrubs speed AND misaligns velocity. That is what drove
// Action/Steer Nonzero to 0.0006 on p3strike. SpeedSquared + FaceBall is the
// same intent factored additively, where a turn is charged once.
//
// The same factoring is why previous bots never flipped or boosted for speed:
// a flip's impulse is along the car's forward axis and costs ~1.25 s of
// steering authority, so under the product form its value is gated by
// alignment, while under |v| it is paid unconditionally.
class SpeedSquaredReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const float f = RS_MIN(1.f, player.vel.Length() / RLGC::CommonValues::CAR_MAX_SPEED);
		return f * f;
	}
};

// Speed that can be lost in one decision step without a collision. RL brakes at
// roughly 3500 uu/s^2, which over a 1/15 s step is 233 uu/s, so 400 is clear of
// any input-driven deceleration and only a collision reaches it.
//
// That 3500 is EMPIRICAL, not a RocketSim constant -- BRAKE_TORQUE_AMOUNT is a
// wheel torque and does not convert directly. The Speed/Max Step Decel metric
// exists to check this threshold against the real distribution. Do not treat
// 400 as settled.
inline constexpr float HARSH_LOSS_THRESHOLD = 400.f;

// Penalty for losing a lot of speed in one step: a wall, a bad recovery, a
// botched landing.
//
// Deliberately overlaps SpeedSquaredReward, which already makes losing speed
// cost future reward. What this adds is a sharp, single-step signal
// attributable to the collision itself, which is worth real money for credit
// assignment when gaeLambda puts the direct credit horizon around 1 second. It
// also fires alongside WrongSurfaceReward on the same events. Both overlaps are
// intentional and are recorded so the RewardShare numbers are not misread.
//
// Returns a NEGATIVE value and carries a POSITIVE weight.
class HarshSpeedLossReward : public RLGC::Reward {
public:
	float threshold;

	explicit HarshSpeedLossReward(float threshold = HARSH_LOSS_THRESHOLD)
		: threshold(threshold) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev)
			return 0.f;

		// A hard hit SHOULD cost speed: that is a good outcome, not a bad
		// recovery. Charging for it would penalise striking the ball.
		if (player.ballTouchedStep)
			return 0.f;

		const float lost = player.prev->vel.Length() - player.vel.Length();
		if (lost <= threshold)
			return 0.f;

		const float span = RLGC::CommonValues::CAR_MAX_SPEED - threshold;
		return -RS_MIN(1.f, (lost - threshold) / span);
	}
};

// Returns heap-allocated rewards; GigaLearn's EnvSet takes ownership, so the
// caller does not free them. Several entries are wrapped in ZeroSumReward,
// converting "reward for this event" into "how much better did I do than the
// opposition" so both sides can't be paid for the same event.

// One reward term with a stable metric name, since Reward::GetName() is
// useless for that on GCC (near-mangled typeid strings).
struct RewardSpec {
	std::string name;   // metric label, e.g. "StrongTouch"
	float weight;
	std::function<RLGC::Reward*()> make;
};

// The order of specs matches the order of envSet->rewards[arena] and
// envSet->state.lastRewards[arena]; the reward-share metrics index by it.
std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg);

// Materializes the specs, same order.
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

} // namespace Hive
