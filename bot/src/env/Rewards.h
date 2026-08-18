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

// StrongTouch's impact-speed window, in KPH. Lives here rather than in the .cpp
// because train/Train.cpp's metrics recompute this reward exactly, and two
// copies of these numbers could drift apart.
inline constexpr float TOUCH_MIN_KPH = 5.f;
inline constexpr float TOUCH_MAX_KPH = 100.f;

// Free functions, not private members, so train/Train.cpp's metrics can call
// the same code the reward uses.

// Impact strength, 0..1, from the ball's velocity CHANGE (not the car's speed).
inline float StrongTouchValue(float hitForce) {
	const float lo = RLGC::Math::KPHToVel(TOUCH_MIN_KPH);
	const float hi = RLGC::Math::KPHToVel(TOUCH_MAX_KPH);
	if (hitForce < lo)
		return 0.f;
	return RS_MIN(1.f, hitForce / hi);
}

// How well a hit is aimed, as a 0..1 multiplier on strength; upstream
// StrongTouchReward is direction-blind. Smooth (0.5*(1+cos), no hard cone) so
// a clear or a pass isn't punished like a shot into your own net.
inline float AimMultiplier(const Vec& ballVelDelta, const Vec& ballPos,
                           Team team, float sharpness) {
	const float len = ballVelDelta.Length();
	if (len < 1e-4f)
		return 0.f;

	const Vec target = (team == Team::BLUE)
		? RLGC::CommonValues::ORANGE_GOAL_BACK
		: RLGC::CommonValues::BLUE_GOAL_BACK;

	const Vec toGoal = target - ballPos;
	if (toGoal.Length() < 1e-4f)
		return 1.f;

	const float alignment = (ballVelDelta / len).Dot(toGoal.Normalized());
	const float m = 0.5f * (1.f + alignment);
	return (sharpness == 1.f) ? m : std::pow(m, sharpness);
}

// ============================================================================
// Custom rewards
// ============================================================================

// Rewards touching the ball high off the ground, scaled 0..1 by height. Used
// only in RewardPhase::Aerial. Measured from the ball's resting height, not
// zero, so a ground touch scores exactly zero instead of a farmable constant.
class TouchHeightReward : public RLGC::Reward {
public:
	// Height at which the reward saturates: roughly a comfortable aerial.
	float maxHeight;

	explicit TouchHeightReward(float maxHeight = 1500.f) : maxHeight(maxHeight) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override;
};

// Pays the child reward only while the player is on its wheels, and forwards
// lifecycle calls. Anti-farm gate for approach shaping: without it, flips give
// free velocity and tumbling pays exactly as well as driving.
class GroundedReward : public RLGC::Reward {
public:
	RLGC::Reward* child;

	explicit GroundedReward(RLGC::Reward* child) : child(child) {}
	~GroundedReward() override { delete child; }

	void Reset(const RLGC::GameState& initialState) override { child->Reset(initialState); }
	void PreStep(const RLGC::GameState& state) override { child->PreStep(state); }

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return player.isOnGround ? child->GetReward(player, state, isFinal) : 0.f;
	}
};

// Pays for pointing the wheels at the ground while airborne. `rotMat.up` is
// the car's own roof axis, so up.z == +1 means landing-ready and -1 inverted.
// VelPlayerToBall and FaceBall are both gated to grounded steps, so this is
// the only directional signal while airborne; needs no anti-farm gate because
// a flip randomizes orientation and so destroys this reward rather than
// collecting it.
class AirRecoveryReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return player.isOnGround ? 0.f : player.rotMat.up.z;
	}
};

// Flat payment for having wheels on the ground: "don't go airborne without a
// good reason." Deliberately not a punishment for jumping -- the bot must
// still be free to leave the ground when it's worth it.
class GroundedBonusReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return player.isOnGround ? 1.f : 0.f;
	}
};


// StrongTouch, but paid only in proportion to how well the hit is aimed at the
// opponent goal; see AimMultiplier. A multiplier rather than an additive aim
// term, since additive could be farmed by nudging the ball goalward at zero
// power.
class AimedStrongTouchReward : public RLGC::Reward {
public:
	// 1 = the plain 0.5*(1+cos) curve. Higher values narrow the useful cone.
	float sharpness;

	explicit AimedStrongTouchReward(float sharpness = 1.f) : sharpness(sharpness) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev || !player.ballTouchedStep)
			return 0.f;

		const Vec delta = state.ball.vel - state.prev->ball.vel;
		const float strength = StrongTouchValue(delta.Length());
		if (strength <= 0.f)
			return 0.f;

		return strength * AimMultiplier(delta, state.ball.pos, player.team, sharpness);
	}
};

// Potential-based approach shaping: pays the DISTANCE CLOSED to the ball this
// step, not the speed at which it was closed.
//
// Replaces GroundedReward(VelocityPlayerToBallReward), which computed
// `dirToBall . (vel / CAR_MAX_SPEED)` -- dividing by a CONSTANT, so it scaled
// with how fast you were going rather than whether you were going the right
// way. Boosting flat out in a straight line was its global maximum, and
// steering was penalised twice: turning scrubs speed AND points velocity off
// the ball line. Measured on p3strike at 100M steps, the policy had driven
// `Action/Steer Nonzero` to 0.0006 -- it had stopped steering at all.
//
// This is the potential difference of Phi(s) = -dist(car, ball), which makes it
// policy-invariant in the sense of Ng et al. (1999): it can speed learning up
// but cannot move the optimum, because around any cycle the terms telescope to
// zero. Approach then retreat nets exactly nothing, so there is no farm.
//
// Two deliberate departures:
//
//   * gamma = 1 rather than gaeGamma. With Phi negative and gamma < 1 a
//     motionless car collects (1 - gamma) * dist every step -- a bonus for
//     standing still far from the ball, which is exactly the pathology being
//     removed here. Exact invariance at gamma = 0.99 is worth less than not
//     reintroducing an annuity.
//
//   * NOT wrapped in GroundedReward. Gating breaks the telescoping property and
//     is what made the old term farmable: close on the ground (paid), retreat
//     through the air (free). A true potential needs no anti-farm gate, since
//     no cycle of any kind can accumulate reward.
class BallProgressReward : public RLGC::Reward {
public:
	// Divisor only; sets the units. CAR_MAX_SPEED keeps the per-step magnitude
	// close to the term this replaces (max closing is ~153 uu/step, so ~0.066),
	// which keeps the existing weight meaningful.
	float scale;

	explicit BallProgressReward(float scale = RLGC::CommonValues::CAR_MAX_SPEED)
		: scale(scale) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev || !player.prev)
			return 0.f;

		const float dPrev = (state.prev->ball.pos - player.prev->pos).Length();
		const float dCur = (state.ball.pos - player.pos).Length();
		return (dPrev - dCur) / scale;
	}
};

// Potential-based shaping on the BALL's progress toward the opponent goal.
//
// Phi(s) = -dist(ball, their goal). Same telescoping no-farm guarantee as any
// potential, but chosen so the potential RISES exactly when the bot does the
// thing we want. That is the fix for what killed p4pbrs, which used
// Phi = -dist(car, ball): chase-hit-chase is a cycle in that potential, so it
// summed to ~zero over an episode and taught nothing, and in the short run it
// actively penalised striking the ball (a hard hit sends the ball away, which
// lowered the potential until the car caught up). A term that cannot be farmed
// around a cycle also cannot teach around one -- the potential has to be on a
// quantity the agent's own success improves, not one it degrades.
//
// gamma = 1 for the same reason as BallProgressReward: with a negative
// potential and gamma < 1, a static ball pays (1 - gamma) * dist every step,
// which is an annuity for leaving the ball far from the goal.
//
// NOT wrapped in ZeroSumReward, unlike the VelBallToGoal it replaces. The
// potential is already naturally opposed between teams -- ball motion toward
// their goal is motion away from ours, so one side's gain is the other's loss
// by construction. Wrapping it would subtract that twice.
class BallGoalProgressReward : public RLGC::Reward {
public:
	// Roughly the longest ball-to-goal distance on the pitch, so Phi lands in
	// about [-1, 0] and a maximum-speed ball moves it ~0.036 per step -- the
	// same order as what a grounded approach step pays (0.045 measured).
	float scale;

	explicit BallGoalProgressReward(float scale = 11000.f) : scale(scale) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev)
			return 0.f;

		const Vec target = (player.team == Team::BLUE)
			? RLGC::CommonValues::ORANGE_GOAL_BACK
			: RLGC::CommonValues::BLUE_GOAL_BACK;

		const float dPrev = (target - state.prev->ball.pos).Length();
		const float dCur = (target - state.ball.pos).Length();
		return (dPrev - dCur) / scale;
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
