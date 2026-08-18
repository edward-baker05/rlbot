#pragma once

#include "../Config.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/Reward.h>

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
