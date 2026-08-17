#pragma once

#include "../Config.h"

#include <RLGymCPP/Rewards/Reward.h>

#include <functional>
#include <string>
#include <vector>

namespace Hive {

// StrongTouch's impact-speed window, in KPH. Lives here rather than in the .cpp
// because the metrics in train/Train.cpp recompute StrongTouchReward exactly to
// report what a touch actually pays; two copies of these numbers would let that
// silently drift from the reward it is meant to measure.
inline constexpr float TOUCH_MIN_KPH = 5.f;
inline constexpr float TOUCH_MAX_KPH = 100.f;

// ============================================================================
// Custom rewards
// ============================================================================

// Rewards touching the ball high off the ground, scaled 0..1 by height.
//
// NOT used in RewardPhase::Foundations. It belongs to the aerial phase, and
// enabling it early is actively harmful: before the bot can reliably strike a
// ball at all, paying for height just teaches it to jump at everything and
// miss. Kept here because it is the right tool once touch rate is healthy.
//
// Measured from the ball's resting height so that a touch on the ground scores
// zero rather than a small constant -- otherwise a ground dribble farms it.
class TouchHeightReward : public RLGC::Reward {
public:
	// Height at which the reward saturates: roughly a comfortable aerial.
	float maxHeight;

	explicit TouchHeightReward(float maxHeight = 1500.f) : maxHeight(maxHeight) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override;
};

// Pays the child reward only while the player is on its wheels, and forwards
// lifecycle calls. Anti-farm gate for approach shaping: flips give free
// velocity, so an ungated velocity-to-ball term pays tumbling exactly as well
// as driving -- ~73% of DefaultAction's 90 actions involve air/jump inputs,
// so a fresh policy is airborne ~90% of the time and never discovers that
// driving exists unless the money is on the ground.
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

// Pays for pointing the wheels at the ground while airborne.
//
// `rotMat.up` is the car's own roof axis, so up.z == +1 means the roof faces
// the sky and the wheels face the floor -- landing-ready. -1 is inverted.
//
// Why this exists: VelPlayerToBall and FaceBall are both gated to grounded
// steps, and the bot is airborne ~92% of the time, so almost its entire life
// carries no directional reward at all. Measured on p1advnorm, that desert is
// where behaviour rots: touch rate falls 25x from the first second of an
// episode (0.0067) to four seconds in (0.00027) while approach speed drops
// 959 -> 192. The bot never re-orients because nothing ever paid it to.
//
// It resists the farming failure that sank the other shaping terms, for two
// reasons. Orientation is entirely under the policy's control -- unlike ball
// velocity or touch outcomes, which are mostly not -- so the signal is clean
// rather than noise. And a flip randomizes orientation, so flip-spam actively
// destroys this reward instead of collecting it; it needs no anti-farm gate.
class AirRecoveryReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return player.isOnGround ? 0.f : player.rotMat.up.z;
	}
};

// Flat payment for having wheels on the ground.
//
// "Do not go in the air without a good reason" is the standard beginner rule in
// Rocket League, and there is no reason to depart from it at this stage. This
// makes a grounded step strictly better than an airborne one: the flat bonus
// alone beats a perfectly-oriented airborne step, and a grounded car also
// collects VelPlayerToBall and FaceBall on top, which an airborne one cannot.
//
// Deliberately NOT a punishment for jumping. The bot must still be free to
// leave the ground when that is worth it -- it just has to be worth it.
class GroundedBonusReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return player.isOnGround ? 1.f : 0.f;
	}
};

// ============================================================================
// Reward stack
// ============================================================================
// Returns heap-allocated rewards; GigaLearn's EnvSet takes ownership, so the
// caller does not free them.
//
// ZeroSumReward wraps several entries. It converts a reward into "how much
// better did I do than the opposition": without it both sides can be paid for
// the same event, so total reward inflates while nobody plays better. Its
// first argument is team spirit, meaningless in 1v1 -- kept at the upstream
// defaults.

// One reward term with a stable metric name. The name is what the per-term
// reward-share metrics report against; Reward::GetName() is useless for that
// on GCC (near-mangled typeid strings).
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
