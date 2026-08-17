#pragma once

#include "../Config.h"

#include <RLGymCPP/Rewards/Reward.h>

#include <functional>
#include <string>
#include <vector>

namespace Hive {

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
