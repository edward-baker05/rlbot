#pragma once

#include "../Config.h"

#include <RLGymCPP/Rewards/Reward.h>

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
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

} // namespace Hive
