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
// This is the single most useful addition to the stock reward set for getting
// a bot off the floor. Without it a policy converges on a strong ground game
// and simply never explores aerials, because every airborne attempt initially
// costs it ground-game reward. It is not in CommonRewards, so it lives here.
class TouchHeightReward : public RLGC::Reward {
public:
	// Height at which the reward saturates. Roughly the height of a
	// comfortable aerial; above this, more height is not more skill.
	float maxHeight;

	explicit TouchHeightReward(float maxHeight = 1500.f) : maxHeight(maxHeight) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override;
};

// ============================================================================
// Reward stacks
// ============================================================================
// Both builders return heap-allocated rewards. GigaLearn's EnvSet takes
// ownership of them, so the caller does not free them.
//
// A note on ZeroSumReward, which several entries are wrapped in: it converts a
// reward into "how much better did I do than the opposition", and its first
// argument is team spirit -- the fraction of a reward shared across teammates.
//
// Team spirit is the main lever for teamplay in 2s and 3s. At 0 each car is
// selfish and you get three bots chasing the same ball. At 1 credit is fully
// shared and individual contributions get lost in the noise, which slows
// learning badly. The values below sit deliberately low-to-middle: enough to
// discourage ball-chasing, not so much that the gradient signal washes out.

// The general policy: full-game rewards.
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const RewardWeights& w);

// The kickoff policy: short-horizon rewards only.
//
// Kickoff training ends at first touch, so rewards that need a long horizon
// (boost economy over a whole possession, positioning) have nothing to act on.
// Everything here pays off within the two seconds the model actually controls.
std::vector<RLGC::WeightedReward> BuildKickoffRewards(const KickoffRewardWeights& w);

} // namespace Hive
