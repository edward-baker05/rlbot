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
// Reward stacks
// ============================================================================
// Both builders return heap-allocated rewards; GigaLearn's EnvSet takes
// ownership, so the caller does not free them.
//
// ZeroSumReward wraps several entries. It converts a reward into "how much
// better did I do than the opposition", which matters for two reasons:
//
//   * It keeps the game adversarial. Without it, both teams can be paid for the
//     same event and total reward inflates while nobody plays better.
//   * Its first argument is team spirit -- the fraction of a reward shared
//     across teammates. Spirit is kept LOW in this phase (0.0-0.3). High spirit
//     is how you eventually get rotations, but it also smears credit across
//     cars, and a bot that cannot yet strike the ball needs to know precisely
//     which of its own actions worked. Raise it in the teamplay phase.

// The general policy. See RewardWeights in Config.h for the derivation of each
// weight and docs/rewards.md for the method.
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

// The kickoff policy: reset to first touch, so short-horizon only.
std::vector<RLGC::WeightedReward> BuildKickoffRewards(const KickoffRewardWeights& w);

} // namespace Hive
