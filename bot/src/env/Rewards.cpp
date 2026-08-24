#include "Rewards.h"

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {
		{"Goal", b.goal, [] { return new GoalReward(); }},
		{"Strong Touch", b.strongTouch,
		 [] { return new ZeroSumReward(new DirectionalTouchReward(), 1); }},
		{"Velocity: Ball to Goal", b.velocityBallToGoal,
		 [] { return new VelocityBallToGoalReward(); }},
		{"Velocity: Player to Ball", b.velocityPlayerToBall,
		 [] { return new VelocityPlayerToBallReward(); }},
		{"Face Ball", b.faceBall, [] { return new FaceBallReward(); }},
		{"Air Touch", b.airTouch,
		 [] { return new ZeroSumReward(new ImprovedAirTouchReward(), 1); }},
		{"Speed", b.speed,
		 [] { return new ZeroSumReward(new SpeedReward(), 1); }},
		{"Boost Pickup", b.pickupBoost,
		 [] { return new ZeroSumReward(new PickupBoostReward(), 1); }},
		{"Save Boost", b.saveBoost, [] { return new SaveBoostReward(); }},
		// {"Flip", b.flip, [] { return new FlipReward(); }}
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
