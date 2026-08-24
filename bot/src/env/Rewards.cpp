#include "Rewards.h"

#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {{"Goal", b.goal, [] { return new GoalReward(); }},
			{"Strong Touch", b.strongTouch,
			 [] { return new ZeroSumReward(new DirectionalTouchReward(), 1); }},
			{"Velocity: Ball to Goal", b.velocityBallToGoal,
			 [] { return new VelocityBallToGoalReward(); }},
			{"Velocity: Player to Ball", b.velocityPlayerToBall,
			 [] { return new VelocityPlayerToBallReward(); }},
			{"Face Ball", b.faceBall, [] { return new FaceBallReward(); }},
			{"Air", b.air, [] { return new AirReward(); }}};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
