#include "Rewards.h"

#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {
		{"Goal", b.goal, [] { return new GoalReward(); }},
		{"Ball Touch Acceleration", b.touchAccel,
		 [] { return new TouchAccelReward(); }},
		{"Velocity: Player to Ball", b.velocityPlayerToBall,
		 [] { return new VelocityPlayerToBallReward(); }},
		{"Velocity: Ball to Goal", b.velocityBallToGoal,
		 [] { return new VelocityBallToGoalReward(); }},
		{"Demo", b.demo, [] { return new DemoReward(); }},
		{"Face Ball", b.faceBall, [] { return new RLGC::FaceBallReward(); }}};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
