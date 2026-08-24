#include "Rewards.h"

#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {{"Air", b.air, new AirReward()},
			{"Face Ball", b.faceBall, new FaceBallReward()},
			{"Velocity: Player to Ball", b.velocityPlayerToBall,
			 new VelocityPlayerToBallReward()},
			{"Strong Touch", b.strongTouch, new StrongTouchReward(20, 100)},
			{"Velocity: Ball to Goal", b.velocityBallToGoal,
			 new ZeroSumReward(new VelocityBallToGoalReward(), 1)},
			{"Pickup Boost", b.pickupBoost, new PickupBoostReward(), 10.f},
			{"Save Boost", b.saveBoost, new SaveBoostReward()},
			{"Bump", b.bump, new ZeroSumReward(new BumpReward(), 0.5f)},
			{"Demo", b.demo, new ZeroSumReward(new DemoReward(), 0.5f)},
			{"Goal", b.goal, new GoalReward(), 150}};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
