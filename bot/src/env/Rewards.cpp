#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

using namespace RLGC;

namespace Hive {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardBudget& b = cfg.rewards;

	return {
		{"TouchGoalAccel", b.touchGoalAccel,
		 [e = b.touchAccelExponent, s = b.touchGoalAccelOpponentScale, ts = b.touchGoalAccelTeamSpirit] {
			return new ZeroSumReward(new TouchGoalAccelReward(e), ts, s);
		 }},

		{"Goal", b.goal, [] { return new GoalReward(); }},

		{"TouchEdge", b.touchEdge, [] { return new TouchEdgeReward(); }},

		{"SpeedToBall", RateWeight(b.speedToBall),
		 [] { return new SpeedToBallReward(); }},

		{"FaceBall", RateWeight(b.faceBall),
		 [] { return new FaceBallReward(); }},

		{"SaveBoost", RateWeight(b.saveBoost), [] { return new SaveBoostReward(); }},

		{"PickupBoost", b.pickupBoost, [] { return new TieredPickupBoostReward(); }},

		{"FlipSpeed", b.flipSpeed, [] { return new FlipSpeedReward(); }},

		{"AirTouch", b.airTouch,
		 [e = b.airTouchHeightExponent] { return new AirTouchReward(e); }},

		{"Air", RateWeight(b.air), [] { return new AirReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

}  // namespace Hive
