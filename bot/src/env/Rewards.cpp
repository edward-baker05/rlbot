#include "Rewards.h"

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {
		{"Goal", b.goal, [] { return new GoalReward(); }},
		{"Strong Touch", b.strongTouch,
		 [] { return new ZeroSumReward(new DirectionalTouchReward(), 1); }},
		{"Air Touch", b.airTouch,
		 [] { return new ZeroSumReward(new ImprovedAirTouchReward(), 1); }},
		{"Boost Pickup", b.pickupBoost,
		 [] { return new ZeroSumReward(new PickupBoostReward(), 1); }},
		{"Save Boost", b.saveBoost, [] { return new SaveBoostReward(); }},
		{"Bump", b.bump, [] { return new BumpReward(); }},
		{"Demo", b.demo,
		 [] { return new ZeroSumReward(new DemoReward(), 0, 2); }},
		{"Save", b.save, [] { return new ZeroSumReward(new SaveReward(), 1); }},
		{"Awkward Contact", b.awkwardContact,
		 [] { return new AwkwardContactPenalty(); }},
		// {"Possession", b.possession, [] { return new PossessionReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
