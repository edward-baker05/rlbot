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
		 [=] {
			 return new ImprovedAirTouchReward(cfg.aerial.minBallHeight,
											   cfg.aerial.maxBallHeight);
		 }},
		{"Air Face Ball", b.airFaceBall,
		 [=] {
			 return new AirFaceBallReward(cfg.aerial.minBallHeight,
										  cfg.aerial.maxBallHeight);
		 }},
		{"Air Vel to Ball", b.airVelToBall,
		 [=] {
			 return new AirVelToBallReward(cfg.aerial.minBallHeight,
										   cfg.aerial.maxBallHeight);
		 }},
		{"Air Launch", b.airLaunch,
		 [=] {
			 return new AirLaunchReward(cfg.aerial.minBallHeight);
		 }},
		{"Boost Pickup", b.pickupBoost,
		 [] { return new ZeroSumReward(new PickupBoostReward(), 1); }},
		{"Save Boost", b.saveBoost, [] { return new SaveBoostReward(); }},
		{"Bump", b.bump, [] { return new BumpReward(); }},
		{"Demo", b.demo, [] { return new ZeroSumReward(new DemoReward(), 0); }},
		{"Save", b.save,
		 [] { return new ZeroSumReward(new SaveReward(), 1, 0.7); }},
		{"Awkward Contact", b.awkwardContact,
		 [] { return new AwkwardContactPenalty(); }},
		{"Possession", b.possession, [] { return new PossessionReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
