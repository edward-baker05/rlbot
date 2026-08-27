// Included so the imported community reward pack is compiled on every build
// rather than rotting until the day something first tries to use it.
#include "CommunityRewards.h"
#include "Rewards.h"

using namespace RLGC;

namespace Dash {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {
		{"Goal", b.goal, RewardKind::Event,
		 [] { return new GoalReward(-1, true); }},
		{"Strong Touch", b.strongTouch, RewardKind::Event,
		 [] { return new ZeroSumReward(new DirectionalTouchReward(), 1); }},
		{"Air Touch", b.airTouch, RewardKind::Event,
		 [=] {
			 return new ImprovedAirTouchReward(cfg.aerial.minBallHeight,
											   cfg.aerial.maxBallHeight);
		 }},
		{"Boost Pickup", b.pickupBoost, RewardKind::Event,
		 [] { return new ZeroSumReward(new PickupBoostReward(), 1); }},
		{"Save Boost", b.saveBoost, RewardKind::Continuous,
		 [] { return new SaveBoostReward(); }},
		// {"Wavedash", b.wavedash, RewardKind::Event,
		//  [] { return new WavedashReward(); }},
		{"Speed", b.speed, RewardKind::Continuous,
		 [] { return new SpeedReward(); }},
		{"Bump", b.bump, RewardKind::Event, [] { return new BumpReward(); }},
		{"Demo", b.demo, RewardKind::Event,
		 [] { return new ZeroSumReward(new DemoReward(), 0); }},
		{"Save", b.save, RewardKind::Event,
		 // [] { return new ZeroSumReward(new SaveReward(), 1, 0); }},
		 [] { return new SaveReward(); }},
		{"Awkward Contact", b.awkwardContact, RewardKind::Continuous,
		 [] { return new AwkwardContactPenalty(); }},
		{"Possession", b.possession, RewardKind::Continuous,
		 [] { return new PossessionReward(); }},
		{"Velocity: Ball to Goal (Own)", b.velBtG, RewardKind::Continuous,
		 [] { return new ConditionalVelocityBallToGoalReward(true); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
