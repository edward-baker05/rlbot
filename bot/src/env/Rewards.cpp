// Included so the imported community reward pack is compiled on every build
// rather than rotting until the day something first tries to use it.
#include "Rewards.h"
#include "CommunityRewards.h"

using namespace RLGC;

namespace Dash {

using namespace Community;

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;

	return {
		{"Goal", b.goal, RewardKind::Event,
		 [] { return new GoalReward(-0.8, true); }},

		{"Strong Touch", b.strongTouch, RewardKind::Event,
		 [] { return new ZeroSumReward(new DirectionalTouchReward(), 1); }},

		{"Air Touch", b.airTouch, RewardKind::Event,
		 [=] {
			 return new ImprovedAirTouchReward(cfg.aerial.minBallHeight,
											   cfg.aerial.maxBallHeight);
		 }},

		{"Dribble Flick", b.dribbleFlick, RewardKind::Continuous,
		 [] { return new ZeroSumReward(new DribbleFlickReward(), 0, 1.1f); }},

		{"On Target", b.onTarget, RewardKind::Event,
		 [] { return new ShotOnTargetReward(); }},

		{"Boost Pickup", b.pickupBoost, RewardKind::Event,
		 [] { return new ZeroSumReward(new PickupBoostReward(), 1); }},

		{"Save Boost", b.saveBoost, RewardKind::Continuous,
		 [] { return new Community::SaveBoostReward(); }},

		{"Speed", b.speed, RewardKind::Continuous,
		 [] { return new VelocityReward(); }},

		{"Bump", b.bump, RewardKind::Event,
		 [] { return new ZeroSumReward(new BumpReward(), 0, 1); }},

		{"Demo", b.demo, RewardKind::Event,
		 [] { return new ZeroSumReward(new DemoReward(), 1); }},

		{"Save", b.save, RewardKind::Event,
		 [] { return new ImprovedSaveReward(); }},

		{"Awkward Contact", b.awkwardContact, RewardKind::Continuous,
		 [] { return new AwkwardContactPenalty(); }},

		{"Possession", b.possession, RewardKind::Continuous,
		 [] { return new PossessionReward(); }},

		{"Own Goal Threat", b.ballToOwnGoal, RewardKind::Continuous,
		 [] { return new OwnGoalThreatPunishment(); }},

		{"Kickoff Distance", b.kickoff, RewardKind::Continuous,
		 [] { return new MillennialKickoffReward(); }},

		{"Goalside", b.goalside, RewardKind::Continuous,
		 [] { return new GoalsidePunishment(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
