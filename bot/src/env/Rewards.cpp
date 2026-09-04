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

		// {"Air Touch", b.airTouch, RewardKind::Event,
		//  [=] {
		// 	 return new ImprovedAirTouchReward(cfg.aerial.minBallHeight,
		// 									   cfg.aerial.maxBallHeight);
		//  }},

		{"Aerial Distance", b.aerialDistance, RewardKind::Event,
		 [] { return new ZeroSumReward(new AerialDistanceReward(), 1, 1); }},

		// TODO: ZeroSumReward() following 2 rewards
		{"Dribble Flick", b.dribbleFlick, RewardKind::Continuous,
		 [] { return new Community::DribbleFlickReward(); }},

		{"Flick", b.flick, RewardKind::Event, [] { return new FlickReward(); }},

		{"Air Velocity to Ball", b.airVtB, RewardKind::Continuous,
		 [=] {
			 return new AirVelToBallReward(cfg.aerial.minBallHeight,
										   cfg.aerial.maxBallHeight);
		 }},

		// This is causing issues with suboptimal direction in the air; the bot
		// prefers to face where the ball is right now, which for far/fast balls
		// leads to chasing and incredibly inefficient boost usage, making it
		// either late to the ball or miss it entirely. Important to fix.
		{"Air Face Ball", b.airFaceBall, RewardKind::Continuous,
		 [=] {
			 return new AirFaceBallReward(cfg.aerial.minBallHeight,
										  cfg.aerial.maxBallHeight);
		 }},

		// {"Air Launch", b.airLaunch, RewardKind::Continuous,
		// [=] { return new AirLaunchReward(cfg.aerial.minBallHeight); }},

		{"Boost Pickup", b.pickupBoost, RewardKind::Event,
		 [] {
			 return new ZeroSumReward(new PadAwarePickupBoostReward(), 1, 0.5);
		 }},

		{"Save Boost", b.saveBoost, RewardKind::Continuous,
		 [] { return new SaveBoostReward(); }},

		{"Speed", b.speed, RewardKind::Continuous,
		 [] { return new SpeedReward(); }},

		// Temporarily removed until I have a better reward; this is an
		// important reward to get working, alongside wall/chaindashing
		// {"Wavedash", b.wavedash, RewardKind::Event, [] { return new
		// WavedashReward(); }},

		{"Bump", b.bump, RewardKind::Event, [] { return new BumpReward(); }},

		{"Demo", b.demo, RewardKind::Event,
		 [] { return new ZeroSumReward(new DribbleDemoReward(), 1); }},

		{"Save", b.save, RewardKind::Event,
		 [] { return new ZeroSumReward(new SaveReward(), 1, 1); }},

		{"Awkward Contact", b.awkwardContact, RewardKind::Continuous,
		 [] { return new AwkwardContactPenalty(); }},

		// {"Possession", b.possession, RewardKind::Continuous,
		//  [] { return new PossessionReward(); }},
		//
		// {"Velocity: Ball to Goal (Own)", b.velBtG, RewardKind::Continuous,
		//  [] { return new ConditionalVelocityBallToGoalReward(true); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
	std::vector<WeightedReward> out;
	for (auto &spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Dash
