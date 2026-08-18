#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

float TouchHeightReward::GetReward(const Player& player, const GameState& state,
                                   bool isFinal) {
	if (!player.ballTouchedStep)
		return 0.f;

	const float height = state.ball.pos.z - CommonValues::BALL_RADIUS;
	if (height <= 0.f)
		return 0.f;

	return RS_MIN(1.f, height / maxHeight);
}

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardWeights& w = cfg.rewards;

	if (cfg.rewardPhase != RewardPhase::Foundations &&
	    cfg.rewardPhase != RewardPhase::Aerial) {
		throw std::runtime_error(
			"GeneralRewardSpecs(): only RewardPhase::Foundations and ::Aerial are "
			"designed. Later phases are derived from the telemetry of the run "
			"before them.");
	}

	// Aerial differs from Foundations in exactly two places, both marked below:
	// TouchHeightReward switches on, and the flat Grounded bonus switches off.
	// The GroundedReward gates on VelPlayerToBall/FaceBall stay in both phases
	// as the anti-farm anchor; the bot is paid for arriving (TouchHeight,
	// StrongTouch), not for approaching while airborne.
	const bool aerial = (cfg.rewardPhase == RewardPhase::Aerial);

	return {
		// Restored to the p1air form. p4pbrs replaced this with a car-to-ball
		// potential and touch rate FELL 0.0011 -> 0.0007: the potential telescoped
		// to zero over chase-hit-chase, leaving outcomes alone to bootstrap at a
		// touch rate too low to do it. The farmability is what made it teach.
		{"VelPlayerToBall", w.velPlayerToBall,
		 []() -> Reward* { return new GroundedReward(new VelocityPlayerToBallReward()); }},
		{"Touch", w.touch, [] { return new TouchBallReward(); }},
		{"StrongTouch", w.strongTouch,
		 [sharpness = w.aimSharpness]() -> Reward* { return new ZeroSumReward(new AimedStrongTouchReward(sharpness), 0.2f); }},
		// Potential-based on ball->their-goal: same quantity this slot always
		// measured, but farm-proof, and it pays for striking the ball goalward
		// instead of for passive ball motion. Not ZeroSum-wrapped -- see class.
		{"VelBallToGoal", w.velBallToGoal,
		 []() -> Reward* { return new BallGoalProgressReward(); }},
		{"Goal", w.goal, [] { return new GoalReward(); }},
		{"PickupBoost", w.pickupBoost, [] { return new PickupBoostReward(); }},
		{"FaceBall", w.faceBall,
		 []() -> Reward* { return new GroundedReward(new FaceBallReward()); }},

		// Not zero-summed or gated: pure car-control, unaffected by the opponent.
		{"AirRecovery", w.airRecovery, [] { return new AirRecoveryReward(); }},

		// Zero-weight specs stay in the list (rather than omitted) so RewardShare
		// metric indices stay aligned between Foundations and Aerial runs.
		{"Grounded", aerial ? 0.f : w.grounded, [] { return new GroundedBonusReward(); }},
		{"TouchHeight", aerial ? w.touchHeight : 0.f,
		 [] { return new TouchHeightReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
