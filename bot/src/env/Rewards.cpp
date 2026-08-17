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

static constexpr float TOUCH_MIN_KPH = 5.f;
static constexpr float TOUCH_MAX_KPH = 100.f;

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardWeights& w = cfg.rewards;

	if (cfg.rewardPhase != RewardPhase::Foundations) {
		throw std::runtime_error(
			"GeneralRewardSpecs(): only RewardPhase::Foundations is designed. "
			"Later phases are derived from the telemetry of the run before them.");
	}

	return {
		{"VelPlayerToBall", w.velPlayerToBall,
		 []() -> Reward* { return new GroundedReward(new VelocityPlayerToBallReward()); }},
		{"Touch", w.touch, [] { return new TouchBallReward(); }},
		{"StrongTouch", w.strongTouch,
		 []() -> Reward* { return new ZeroSumReward(new StrongTouchReward(TOUCH_MIN_KPH, TOUCH_MAX_KPH), 0.2f); }},
		{"VelBallToGoal", w.velBallToGoal,
		 []() -> Reward* { return new ZeroSumReward(new VelocityBallToGoalReward(), 0.3f); }},
		{"Goal", w.goal, [] { return new GoalReward(); }},
		{"PickupBoost", w.pickupBoost, [] { return new PickupBoostReward(); }},
		{"FaceBall", w.faceBall,
		 []() -> Reward* { return new GroundedReward(new FaceBallReward()); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
