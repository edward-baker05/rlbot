#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

using namespace RLGC;

namespace Hive {

float TouchHeightReward::GetReward(const Player& player, const GameState& state, bool isFinal) {
	if (!player.ballTouchedStep)
		return 0.f;

	// Measure from the ball's resting height so a touch on the ground scores
	// zero rather than a small constant. Otherwise the policy can farm this
	// reward by dribbling, which is the opposite of the intent.
	const float height = state.ball.pos.z - CommonValues::BALL_RADIUS;
	if (height <= 0.f)
		return 0.f;

	return RS_MIN(1.f, height / maxHeight);
}

std::vector<WeightedReward> BuildGeneralRewards(const RewardWeights& w) {
	return {
		// --- Movement and orientation --------------------------------------
		// Small continuous rewards. They give the policy something to climb
		// early on, before it can reliably touch the ball at all.
		{new AirReward(), w.air},
		{new FaceBallReward(), w.faceBall},
		{new VelocityPlayerToBallReward(), w.velPlayerToBall},

		// --- Striking the ball ----------------------------------------------
		// Zero-sum with modest team spirit: a good touch by any teammate is
		// worth something to all of them, which discourages three cars
		// converging on the same ball.
		{new ZeroSumReward(new StrongTouchReward(20, 130), 0.3f), w.strongTouch},
		{new TouchHeightReward(), w.touchHeight},

		// --- Moving the ball towards the goal -------------------------------
		{new ZeroSumReward(new VelocityBallToGoalReward(), 0.3f), w.velBallToGoal},

		// --- Boost economy ---------------------------------------------------
		// Not zero-sum: collecting boost is individually useful and sharing the
		// credit would just add noise.
		{new PickupBoostReward(), w.pickupBoost},
		{new SaveBoostReward(), w.saveBoost},

		// --- Contact ---------------------------------------------------------
		// Higher team spirit here. Demos are a team play -- the value is in the
		// space it creates for a teammate, not in the demo itself.
		{new ZeroSumReward(new BumpReward(), 0.5f), w.bump},
		{new ZeroSumReward(new DemoReward(), 0.5f), w.demo},

		// --- Terminal ---------------------------------------------------------
		// GoalReward is already zero-sum internally; do not wrap it again.
		{new GoalReward(), w.goal},
	};
}

std::vector<WeightedReward> BuildKickoffRewards(const KickoffRewardWeights& w) {
	return {
		// Get there first.
		{new VelocityPlayerToBallReward(), w.velPlayerToBall},

		// Win the contact. Zero-sum with no team spirit: on a kickoff only one
		// car realistically takes the ball, and sharing credit would reward
		// the cheating cars for the taker's work.
		{new ZeroSumReward(new StrongTouchReward(20, 130), 0.f), w.strongTouch},

		// Send it the right way.
		{new ZeroSumReward(new VelocityBallToGoalReward(), 0.f), w.velBallToGoal},

		// Do not burn every drop of boost getting there -- the general policy
		// inherits whatever is left the instant the kickoff ends.
		{new SaveBoostReward(), w.saveBoost},

		// Occasionally a kickoff goes straight in.
		{new GoalReward(), w.goal},
	};
}

} // namespace Hive
