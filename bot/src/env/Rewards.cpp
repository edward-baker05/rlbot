#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

float TouchHeightReward::GetReward(const Player& player, const GameState& state, bool isFinal) {
	if (!player.ballTouchedStep)
		return 0.f;

	const float height = state.ball.pos.z - CommonValues::BALL_RADIUS;
	if (height <= 0.f)
		return 0.f;

	return RS_MIN(1.f, height / maxHeight);
}

// ---------------------------------------------------------------------------
// Strong touch thresholds
// ---------------------------------------------------------------------------
// StrongTouchReward pays min(1, |delta ball velocity| / max), and zero below
// min. Those two numbers decide what "hitting the ball" means to the policy.
//
// MIN low (5 km/h): the reward has to exist for the first clumsy contacts, or a
// policy that has never touched the ball gets no gradient towards touching it.
// It is not zero, though -- a nonzero floor is exactly what stops a bot from
// resting against the ball and collecting for contact it did not earn.
//
// MAX moderate (100 km/h): the point of saturation should be a solid strike,
// not a world-class one. Set it too high and every early touch is worth
// approximately nothing, which flattens the gradient in the range the policy
// actually operates in.
static constexpr float TOUCH_MIN_KPH = 5.f;
static constexpr float TOUCH_MAX_KPH = 100.f;

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	const RewardWeights& w = cfg.rewards;

	if (cfg.rewardPhase != RewardPhase::Foundations) {
		throw std::runtime_error(
			"BuildGeneralRewards(): only RewardPhase::Foundations is designed. "
			"See docs/rewards.md -- later phases must be derived from the run "
			"that precedes them, not guessed at.");
	}

	return {
		// --- Reach the ball -------------------------------------------------
		// TELESCOPING. Exactly -d(distance to ball)/dt, so it cannot be farmed:
		// any path that returns to where it started sums to zero. Not zero-sum,
		// because closing on the ball is an individual act.
		{new VelocityPlayerToBallReward(), w.velPlayerToBall},

		// --- Strike it ------------------------------------------------------
		// IMPULSE. Requires the ball's velocity to actually change, so leaning
		// on the ball pays nothing however long you do it. This is the primary
		// outcome signal of the whole phase.
		//
		// Team spirit 0.2: in 2s and 3s a touch is mostly the work of the car
		// that made it, but a little sharing discourages all three cars
		// converging on the same ball.
		{new ZeroSumReward(new StrongTouchReward(TOUCH_MIN_KPH, TOUCH_MAX_KPH), 0.2f),
		 w.strongTouch},

		// --- Aim it ---------------------------------------------------------
		// Ball velocity towards the opponent goal. Zero-sum with slightly
		// higher spirit than touches: which way the ball is travelling is a
		// team-level outcome, not one car's.
		{new ZeroSumReward(new VelocityBallToGoalReward(), 0.3f), w.velBallToGoal},

		// --- Score it -------------------------------------------------------
		// TERMINAL. GoalReward is already zero-sum internally (it pays
		// concedeScale to the conceding team), so it must not be wrapped again.
		{new GoalReward(), w.goal},

		// --- Boost ----------------------------------------------------------
		// BOUNDED. A sqrt-delta: an empty-to-full tank is worth 1.0 in total,
		// and topping up from nearly full is worth almost nothing, so circling
		// pads is self-limiting. Individual, so not zero-sum.
		{new PickupBoostReward(), w.pickupBoost},
	};
}

std::vector<WeightedReward> BuildKickoffRewards(const KickoffRewardWeights& w) {
	return {
		// Get there first. Dense, and the episode is far too short for a sparse
		// signal to carry it alone.
		{new VelocityPlayerToBallReward(), w.velPlayerToBall},

		// Win the contact. Team spirit 0.0: only one car realistically takes a
		// kickoff, and sharing the credit would pay the cheating cars for the
		// taker's work.
		{new ZeroSumReward(new StrongTouchReward(TOUCH_MIN_KPH, TOUCH_MAX_KPH), 0.f),
		 w.strongTouch},

		// Send it the right way. Only accrues because FirstTouchCondition holds
		// the episode open briefly past first contact -- without that grace
		// window the episode ends on the touch step and this term would be
		// almost pure noise.
		{new ZeroSumReward(new VelocityBallToGoalReward(), 0.f), w.velBallToGoal},

		// Occasionally a kickoff goes straight in.
		{new GoalReward(), w.goal},
	};
}

} // namespace Hive
