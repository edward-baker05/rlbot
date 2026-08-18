#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Hive {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardBudget& b = cfg.rewards;

	// Budgets become per-step weights HERE and nowhere else. That single
	// conversion site is the point of the budget system: p1air's do-nothing
	// attractor was a per-step float whose episode integral nobody computed.
	//
	// The UNIT is one ball touch. A goal was the unit until p7approach, and it
	// could not be audited: goals arrive 0.116 times per episode, and 49% of
	// those are `Scenario/Defend` conceding rather than anyone scoring. A touch
	// occurs 0.16-2 times per episode and is read directly off
	// `Touch/Edge Rate` and `Player/Ball Touch Ratio`, so the ledger can be
	// checked against telemetry instead of reconstructed by hand.
	//
	// Eight terms. The early-stage stack from Zealan's RLGym-PPO-Guide
	// (making_a_good_bot.md) with its touch term taken to the guide's
	// MIDDLE-stage form, because p9rel hit the transition the guide describes:
	// "The default touch part of EventReward is not very good once your bot can
	// touch the ball. This is because ball touches can easily be farmed by
	// constantly pushing the ball." It was -- 13% of all steps in contact and
	// 74% of reward mass on a dribble.
	//
	// NOT PRESENT, and each absence is a decision:
	//
	//   * No Goal term. The guide: "Having these rewards before the bot is
	//     capable of actually hitting the ball just adds lots of noise to the
	//     overall reward and will slow learning." At a touch ratio of 0.001 the
	//     bot cannot cause a goal, so the term is pure variance.
	//   * No boost, no ball-to-goal, no tuning penalties. The guide's
	//     troubleshooting section says to reduce or remove tuning rewards;
	//     p7approach had `WrongSurface` holding 30% of reward mass.
	//   * Nothing is ZeroSum-wrapped. Zealan's rule is that a reward should be
	//     zero-sum only if it is useful for the OPPONENT to prevent it.
	//     Approach and orientation are movement tuning. With no Goal term there
	//     is no adversarial structure in the stack at all, which is correct for
	//     a bot that cannot yet reach the ball.
	return {
		// THE UNIT: a maximal goal-directed strike. Signed, so putting the ball
		// toward your own net costs. Touch-gated, so it measures only the ball
		// motion this car caused -- the continuous VelocityBallToGoal form is
		// known-bad here (p1probe-b: 67% of reward mass as passive ball noise).
		{"TouchGoalAccel", b.touchGoalAccel,
		 [] { return new TouchGoalAccelReward(); }},

		// The scoreboard. Already zero-sum: +1 scored, -1 conceded. Moderate on
		// purpose -- see the budget comment; a huge goal reward scales variance,
		// not signal, and this is also the only thing that ends an episode.
		{"Goal", b.goal, [] { return new GoalReward(); }},

		// Arriving at the ball, on the RISING EDGE so carrying pays once.
		{"TouchEdge", b.touchEdge, [] { return new TouchEdgeReward(); }},

		{"SpeedToBall", RateWeight(b.speedToBall),
		 [] { return new SpeedToBallReward(); }},

		// SIGNED. Facing away from the ball returns a negative value and is
		// punished. RLGymCPP's FaceBallReward unmodified, matching rlgym, and
		// the only term in the stack that charges for anything. See the pairing
		// note on SpeedToBallReward.
		{"FaceBall", RateWeight(b.faceBall),
		 [] { return new FaceBallReward(); }},

		// Pays for being airborne. Measured ~50x too small to cover what a jump
		// costs in traction and contact, and deliberately left that way: see
		// the note on RewardBudget::air.
		// Per step, on the boost LEVEL: discourages wasting it.
		{"SaveBoost", RateWeight(b.saveBoost), [] { return new SaveBoostReward(); }},

		// Per pickup, on the boost INCREMENT: encourages collecting it.
		{"PickupBoost", b.pickupBoost, [] { return new PickupBoostReward(); }},

		// Pays for touching the ball high AFTER real air time. The min() makes a
		// wall shot worth exactly zero, which is the farm this bot already runs.
		{"AirTouch", b.airTouch, [] { return new AirTouchReward(); }},

		// Pays for being airborne at all. Measured ~50x too small to cover what
		// a jump costs, and left that way on purpose: AirTouch is the term that
		// pays for air now, and it pays for PRODUCTIVE air.
		{"Air", RateWeight(b.air), [] { return new AirReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
