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
	return {
		// THE UNIT: a maximal goal-directed strike. Signed, so putting the ball
		// toward your own net costs. Touch-gated, so it measures only the ball
		// motion this car caused -- the continuous VelocityBallToGoal form is
		// known-bad here (p1probe-b: 67% of reward mass as passive ball noise).
		{"TouchGoalAccel", b.touchGoalAccel,
		 [e = b.touchAccelExponent] { return new TouchGoalAccelReward(e); }},

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

		// Per pickup, on the boost INCREMENT: encourages collecting it. Small
		// pads carry a guaranteed baseline floor so routing over them stays attractive.
		{"PickupBoost", b.pickupBoost, [] { return new TieredPickupBoostReward(); }},

		// Forward flip closing acceleration towards the ball. Boost-neutral,
		// supporting both ground flips and speed-flips to traverse and conserve boost.
		{"FlipSpeed", b.flipSpeed, [] { return new FlipSpeedReward(); }},

		// Pays for touching the ball high AFTER real air time. The min() makes a
		// wall shot worth exactly zero, which is the farm this bot already runs.
		{"AirTouch", b.airTouch,
		 [e = b.airTouchHeightExponent] { return new AirTouchReward(e); }},

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
