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
	// Four terms, exactly the early-stage stack from Zealan's RLGym-PPO-Guide
	// (making_a_good_bot.md), in the guide's own proportions. Nothing here is
	// this project's invention, deliberately -- see
	// docs/superpowers/specs/2026-08-18-known-good-baseline-design.md D1.
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
		// The unit. Per STEP of contact, not per rising edge -- the guide's
		// EventReward(touch=1) fires every step `ball_touched` is set, and this
		// is a reproduction. `Touch/Edge Rate` stays instrumented alongside
		// `Player/Ball Touch Ratio` so the carrying gap remains visible; if it
		// opens up, a rising-edge form is the fix and phase B is where it goes.
		{"Touch", b.touch, [] { return new TouchBallReward(); }},

		{"SpeedToBall", RateWeight(b.speedToBall),
		 [] { return new SpeedToBallReward(); }},

		// SIGNED. Facing away from the ball returns a negative value and is
		// punished. This is RLGymCPP's FaceBallReward unmodified, matching
		// rlgym, and it is the only term in the stack that charges for
		// anything. See the pairing note on SpeedToBallReward.
		{"FaceBall", RateWeight(b.faceBall),
		 [] { return new FaceBallReward(); }},

		// Pays for being airborne. Deliberately ported despite this project's
		// bot already spending 93% of its life in the air, because at 3% of the
		// dense budget it cannot plausibly cause that -- and if air time
		// survives a reward that actively pays for air, the cause is the action
		// mask (Actions.h) rather than the reward function. Isolating that is
		// what the reproduction is for.
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
