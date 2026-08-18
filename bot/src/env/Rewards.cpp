#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Hive {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardBudget& b = cfg.rewards;

	// Budgets become per-step weights HERE and nowhere else. That single
	// conversion site is the whole point of the budget system: p1air's
	// do-nothing attractor was a per-step float whose episode integral nobody
	// computed.
	//
	// SIGN CONVENTION: penalty classes return negative values and carry
	// POSITIVE weights, matching upstream BumpedPenalty/DemoedPenalty. A
	// negative weight here would double-negate a penalty into a reward, and
	// nothing downstream would reveal it.
	//
	// Nothing except Goal is ZeroSum-wrapped. Zealan's rule of thumb is that a
	// reward should be zero-sum only if it is useful for the OPPONENT to
	// prevent it; approach and orientation are movement tuning, and wrapping
	// them buys opponent variance for no competitive meaning. GoalReward
	// already carries the entire adversarial structure.
	//
	// Five terms, down from eight. p6budget ran eight and the guide's
	// troubleshooting advice ("reduce or remove tuning rewards") plus its
	// early-stage stack both point the other way: at this stage the bot needs
	// to reach the ball, and every term that is not about reaching the ball is
	// noise competing with the one that is.
	return {
		// The unit. Already zero-sum: +1 scored, -1 conceded.
		{"Goal", RewardBudget::GOAL, [] { return new GoalReward(); }},

		{"Touch", b.touch, [] { return new TouchEdgeReward(); }},

		{"SpeedToBall", RateWeight(b.speedToBall),
		 [] { return new SpeedToBallReward(); }},

		{"FaceBall", RateWeight(b.faceBall),
		 [] { return new FaceBallRectifiedReward(); }},

		{"WrongSurface", PerSecondWeight(b.wrongSurface),
		 [] { return new WrongSurfaceReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
