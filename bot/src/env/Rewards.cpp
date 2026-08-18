#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Hive {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardBudget& b = cfg.rewards;

	// Budgets become per-step weights HERE and nowhere else. That single
	// conversion site is the whole point of the redesign: p1air's do-nothing
	// attractor was a per-step float whose episode integral nobody computed.
	//
	// SIGN CONVENTION: penalty classes return negative values and carry
	// POSITIVE weights, matching upstream BumpedPenalty/DemoedPenalty. A
	// negative weight here would double-negate a penalty into a reward, and
	// nothing downstream would reveal it.
	//
	// Nothing except Goal is ZeroSum-wrapped. These are car-control terms;
	// wrapping them would add opponent variance for no competitive meaning,
	// and GoalReward already carries the entire adversarial structure.
	return {
		// The unit. Already zero-sum: +1 scored, -1 conceded.
		{"Goal", RewardBudget::GOAL, [] { return new GoalReward(); }},

		{"Touch", b.touch, [] { return new TouchEdgeReward(); }},
		{"CleanLanding", b.cleanLanding, [] { return new CleanLandingReward(); }},
		{"WrongSurface", PerSecondWeight(b.wrongSurface),
		 [] { return new WrongSurfaceReward(); }},
		{"HarshSpeedLoss", b.harshSpeedLoss, [] { return new HarshSpeedLossReward(); }},

		{"Speed", RateWeight(b.speed), [] { return new SpeedSquaredReward(); }},
		// Upstream's FaceBallReward is already the full 3-D dot product; the
		// old stack gated it to grounded steps, which left the airborne policy
		// with no directional signal at all. Ungated here.
		{"FaceBall", RateWeight(b.faceBall), [] { return new FaceBallReward(); }},
		{"FaceBallAxis", RateWeight(b.faceBallAxis),
		 [] { return new FaceBallAxisReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
