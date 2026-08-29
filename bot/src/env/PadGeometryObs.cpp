#include "PadGeometryObs.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>
#include <array>
#include <utility>

using namespace RLGC;

namespace Dash {

namespace {

bool IsBigPad(int index) {
	return std::find(std::begin(PadGeometryObs::BIG_PADS),
					 std::end(PadGeometryObs::BIG_PADS),
					 index) != std::end(PadGeometryObs::BIG_PADS);
}

} // namespace

FList PadGeometryObs::BuildObs(const Player &player, const GameState &state) {
	FList result = PredictiveObs::BuildObs(player, state);

	const bool inv = player.team == Team::ORANGE;
	const PhysState self = InvertPhys(player, inv);

	auto &pads = state.GetBoostPads(inv);
	auto &padTimers = state.GetBoostPadTimers(inv);

	// Same blend the inherited availability block uses: 1 when the pad is up,
	// approaching 1 as it comes back.
	auto availability = [&](int i) {
		return pads[i] ? 1.f : 1.f / (1.f + padTimers[i]);
	};

	auto addPad = [&](int i) {
		const Vec padPos = CommonValues::BOOST_LOCATIONS[i];
		result += self.rotMat.Dot(padPos - self.pos) * POS_COEF;
		result += availability(i);
	};

	for (int i : BIG_PADS)
		addPad(i);

	std::array<std::pair<float, int>, CommonValues::BOOST_LOCATIONS_AMOUNT>
		ranked = {};
	int smallCount = 0;
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		if (IsBigPad(i))
			continue;
		ranked[smallCount++] = {
			CommonValues::BOOST_LOCATIONS[i].DistSq(self.pos), i};
	}

	std::partial_sort(ranked.begin(), ranked.begin() + NEAREST_SMALL,
					  ranked.begin() + smallCount);

	for (int n = 0; n < NEAREST_SMALL; n++)
		addPad(ranked[n].second);

	return result;
}

} // namespace Dash
