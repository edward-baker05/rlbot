#include "AdvancedObsPadded.h"

#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>
#include <utility>
#include <vector>

using namespace RLGC;

namespace Dash {

FList AdvancedObsPadded::BuildObs(const Player &player,
								  const GameState &state) {
	FList result = {};

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto &pads = state.GetBoostPads(inv);
	auto &padTimers = state.GetBoostPadTimers(inv);

	result += ball.pos * POS_COEF;
	result += ball.vel * VEL_COEF;
	result += ball.angVel * ANG_VEL_COEF;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		result += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		if (pads[i]) {
			result += 1.f;
		} else {
			result += 1.f / (1.f + padTimers[i]);
		}
	}

	FList selfObs = {};
	AddPlayerToObs(selfObs, player, inv, ball);
	result += selfObs;
	int playerObsSize = static_cast<int>(selfObs.size());

	// Distance to the ball, so that slot order is a property of the state
	// rather than of the draw. This used to be a shuffle, on the usual
	// permutation-invariance reasoning -- but zero padding ran first, so at 1v1
	// the single real opponent landed in a random one of `maxPlayers` slots on
	// every step.
	std::vector<std::pair<float, FList>> teammates = {}, opponents = {};

	for (auto &otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		FList playerObs = {};
		AddPlayerToObs(playerObs, otherPlayer, inv, ball);
		((otherPlayer.team == player.team) ? teammates : opponents)
			.emplace_back(otherPlayer.pos.DistSq(state.ball.pos),
						  std::move(playerObs));
	}

	if (static_cast<int>(teammates.size()) > maxPlayers - 1)
		RG_ERR_CLOSE(
			"AdvancedObsPadded: Too many teammates for Obs, maximum is "
			<< (maxPlayers - 1));

	if (static_cast<int>(opponents.size()) > maxPlayers)
		RG_ERR_CLOSE(
			"AdvancedObsPadded: Too many opponents for Obs, maximum is "
			<< maxPlayers);

	auto emitSorted = [&](std::vector<std::pair<float, FList>> &playerList,
						  int targetCount) {
		std::sort(playerList.begin(), playerList.end(),
				  [](const std::pair<float, FList> &a,
					 const std::pair<float, FList> &b) {
					  return a.first < b.first;
				  });

		for (auto &entry : playerList)
			result += entry.second;
		for (int i = static_cast<int>(playerList.size()); i < targetCount; i++)
			result += FList(playerObsSize);
	};

	emitSorted(teammates, maxPlayers - 1);
	emitSorted(opponents, maxPlayers);

	return result;
}

} // namespace Dash
