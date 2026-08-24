#include "AdvancedObsPadded.h"

#include <RLGymCPP/Gamestates/StateUtil.h>

#include <algorithm>
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

	std::vector<FList> teammates = {}, opponents = {};

	for (auto &otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		FList playerObs = {};
		AddPlayerToObs(playerObs, otherPlayer, inv, ball);
		((otherPlayer.team == player.team) ? teammates : opponents)
			.push_back(playerObs);
	}

	if (static_cast<int>(teammates.size()) > maxPlayers - 1)
		RG_ERR_CLOSE(
			"AdvancedObsPadded: Too many teammates for Obs, maximum is "
			<< (maxPlayers - 1));

	if (static_cast<int>(opponents.size()) > maxPlayers)
		RG_ERR_CLOSE(
			"AdvancedObsPadded: Too many opponents for Obs, maximum is "
			<< maxPlayers);

	for (int i = 0; i < 2; i++) {
		auto &playerList = i ? teammates : opponents;
		int targetCount = i ? maxPlayers - 1 : maxPlayers;

		while (static_cast<int>(playerList.size()) < targetCount) {
			FList pad = FList(playerObsSize);
			playerList.push_back(pad);
		}
	}

	std::shuffle(teammates.begin(), teammates.end(), ::Math::GetRandEngine());
	std::shuffle(opponents.begin(), opponents.end(), ::Math::GetRandEngine());

	for (auto &teammate : teammates)
		result += teammate;
	for (auto &opponent : opponents)
		result += opponent;

	return result;
}

} // namespace Dash
