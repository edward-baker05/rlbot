#include "NectoDriver.h"

using namespace RLGC;

namespace Dash {

NectoDriver::NectoDriver(const std::filesystem::path &modelPath, float beta,
						 int64_t seed)
	: policy(modelPath, beta, seed) {}

void NectoDriver::BuildMask(EnvSet *envSet, std::vector<uint8_t> &outMask) {
	outMask.assign(envSet->state.numPlayers, 0);
	numControlledCars = 0;

	for (size_t a = 0; a < envSet->arenas.size(); a++) {
		auto *arenaState = static_cast<NectoArenaState *>(envSet->userInfos[a]);
		if (!arenaState || !arenaState->active)
			continue;

		const GameState &gs = envSet->state.gameStates[a];
		const int base = envSet->state.arenaPlayerStartIdx[a];
		for (size_t p = 0; p < gs.players.size(); p++) {
			if (gs.players[p].team != arenaState->nectoTeam)
				continue;
			outMask[base + p] = 1;
			numControlledCars++;
		}
	}
}

void NectoDriver::Step(EnvSet *envSet) {
	requests.clear();
	slots.clear();

	for (size_t a = 0; a < envSet->arenas.size(); a++) {
		auto *arenaState = static_cast<NectoArenaState *>(envSet->userInfos[a]);
		if (!arenaState || !arenaState->active)
			continue;

		const GameState &gs = envSet->state.gameStates[a];
		if (arenaState->pending.size() < gs.players.size())
			arenaState->pending.resize(gs.players.size());

		for (size_t p = 0; p < gs.players.size(); p++) {
			if (gs.players[p].team != arenaState->nectoTeam)
				continue;

			// prevAction comes straight from the game state: StepSecondHalf
			// feeds the applied controls back into it, and ResetArena rebuilds
			// the state from scratch, so it is already zeroed at the start of
			// every episode. Tracking a second copy here would only be another
			// thing to get out of sync.
			requests.push_back(
				{&gs, static_cast<int>(p), gs.players[p].prevAction});
			slots.push_back({arenaState, static_cast<int>(p)});
		}
	}

	if (requests.empty())
		return;

	policy.InferBatch(requests, actions);

	for (size_t i = 0; i < slots.size(); i++)
		slots[i].arena->pending[slots[i].playerIdx] = actions[i];
}

} // namespace Dash
