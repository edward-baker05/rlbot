#include "Obs.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

std::unique_ptr<ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                           ObsMode mode) {
	if (maxPlayersPerTeam < 1)
		throw std::runtime_error("MakeObsBuilder(): maxPlayersPerTeam must be >= 1");

	return std::make_unique<DefaultObsPadded>(maxPlayersPerTeam);
}

int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < maxPlayersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	int size = 0;
	try {
		GameState state = GameState(arena);
		auto obsBuilder = MakeObsBuilder(maxPlayersPerTeam, mode);
		size = static_cast<int>(obsBuilder->BuildObs(state.players[0], state).size());
	} catch (...) {
		delete arena;
		throw;
	}

	delete arena;

	if (size <= 0)
		throw std::runtime_error("ProbeObsSize(): obs builder produced an empty observation");

	return size;
}

}  // namespace Hive
