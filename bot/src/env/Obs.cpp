#include "Obs.h"

#include <RLGymCPP/Gamestates/GameState.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

std::unique_ptr<DefaultObsPadded> MakeObsBuilder(int maxPlayersPerTeam) {
	if (maxPlayersPerTeam < 1)
		throw std::runtime_error("MakeObsBuilder(): maxPlayersPerTeam must be >= 1");

	return std::make_unique<DefaultObsPadded>(maxPlayersPerTeam);
}

int ProbeObsSize(int maxPlayersPerTeam) {
	// Build a full-size arena so the padding is exercised at its maximum.
	// A smaller arena would give the same answer (padding is what makes the
	// width fixed) but building the biggest case also validates that
	// maxPlayersPerTeam is actually large enough for the cars we intend to add.
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < maxPlayersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	int size = 0;
	try {
		GameState state = GameState(arena);
		auto obsBuilder = MakeObsBuilder(maxPlayersPerTeam);
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

} // namespace Hive
