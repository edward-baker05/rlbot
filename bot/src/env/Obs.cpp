#include "Obs.h"

#include "RelativeObs.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <atomic>
#include <stdexcept>

using namespace RLGC;

namespace Hive {

namespace {
std::atomic<uint64_t> g_ObsChecked{0};
std::atomic<uint64_t> g_ObsNonFinite{0};
} // namespace

void NoteObsHealth(uint64_t checked, uint64_t nonFinite) {
	g_ObsChecked.fetch_add(checked, std::memory_order_relaxed);
	if (nonFinite)
		g_ObsNonFinite.fetch_add(nonFinite, std::memory_order_relaxed);
}

ObsHealth ConsumeObsHealth() {
	ObsHealth out;
	out.checked = g_ObsChecked.exchange(0, std::memory_order_relaxed);
	out.nonFinite = g_ObsNonFinite.exchange(0, std::memory_order_relaxed);
	return out;
}


std::unique_ptr<ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                           ObsMode mode) {
	if (maxPlayersPerTeam < 1)
		throw std::runtime_error("MakeObsBuilder(): maxPlayersPerTeam must be >= 1");

	if (mode == ObsMode::Default)
		return std::make_unique<DefaultObsPadded>(maxPlayersPerTeam);

	return std::make_unique<RelativeObs>(maxPlayersPerTeam);
}

int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode) {
	// Build a full-size arena so padding is exercised at its maximum, which
	// also validates maxPlayersPerTeam is large enough for the intended cars.
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

} // namespace Hive
