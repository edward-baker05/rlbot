#include "Obs.h"
#include "AdvancedObsPadded.h"
#include "PadGeometryObs.h"
#include "PredictiveObs.h"
#include "RelativeObs.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <stdexcept>

using namespace RLGC;

namespace Dash {

std::unique_ptr<ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                           ObsMode mode) {
	if (maxPlayersPerTeam < 1)
		throw std::runtime_error("MakeObsBuilder(): maxPlayersPerTeam must be >= 1");

	switch (mode) {
	case ObsMode::Default:
		return std::make_unique<DefaultObsPadded>(maxPlayersPerTeam);
	case ObsMode::Advanced:
		return std::make_unique<AdvancedObsPadded>(maxPlayersPerTeam);
	case ObsMode::Relative:
		return std::make_unique<RelativeObs>(maxPlayersPerTeam);
	case ObsMode::Predictive:
		return std::make_unique<PredictiveObs>(maxPlayersPerTeam);
	case ObsMode::PadGeometry:
		return std::make_unique<PadGeometryObs>(maxPlayersPerTeam);
	default:
		throw std::runtime_error("MakeObsBuilder(): unknown ObsMode");
	}
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

const char* ObsModeName(ObsMode mode) {
	switch (mode) {
	case ObsMode::Default:    return "Default";
	case ObsMode::Advanced:   return "Advanced";
	case ObsMode::Relative:   return "Relative";
	case ObsMode::Predictive: return "Predictive";
	case ObsMode::PadGeometry: return "PadGeometry";
	}
	return "Unknown";
}

}  // namespace Dash
