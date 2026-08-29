#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <memory>

namespace Dash {

enum class ObsMode {
	Default,
	Advanced,
	Relative,
	Predictive,
	PadGeometry,
};

std::unique_ptr<RLGC::ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                                 ObsMode mode);

int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode);

// Name used in CONFIG.json and console output. A two-way ternary used to
// do this job and would silently mislabel any third mode.
const char* ObsModeName(ObsMode mode);

}  // namespace Dash
