#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <memory>

namespace Hive {

enum class ObsMode {
	Default,
};

std::unique_ptr<RLGC::ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                                 ObsMode mode);

int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode);

}  // namespace Hive
