#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <cstdint>
#include <memory>

namespace Hive {

enum class ObsMode {
	Relative,
	Default,
};

std::unique_ptr<RLGC::ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                                 ObsMode mode);

struct ObsHealth {
	uint64_t checked = 0;
	uint64_t nonFinite = 0;
};

ObsHealth ConsumeObsHealth();

int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode);

}  // namespace Hive
