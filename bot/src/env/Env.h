#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Dash {

RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

// includeAerialDrill = false drops the aerial spawn slice, leaving the football
// distribution. NectoBench uses that so retuning the drill does not move what
// the benchmark measures.
RLGC::StateSetter* BuildSpawner(const TrainConfig& cfg,
								bool includeAerialDrill = true);

}  // namespace Dash
