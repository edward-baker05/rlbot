#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Dash {

RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

RLGC::StateSetter* BuildSpawner(const TrainConfig& cfg);

}  // namespace Dash
