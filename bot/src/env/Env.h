#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Hive {

// Build the environment for game `index`. This is the function handed to
// GigaLearn's Learner; it is called once per game at startup.
RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

} // namespace Hive
