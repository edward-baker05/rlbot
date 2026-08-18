#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Hive {

// Build the environment for game `index`. This is the function handed to
// GigaLearn's Learner; it is called once per game at startup.
RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

// The scenario mix training spawns from. Exposed so the spectator can show the
// same distribution the policy is actually practising.
RLGC::StateSetter* BuildGeneralCurriculum(const CurriculumWeights& weights);

} // namespace Hive
