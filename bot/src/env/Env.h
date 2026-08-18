#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Hive {

// Build the environment for game `index`. This is the function handed to
// GigaLearn's Learner; it is called once per game at startup.
RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

// The spawn distribution training resets to. Exposed so the spectator shows the
// same distribution the policy is actually practising -- watching a different
// one is how you conclude a bot handles situations it has never seen.
RLGC::StateSetter* BuildSpawner(const TrainConfig& cfg);

// The scenario mix, when TrainConfig::spawn is Curriculum.
RLGC::StateSetter* BuildGeneralCurriculum(const CurriculumWeights& weights);

} // namespace Hive
