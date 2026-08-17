#pragma once

#include "../Config.h"

namespace Hive {

// Run a training session to completion (or until you stop it).
// Blocks until the learner exits.
void RunTraining(const TrainConfig& cfg);

} // namespace Hive
