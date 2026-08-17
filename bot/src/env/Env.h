#pragma once

#include "../Config.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace Hive {

// Team sizes for one training game.
struct TeamSizes {
	int blue;
	int orange;
};

// Choose the team sizes for game `index`, following cfg.teamSizes.
//
// Sizes are assigned deterministically by index rather than randomly, so the
// mix across the whole run is exactly the configured ratio instead of only
// converging to it. With 128 games and a 30/30/30/10 split you get precisely
// the intended number of each, every run, which makes comparing runs honest.
TeamSizes PickTeamSizes(int index, const TrainConfig& cfg);

// Build the environment for game `index`. This is the function handed to
// GigaLearn's Learner; it is called once per game at startup.
RLGC::EnvCreateResult CreateEnv(int index, const TrainConfig& cfg);

} // namespace Hive
