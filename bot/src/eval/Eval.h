#pragma once

#include <cstdint>
#include <filesystem>

namespace Hive {

// Headless checkpoint-vs-checkpoint matches in RocketSim. This is the
// frozen-reference-pool tool: pit any two checkpoints (current vs a gate
// checkpoint, run A vs run B) without a learner or a game client.
struct EvalConfig {
	std::filesystem::path blueModel;
	std::filesystem::path orangeModel;
	int games = 20;
	float maxSeconds = 300.f;   // per game, sim time
	bool useGPU = true;
	int64_t seed = -1;
};

struct EvalResult {
	int blueWins = 0, orangeWins = 0, draws = 0;
	int blueGoals = 0, orangeGoals = 0;
};

EvalResult RunEval(const EvalConfig& cfg);

} // namespace Hive
