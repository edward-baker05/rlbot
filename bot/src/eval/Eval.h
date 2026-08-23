#pragma once

#include <cstdint>
#include <filesystem>

namespace Hive {

struct EvalConfig {
	std::filesystem::path blueModel;
	std::filesystem::path orangeModel;
	int games = 20;
	float maxSeconds = 300.f;
	bool useGPU = true;
	int64_t seed = -1;
	bool deterministic = false;
	bool randomSpawn = false;
};

struct EvalResult {
	int blueWins = 0, orangeWins = 0, draws = 0;
	int blueGoals = 0, orangeGoals = 0;
	int episodes = 0, stateGoals = 0;
	long long steps = 0, touchSteps = 0;
};

EvalResult RunEval(const EvalConfig& cfg);

}  // namespace Hive
