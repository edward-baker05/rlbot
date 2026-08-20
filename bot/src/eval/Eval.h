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
	int lookaheadBlue = 0;
	int lookaheadOrange = 0;
	int candidates = 32;
};

struct EvalResult {
	int blueWins = 0, orangeWins = 0, draws = 0;
	int blueGoals = 0, orangeGoals = 0;
};

EvalResult RunEval(const EvalConfig& cfg);

}  // namespace Hive
