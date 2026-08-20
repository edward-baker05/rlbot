#pragma once

#include <cstdint>
#include <filesystem>

namespace Hive {

enum class SpectateSpawns {
	Training,

	Kickoff,
};

struct SpectateConfig {
	std::filesystem::path model;
	std::filesystem::path followRun;
	bool useGPU = false;
	bool deterministic = false;
	SpectateSpawns spawns = SpectateSpawns::Training;
	float timeScale = 1.f;
	int episodes = 0;
	int64_t seed = -1;
	int lookaheadTicks = 0;
	int candidates = 32;
};

void RunSpectate(const SpectateConfig& cfg);

}  // namespace Hive
