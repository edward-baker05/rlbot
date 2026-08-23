#pragma once

#include <cstdint>
#include <filesystem>

namespace Hive {

// Where episodes are spawned from.
enum class SpectateSpawns {
	// Whatever TrainConfig::spawn selects, via the learner's own BuildSpawner().
	Training,
	// Kickoff to goal; easier to judge, but only ~8% of training resets.
	Kickoff,
};

// Watch a checkpoint play live in RocketSimVis. Deliberately NOT a learner.
struct SpectateConfig {
	// Exactly one of these; followRun picks up new checkpoints between episodes.
	std::filesystem::path model;
	std::filesystem::path followRun;

	// Training owns the GPU during a run.
	bool useGPU = false;

	// Deterministic shows intent without exploration noise; deployment uses it.
	bool deterministic = false;

	SpectateSpawns spawns = SpectateSpawns::Training;

	// Multiplier on real time. 1.0 is Rocket League speed.
	float timeScale = 1.f;

	// 0 runs until interrupted.
	int episodes = 0;

	int64_t seed = -1;
};

void RunSpectate(const SpectateConfig& cfg);

} // namespace Hive
