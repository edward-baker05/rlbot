#pragma once

#include <cstdint>
#include <filesystem>

namespace Hive {

// Where episodes are spawned from.
enum class SpectateSpawns {
	// The training curriculum: the same scenario mix and weights CreateEnv
	// uses, so what you watch is what the policy is practising.
	Curriculum,
	// Kickoff to goal, like a real match. Easier to judge as play, but not
	// representative -- kickoffs are only ~8% of training resets.
	Kickoff,
};

// Watch a checkpoint play, live, in RocketSimVis.
//
// This is deliberately NOT a learner. `train --render` builds a full Learner
// and collects experience; running it alongside a real run wastes CPU and
// competes for the run's checkpoint folder. This loads a checkpoint, plays it
// against itself in one arena at wall-clock speed, and streams the gamestate.
// It never writes anything.
struct SpectateConfig {
	// Exactly one of these. `model` is a specific checkpoint folder; `followRun`
	// is a run folder (checkpoints/main-<label>) whose newest checkpoint is
	// picked up between episodes, so a live run can be watched improving.
	std::filesystem::path model;
	std::filesystem::path followRun;

	// Training owns the GPU during a run, and one car at 120 ticks/sec does not
	// need it.
	bool useGPU = false;

	// Training samples from the action distribution; deterministic shows the
	// policy's intent without exploration noise, and is what deployment uses.
	bool deterministic = false;

	SpectateSpawns spawns = SpectateSpawns::Curriculum;

	// Multiplier on real time. 1.0 is Rocket League speed.
	float timeScale = 1.f;

	// 0 runs until interrupted.
	int episodes = 0;

	int64_t seed = -1;
};

void RunSpectate(const SpectateConfig& cfg);

} // namespace Hive
