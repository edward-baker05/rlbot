#pragma once

#include <cstdint>
#include <filesystem>

namespace Dash {

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

	// Print the training reward stack's per-step breakdown, and take keyboard
	// control of the clock: space pauses/resumes, '.' advances one step, 'c'
	// runs without auto-pausing, 'b' dumps the recent step history, 's' dumps
	// episode totals, 'q' quits. Pausing simply stops streaming, which the
	// visualizer renders as a held frame.
	bool debugRewards = false;

	// Weighted contribution magnitude that auto-pauses play. Only event
	// rewards arm it; the continuous shaping rewards are displayed but would
	// trip on nearly every step. 0 disables auto-pause.
	float rewardPauseThreshold = 0.25f;

	// Begin paused rather than playing, for stepping a kickoff from step 0.
	bool rewardStartPaused = false;

	// Steps of breakdown history retained for the 'b' key.
	int rewardHistorySteps = 30;
};

void RunSpectate(const SpectateConfig& cfg);

} // namespace Dash
