#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <array>
#include <cstdint>

namespace Hive {

// Play phases are metrics and curriculum labels only, never routing. They
// describe what the (single) policy is doing at a given moment, so wandb can
// show how often each situation occurs and how the policy performs in it.
// Deliberately NOT used to select a model; there is only one.

enum class PlayPhase : int {
	Aerial = 0,
	AirDribble,
	GroundDribble,
	Defend,
	Recover,
	Neutral,

	COUNT
};

constexpr int PLAY_PHASE_COUNT = static_cast<int>(PlayPhase::COUNT);

const char* PlayPhaseName(PlayPhase p);

struct PhaseThresholds {
	float airborneZ = 200.f;
	float aerialBallZ = 500.f;
	float airDribbleBallZ = 400.f;
	float ballNearDist = 350.f;
	float dribbleDist = 200.f;
	float dribbleBallZMax = 300.f;
	float defendThirdY = -1700.f;
};

PlayPhase ClassifyPhase(const RLGC::Player& player,
                        const RLGC::GameState& state,
                        const PhaseThresholds& t = {});

// Per-phase counters for training metrics.
struct PhaseCounts {
	std::array<int64_t, PLAY_PHASE_COUNT> counts = {};

	void Add(PlayPhase p) { counts[static_cast<int>(p)]++; }

	int64_t Total() const {
		int64_t total = 0;
		for (auto c : counts)
			total += c;
		return total;
	}
};

} // namespace Hive
