#pragma once

#include "env/Obs.h"
#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Hive {

struct CurriculumWeights {
	float neutralPlay = 35.f;
	float ballContact = 10.f;
	float defend = 15.f;
	float recover = 8.f;
	float strike = 15.f;
	float aerial = 10.f;
	float kickoff = 8.f;

	float airDribble = 0.f;
	float flipReset = 0.f;
	float demo = 0.f;
};

inline constexpr float STEPS_PER_SECOND = 15.f;
inline constexpr float REFERENCE_EPISODE_SECONDS = 26.1f;
inline constexpr float REFERENCE_EPISODE_STEPS =
	STEPS_PER_SECOND * REFERENCE_EPISODE_SECONDS;

inline constexpr float RateWeight(float budgetPerEpisode) {
	return budgetPerEpisode / REFERENCE_EPISODE_STEPS;
}

inline constexpr float PerSecondWeight(float budgetPerSecond) {
	return budgetPerSecond / STEPS_PER_SECOND;
}

struct RewardBudget {
	float touchGoalAccel = 45.0f;
	float touchAccelExponent = 2.0f;
	float touchGoalAccelOpponentScale = 0.5f;
	float touchGoalAccelTeamSpirit = 0.0f;

	float goal = 25.0f;
	float shotOnTarget = 35.0f;
	float touchEdge = 0.1394f;

	float speedToBall = 18.9f;
	float faceBall = 1.59f;

	float saveBoost = 0.79f;
	float pickupBoost = 0.5067f;

	float airTouch = 12.f;
	float airTouchHeightExponent = 1.f;
	float airTouchDirectionExponent = 1.f;
	float air = 0.88f;
	float flipSpeed = 2.5f;
	float wrongSurface = 1.0f;
};
struct SelfPlayConfig {
	bool trainAgainstOldVersions = true;
	float trainAgainstOldChance = 0.2f;
	int64_t tsPerVersion = 5'000'000;
	int maxOldVersions = 32;

	bool trackSkill = true;
	int skillArenas = 8;
	int skillUpdateInterval = 100;
	float skillSimTime = 45.f;
	float skillMaxSimTime = 240.f;
};

struct TrainConfig {
	int maxPlayersPerTeam = 1;

	enum class SpawnMode { Random, Curriculum };
	SpawnMode spawn = SpawnMode::Random;
	CurriculumWeights curriculum = {};

	bool maskActions = false;
	ObsMode obs = ObsMode::Relative;
	float infiniteBoostChance = 0.1f;

	RewardBudget rewards = {};
	float teamSpirit = 0.0f;
	ModelShape modelShape = {};
	SelfPlayConfig selfPlay = {};

	float noTouchTimeoutSeconds = 12.f;
	int numGames = 128;
	int tickSkip = 8;
	int actionDelay = 7;

	int tsPerItr = 50'000;
	int miniBatchSize = 25'000;
	int epochs = 2;

	float entropyScale = 0.002f;
	float entropyTarget = 0.40f;
	float entropyAdjustRate = 0.15f;
	float gaeGamma = 0.99f;
	float policyLR = 2e-4f;
	float criticLR = 2e-4f;
	int64_t maxSteps = 0;

	std::filesystem::path checkpointRoot = "checkpoints";
	std::string runLabel = {};
	int64_t tsPerSave = 1'000'000;
	int checkpointsToKeep = 8;
	int64_t randomSeed = -1;

	bool sendMetrics = true;
	std::string wandbProject = "hivemind-rl";
	std::string wandbGroup = "dev";

	bool renderMode = false;
	float renderTimeScale = 1.f;
	bool useGPU = true;

	std::string RunName() const {
		return runLabel.empty() ? std::string("main") : "main-" + runLabel;
	}

	std::filesystem::path CheckpointFolder() const {
		return checkpointRoot / RunName();
	}
};

} // namespace Hive
