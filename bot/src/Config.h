#pragma once

#include "env/Obs.h"
#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Dash {

struct RewardBudget {
	float goal = 1.0f;
};

struct SelfPlayConfig {
	bool trainAgainstOldVersions = false;
	float trainAgainstOldChance = 0.5f;
	int64_t tsPerVersion = 5'000'000;
	int maxOldVersions = 32;

	bool trackSkill = false;
	int skillArenas = 8;
	int skillUpdateInterval = 100;
	float skillSimTime = 45.f;
	float skillMaxSimTime = 240.f;
};

struct TrainConfig {
	int maxPlayersPerTeam = 1;

	bool maskActions = false;
	ObsMode obs = ObsMode::Default;
	float infiniteBoostChance = 0.1f;

	RewardBudget rewards = {};
	float teamSpirit = 0.0f;
	ModelShape modelShape = {};
	SelfPlayConfig selfPlay = {};

	float noTouchTimeoutSeconds = 12.f;
	int numGames = 128;
	int tickSkip = 8;
	int actionDelay = 7;

	int tsPerItr = 100'000;
	int miniBatchSize = 25'000;
	int epochs = 2;

	float entropyScale = 0.003f;
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
	std::string wandbProject = "dash-rl";
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

} // namespace Dash
