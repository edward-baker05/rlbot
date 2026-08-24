#pragma once

#include "env/Obs.h"
#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <cmath>
#include <filesystem>
#include <string>

namespace Dash {

struct RewardBudget {
	float goal = 1.f;
	float strongTouch = 0.1f;
	float velocityBallToGoal = 0.15f;
	float velocityPlayerToBall = 0.075f;
	float faceBall = 0.01f;
	float air = 0.004;

	// float pickupBoost = 10.f;
	// float saveBoost = 0.2f;
	// float bump = 20.f;
	// float demo = 80.f;
	// float goal = 150.f;
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

struct TeamDistribution {
	float p1v1 = 1.0f;
	float p2v2 = 0.0f;
	float p3v3 = 0.0f;

	int SampleTeamSize(int arenaIndex, int totalArenas) const {
		float total = p1v1 + p2v2 + p3v3;
		if (total <= 0.f || totalArenas <= 0)
			return 1;

		float norm1 = p1v1 / total;
		float norm2 = p2v2 / total;

		int count1 = static_cast<int>(std::round(norm1 * totalArenas));
		int count2 = static_cast<int>(std::round(norm2 * totalArenas));

		if (norm1 > 0.f && count1 == 0 && totalArenas > 0)
			count1 = 1;
		if (norm2 > 0.f && count2 == 0 && totalArenas > count1)
			count2 = 1;

		if (count1 + count2 > totalArenas) {
			if (count1 > 0 && count2 > 0)
				count2 = totalArenas - count1;
			else if (count1 > totalArenas)
				count1 = totalArenas;
		}

		if (arenaIndex < count1)
			return 1;
		if (arenaIndex < count1 + count2)
			return 2;
		return 3;
	}

	int MaxActivePlayersPerTeam() const {
		if (p3v3 > 0.f)
			return 3;
		if (p2v2 > 0.f)
			return 2;
		return 1;
	}
};

struct TrainConfig {
	int maxPlayersPerTeam = 3;
	TeamDistribution teamDistribution = {};

	bool maskActions = true;
	ObsMode obs = ObsMode::Advanced;
	float infiniteBoostChance = 0.1f;

	RewardBudget rewards = {};
	float teamSpirit = 0.0f;
	ModelShape modelShape = {};
	SelfPlayConfig selfPlay = {};

	float noTouchTimeoutSeconds = 12.f;
	float timeoutSeconds = 60.f;
	int numGames = 256;
	int tickSkip = 8;
	int actionDelay = 7;

	int tsPerItr = 100'000;
	int miniBatchSize = 50'000;
	int epochs = 2;

	float entropyScale = 0.035f;
	float entropyTarget = 0.0f;
	float entropyAdjustRate = 0.15f;
	float gaeGamma = 0.99f;
	float policyLR = 2e-4f;
	float criticLR = 2e-4f;
	int64_t maxSteps = 0;

	std::filesystem::path checkpointRoot = "checkpoints";
	std::string runLabel = {};
	int64_t tsPerSave = 1'000'000;
	int checkpointsToKeep = 16;
	int64_t randomSeed = -1;

	bool sendMetrics = true;
	std::string wandbProject = "dash-rl";
	std::string wandbGroup = "dev";

	bool renderMode = false;
	float renderTimeScale = 1.f;
	bool useGPU = true;

	std::string RunName() const {
		return runLabel.empty() ? std::string("temp") : runLabel;
	}

	std::filesystem::path CheckpointFolder() const {
		return checkpointRoot / RunName();
	}
};

} // namespace Dash
