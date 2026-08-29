#pragma once

#include "env/Obs.h"
#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace Dash {

struct RewardBudget {
	float goal = 1.f;
	float strongTouch = 0.45f;

	float airTouch = 0.50f;
	float airVtB = 0.01f;
	float airFaceBall = 0.005f;
	float airLaunch = 0.01f;

	float pickupBoost = 0.10f;
	float saveBoost = 0.0003f;
	float speed = 0.0002f;

	float bump = 0.15f;
	float demo = 0.4f;

	float save = 0.5f;
	float velBtG = -0.0f;

	float awkwardContact = 0.005f;
	float possession = 0.00f;
};

struct AerialConfig {
	float aerialSpawnChance = 0.15f;

	float hoverFraction = 0.15f;
	float minBallHeight = 500.f;
	float maxBallHeight = 1800.f;
	float initialBoost = 100.f;
};

struct SelfPlayConfig {
	bool trainAgainstOldVersions = true;
	float trainAgainstOldChance = 0.1f;
	int64_t tsPerVersion = 25'000'000;
	int maxOldVersions = 16;

	bool trackSkill = true;
	int skillArenas = 8;
	int skillUpdateInterval = 100;
	float skillSimTime = 45.f;
	float skillMaxSimTime = 240.f;
};

struct NectoConfig {
	bool enabled = true;

	float arenaFraction = 0.1f;

	float trainBeta = 0.5f;
	float benchBeta = 1.0f;

	std::filesystem::path modelPath =
		"../../libs/opponents/NectoFamily/nexto/nexto-model.pt";

	std::filesystem::path ResolvedModelPath() const {
		if (const char *env = std::getenv("DASH_NECTO_MODEL"))
			if (*env)
				return env;
		return modelPath;
	}

	bool benchmark = true;
	int benchInterval = 100;
	int benchArenas = 64;
	float benchSimTime = 45.f;
	float benchMaxSimTime = 240.f;
	float benchEloK = 5.f;
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
	ObsMode obs = ObsMode::Predictive;
	float infiniteBoostChance = 0.1f;

	RewardBudget rewards = {};
	AerialConfig aerial = {};
	float teamSpirit = 0.f;
	ModelShape modelShape = {};
	SelfPlayConfig selfPlay = {};
	NectoConfig necto = {};

	float noTouchTimeoutSeconds = 12.f;
	float timeoutSeconds = 120.f;
	int numGames = 256;
	int tickSkip = 8;
	int actionDelay = 7;

	int tsPerItr = 250'000;
	int miniBatchSize = 50'000;
	int epochs = 2;

	// 0.49 is t3's measured entropy under maskEntropy; a fixed scale only slows
	// entropy's descent, so the controller holds it there instead. The
	// reasoning for these and the four knobs below is in
	// scratch/02-ppo-algorithm-surface.md.
	float entropyScale = 0.02f;
	float entropyTarget = 0.49f;
	float entropyAdjustRate = 0.15f;

	bool maskEntropy = true;

	// PPO's vf_coef, which only bites on the shared trunk.
	float criticLossScale = 1.f;

	// KL-targeting LR controller; 0 = off. 0.0022 is this lineage's measured
	// median KL, not a textbook value.
	float klTarget = 0.0022f;
	float klAdjustRate = 0.3f;
	float gaeGamma = 0.995f;
	float policyLR = 1e-4f;
	float criticLR = 1e-4f;
	int64_t maxSteps = 0;

	std::filesystem::path checkpointRoot = "checkpoints";
	std::string runLabel = {};
	int64_t tsPerSave = 25'000'000;
	int checkpointsToKeep = 20;
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
