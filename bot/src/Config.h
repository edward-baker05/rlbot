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
	float strongTouch = 0.2f;
	float airTouch = 0.3f;
	float dribbleFlick = 0.003f;
	float onTarget = 0.3f;

	float pickupBoost = 0.1f;
	float saveBoost = 0.0005f;
	float speed = 0.0005f;

	float bump = 0.025f;
	float demo = 0.5f;

	float save = 0.25f;
	float ballToOwnGoal = 0.01f;

	float awkwardContact = 0.003f;
	float possession = 0.0003f;
	float kickoff = 0.025f;
	float goalside = -0.0001f;
};

struct AerialConfig {
	float aerialSpawnChance = 0.15f;

	float hoverFraction = 0.35f;
	float minBallHeight = 350.f;
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
	int benchArenas = 16;
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

	float entropyScale = 0.0025f;
	float entropyTarget = 0.f;
	float entropyAdjustRate = 0.15f;

	// Normalise entropy against the actions actually LEGAL in each state
	// (log of the valid count) rather than against log(90).
	//
	// With this off -- GigaLearn's default -- measured entropy conflates two
	// unrelated things: how sharp the policy is, and how many actions the mask
	// happened to allow. A bot that spends more time airborne has more legal
	// actions and so reads as higher entropy without the policy changing at
	// all. That makes `Policy Entropy` a poor diagnostic and makes it unusable
	// as a controlled variable, since a controller would chase changes in
	// flight time.
	//
	// NOTE: this changes the SCALE of every entropy number. Dividing by
	// log(validCount) instead of log(90) yields larger values, so entropy
	// readings before and after this flag are not comparable, and any
	// entropyTarget has to be recalibrated against fresh measurements.
	bool maskEntropy = true;

	// PPO's vf_coef: scales the critic loss where it is summed with the policy
	// loss, i.e. where the SHARED TRUNK takes its gradient from.
	//
	// Left at 1.0 (GigaLearn's implicit value, and what 1.42B steps of this
	// lineage trained under) because the argument for lowering it did not hold
	// up. Recording why, so it is not re-derived wrongly later:
	//
	//   The case for 0.5 was that `Critic Loss` >> `Policy Loss`, so the trunk
	//   was supposedly learning to predict value rather than to act. But the
	//   policy loss is near zero BY CONSTRUCTION: we standardize advantages, so
	//   they are mean-zero, and -mean(min(ratio*A, clipped*A)) ~= -mean(A) ~= 0
	//   whenever ratio ~= 1. Its gradient scales with std(A) = 1, not with the
	//   loss value. Comparing the two loss magnitudes measures nothing about
	//   which one moves the network.
	//
	// The knob is still real, and still worth an experiment later: the trunk's
	// gradient is g_policy + criticLossScale * g_critic, so this changes the
	// DIRECTION of the combined gradient. What it does not do is change the
	// critic head's step size -- see criticLR below.
	//
	// Decide it from `GAE/Explained Variance` plus `Policy Update Magnitude` vs
	// `Critic Update Magnitude`, not from the loss values.
	float criticLossScale = 1.f;

	// KL-targeting LR controller. 0 = off; policyLR stays fixed.
	//
	// CALIBRATED 2026-08-25 on the `calib` branch of the t3 lineage (17
	// iterations from 1.4005B, this exact config, controller off):
	//
	//   Mean KL Divergence       median 0.00217  range 0.00169 - 0.00299
	//   SB3 Clip Fraction        median 0.0207   range 0.0158  - 0.0302
	//   GAE/Explained Variance   median 0.781    range 0.767   - 0.795
	//   Policy Entropy           median 0.488    range 0.483   - 0.492
	//
	// The target is set to the MEASURED MEDIAN on purpose. The controller's job
	// here is not to push the update size somewhere new -- 0.0022 is what this
	// config demonstrably sustains, with a healthy clip fraction and a critic
	// explaining ~78% of its target's variance. Its job is to stop that value
	// DECAYING, which is the failure this project has actually hit: KL sliding
	// 1.25e-3 -> 6.9e-4 while the run looked alive and learned nothing.
	//
	// So this is a stabiliser, not an intervention. Raising it above 0.0022
	// asks for larger updates than this lineage has ever taken and is a
	// separate, deliberate experiment -- not a knob to nudge while changing
	// other things.
	//
	// Note the textbook target of 0.01 is ~5x this. That figure assumes ~10 PPO
	// epochs; we run 2, so our natural KL is correspondingly smaller. Do not
	// "correct" this number toward 0.01.
	float klTarget = 0.0022f;
	float klAdjustRate = 0.3f;
	float gaeGamma = 0.995f;
	float policyLR = 1e-4f;

	// Deliberately unchanged alongside criticLossScale. Scaling a loss does NOT
	// scale Adam's step: the update is m / (sqrt(v) + eps), and a constant
	// factor on the gradient scales m and sqrt(v) together, so it cancels. For
	// any parameter whose gradient comes from one loss only -- which is every
	// parameter in the critic HEAD -- criticLossScale is invisible.
	//
	// It only bites on the shared trunk, where two gradients are summed and the
	// factor changes their ratio rather than the overall magnitude. So there is
	// no criticLR compensation to make; changing it would be a real,
	// unjustified change of the critic's step size.
	float criticLR = 1e-4f;
	int64_t maxSteps = 0;

	std::filesystem::path checkpointRoot = "checkpoints";
	std::string runLabel = {};
	int64_t tsPerSave = 25'000'000;
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
