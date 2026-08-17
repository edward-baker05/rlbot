#include "Train.h"

#include "../env/Env.h"
#include "../env/Obs.h"
#include "../policy/Regime.h"

#include <GigaLearnCPP/Learner.h>

#include <cstdlib>
#include <iostream>

using namespace GGL;
using namespace RLGC;

namespace Hive {

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
// These are what tell you whether training is working, and they are worth more
// than the reward curve. Reward goes up in every run; what you actually need to
// know is *how* the bot is earning it -- a bot that stopped aerialling and
// started farming ground touches shows a healthy reward curve and a collapsing
// touch height.
//
// Metrics run on a sampled fraction of steps because iterating every player of
// every game every step is a real cost at 128 games.
// Step budget for bounded runs. Set from RunTraining(); 0 disables.
static int64_t g_MaxSteps = 0;

static void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// --- Step budget --------------------------------------------------------
	// GigaLearn's training loop runs until the user presses Q; there is no
	// timestep limit and no documented way to break out of it. The step
	// callback is the only hook that runs inside the loop with access to the
	// learner, so the budget is enforced here.
	//
	// Save first, then _exit rather than return: unwinding out of a callback
	// mid-collection would race the worker threads, and there is nothing left
	// to clean up once the checkpoint is on disk.
	if (g_MaxSteps > 0 && static_cast<int64_t>(learner->totalTimesteps) >= g_MaxSteps) {
		std::cout << "\nReached step budget (" << learner->totalTimesteps
		          << " >= " << g_MaxSteps << "). Saving and exiting.\n";
		std::cout.flush();
		learner->Save();
		std::_Exit(0);
	}

	// Sample roughly a quarter of steps. Averages over an iteration are just
	// as accurate and cost a quarter as much.
	const bool sample = (rand() % 4) == 0;
	if (!sample)
		return;

	PhaseCounts phases;

	for (const GameState& state : states) {
		for (const Player& player : state.players) {
			// --- Play phase distribution -------------------------------------
			// Shows what the policy actually spends its time doing. If you
			// bump the aerial weight in the curriculum and the Aerial share
			// does not move, the setter is not doing what you think it is.
			const PlayPhase phase = ClassifyPhase(player, state);
			phases.Add(phase);

			// --- Core behaviour ----------------------------------------------
			report.AddAvg("Player/In Air Ratio", !player.isOnGround);
			report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
			report.AddAvg("Player/Demoed Ratio", player.isDemoed);
			report.AddAvg("Player/Speed", player.vel.Length());
			report.AddAvg("Player/Boost", player.boost);

			const Vec toBall = state.ball.pos - player.pos;
			const float dist = toBall.Length();
			if (dist > 1.f)
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0.f, player.vel.Dot(toBall / dist)));

			// Touch height is the clearest single indicator of whether the bot
			// is developing an air game. Watch it more than the reward.
			if (player.ballTouchedStep)
				report.AddAvg("Player/Touch Height", state.ball.pos.z);
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());

		report.AddAvg("Game/Ball Height", state.ball.pos.z);
		report.AddAvg("Game/Players", static_cast<float>(state.players.size()));
	}

	// Report each phase as a fraction of sampled player-steps.
	const int64_t total = phases.Total();
	if (total > 0) {
		for (int i = 0; i < PLAY_PHASE_COUNT; i++) {
			const auto phase = static_cast<PlayPhase>(i);
			report.AddAvg(std::string("Phase/") + PlayPhaseName(phase),
			              static_cast<float>(phases.counts[i]) / static_cast<float>(total));
		}
	}
}

// ---------------------------------------------------------------------------

void RunTraining(const TrainConfig& cfg) {
	// RocketSim needs the collision meshes to simulate the arena geometry.
	// Without them cars fall through the world, which presents as a bot that
	// learns nothing rather than as an obvious error.
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	const std::string meshPath = meshEnv ? meshEnv : "collision_meshes";
	RocketSim::Init(meshPath);

	// Probe the observation width rather than deriving it. See env/Obs.h.
	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam);
	std::cout << "Observation size: " << obsSize
	          << " (maxPlayersPerTeam=" << cfg.maxPlayersPerTeam << ")\n";
	std::cout << "Run:              " << cfg.RunName() << "\n";
	std::cout << "Checkpoints:      " << cfg.CheckpointFolder() << "\n";
	std::cout << "Self-play:        "
	          << (cfg.selfPlay.trainAgainstOldVersions
	                  ? "on (" + std::to_string(static_cast<int>(cfg.selfPlay.trainAgainstOldChance * 100)) +
	                        "% of iterations, snapshot every " +
	                        std::to_string(cfg.selfPlay.tsPerVersion / 1'000'000) + "M steps)"
	                  : "off")
	          << "\n";
	std::cout << "Skill tracking:   " << (cfg.selfPlay.trackSkill ? "on" : "off") << "\n";
	if (cfg.maxSteps > 0)
		std::cout << "Step budget:      " << cfg.maxSteps << "\n";

	g_MaxSteps = cfg.maxSteps;

	LearnerConfig lc = {};

	lc.deviceType = cfg.useGPU ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
	lc.numGames = cfg.numGames;
	lc.tickSkip = cfg.tickSkip;
	lc.actionDelay = cfg.actionDelay;
	lc.randomSeed = cfg.randomSeed;

	lc.checkpointFolder = cfg.CheckpointFolder();
	lc.tsPerSave = cfg.tsPerSave;
	lc.checkpointsToKeep = cfg.checkpointsToKeep;

	lc.ppo.tsPerItr = cfg.tsPerItr;
	lc.ppo.batchSize = cfg.tsPerItr;
	lc.ppo.miniBatchSize = cfg.miniBatchSize;
	lc.ppo.epochs = cfg.epochs;
	lc.ppo.entropyScale = cfg.entropyScale;
	lc.ppo.gaeGamma = cfg.gaeGamma;
	lc.ppo.policyLR = cfg.policyLR;
	lc.ppo.criticLR = cfg.criticLR;

	// The policy and shared-head shapes here MUST match ModelShape in
	// Config.h, because that is what the RLBot client rebuilds at load time.
	// Mismatch means the deployed bot silently loads garbage weights.
	lc.ppo.sharedHead.layerSizes = cfg.modelShape.sharedHeadLayers;
	lc.ppo.sharedHead.activationType = cfg.modelShape.activation;
	lc.ppo.sharedHead.addLayerNorm = cfg.modelShape.addLayerNorm;
	lc.ppo.sharedHead.addOutputLayer = false;

	lc.ppo.policy.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.policy.activationType = cfg.modelShape.activation;
	lc.ppo.policy.addLayerNorm = cfg.modelShape.addLayerNorm;

	// The critic is training-only, so it never has to match the client. Give it
	// the same shape as the policy; there is rarely a reason to differ.
	lc.ppo.critic.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.critic.activationType = cfg.modelShape.activation;
	lc.ppo.critic.addLayerNorm = cfg.modelShape.addLayerNorm;

	// --- Self-play ----------------------------------------------------------
	// The learner forces savePolicyVersions on if either of these is enabled,
	// since both need the version pool. Setting it explicitly documents the
	// dependency rather than relying on that.
	lc.trainAgainstOldVersions = cfg.selfPlay.trainAgainstOldVersions;
	lc.trainAgainstOldChance = cfg.selfPlay.trainAgainstOldChance;
	lc.savePolicyVersions = cfg.selfPlay.trainAgainstOldVersions || cfg.selfPlay.trackSkill;
	lc.tsPerVersion = cfg.selfPlay.tsPerVersion;
	lc.maxOldVersions = cfg.selfPlay.maxOldVersions;

	lc.skillTracker.enabled = cfg.selfPlay.trackSkill;
	lc.skillTracker.numArenas = cfg.selfPlay.skillArenas;
	lc.skillTracker.updateInterval = cfg.selfPlay.skillUpdateInterval;
	lc.skillTracker.simTime = cfg.selfPlay.skillSimTime;
	lc.skillTracker.maxSimTime = cfg.selfPlay.skillMaxSimTime;

	lc.sendMetrics = cfg.sendMetrics;
	lc.metricsProjectName = cfg.wandbProject;
	lc.metricsGroupName = cfg.wandbGroup;
	lc.metricsRunName = cfg.RunName();

	lc.renderMode = cfg.renderMode;
	lc.renderTimeScale = cfg.renderTimeScale;

	// Capture cfg by value: the learner calls this for every game at startup,
	// and outliving the caller's stack frame is not worth risking.
	auto envCreateFn = [cfg](int index) -> EnvCreateResult {
		return CreateEnv(index, cfg);
	};

	Learner learner(envCreateFn, lc, StepCallback);
	learner.Start();
}

} // namespace Hive
