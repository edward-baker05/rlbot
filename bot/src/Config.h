#pragma once

#include "policy/PolicySet.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Hive {

// ============================================================================
// Configuration
// ============================================================================
// Everything tunable lives here. Defaults are chosen for the target machine
// (Ryzen 3600, 6 cores / 12 threads; RTX 2060, 6 GB VRAM) and are deliberately
// conservative -- it is much easier to diagnose a run that is too small than one
// that OOMs three hours in. See docs/tuning.md before changing these.
// ============================================================================

enum class TrainTarget {
	General, // The main policy: everything except kickoffs.
	Kickoff  // The kickoff policy: reset until first touch.
};

// ---------------------------------------------------------------------------
// Team sizes
// ---------------------------------------------------------------------------
// A single policy handles 1s, 2s and 3s because the observation is padded to a
// fixed width (see env/Obs.h) and team slots are shuffled. To make that work
// the policy has to actually SEE all three sizes during training, so games are
// distributed across sizes rather than fixed at one.
//
// The asymmetric share matters for your human-player requirement: when a human
// joins, leaves, or is demoed at the wrong moment, teams are briefly uneven.
// Training a slice of games that way stops the policy going off-distribution the
// moment a match stops being perfectly symmetric.
struct TeamSizeMix {
	float weight1v1 = 0.30f;
	float weight2v2 = 0.30f;
	float weight3v3 = 0.30f;

	// Uneven teams, e.g. 1v2, 2v3, 1v3.
	float weightAsymmetric = 0.10f;
};

// ---------------------------------------------------------------------------
// Curriculum
// ---------------------------------------------------------------------------
// Relative weights for how often each scenario spawns when training the general
// policy. These do NOT select a model -- there is only one general model. They
// control what it practises.
//
// Rare-but-valuable skills need weight far above their natural frequency: a
// flip reset essentially never occurs under random play, so without a setter
// the policy sees too few to learn from. Equally, do not starve NeutralPlay --
// it is the situation the bot is actually in most of the time, and over-weighting
// exotic scenarios produces a bot that can air dribble but cannot rotate.
struct CurriculumWeights {
	float neutralPlay = 40.f;   // Ordinary play. Keep this dominant.
	float defend = 15.f;
	float recover = 10.f;
	float aerial = 12.f;
	float groundDribble = 8.f;
	float airDribble = 6.f;
	float flipReset = 4.f;
	float demo = 5.f;

	// Full kickoffs mixed into general training. Small but non-zero: the
	// general policy still has to play the seconds right after the kickoff
	// model hands over, so it needs to have seen post-kickoff states.
	float kickoff = 6.f;
};

// ---------------------------------------------------------------------------
// Rewards
// ---------------------------------------------------------------------------
// Weights are relative; the learner standardises returns, so what matters is
// the ratio between them, not the absolute scale.
struct RewardWeights {
	// Movement and control
	float air = 0.25f;
	float faceBall = 0.25f;
	float velPlayerToBall = 4.f;

	// Ball striking
	float strongTouch = 60.f;
	float velBallToGoal = 2.f;

	// Boost economy
	float pickupBoost = 10.f;
	float saveBoost = 0.2f;

	// Contact
	float bump = 20.f;
	float demo = 80.f;

	// Terminal
	float goal = 150.f;

	// Height of touch, which is what pushes the policy off the floor and into
	// aerial play. Raise this if the bot stays grounded; lower it if it starts
	// jumping at everything.
	float touchHeight = 8.f;
};

// Kickoff training uses a much narrower reward set. The objective is simple and
// short-horizon: get to the ball first, hit it hard, hit it towards their half.
// Adding general-play rewards here teaches the kickoff model to do things it
// will never be asked to do, because it hands over at first touch.
struct KickoffRewardWeights {
	float velPlayerToBall = 6.f;
	float strongTouch = 100.f;
	float velBallToGoal = 3.f;
	float saveBoost = 0.5f;
	float goal = 150.f;
};

// ---------------------------------------------------------------------------
// Top-level training config
// ---------------------------------------------------------------------------
struct TrainConfig {
	TrainTarget target = TrainTarget::General;

	// Max players per team the observation reserves space for. 3 covers 1s, 2s
	// and 3s. Changing this changes the observation width, which invalidates
	// every existing checkpoint -- treat it as permanent once you start a run.
	int maxPlayersPerTeam = 3;

	TeamSizeMix teamSizes = {};
	CurriculumWeights curriculum = {};
	RewardWeights rewards = {};
	KickoffRewardWeights kickoffRewards = {};
	ModelShape modelShape = {};

	// Episode limits
	float noTouchTimeoutSeconds = 12.f;  // General: end a stalled episode.
	float kickoffTimeoutSeconds = 6.f;   // Kickoff: end if nobody touches it.

	// --- Throughput ---------------------------------------------------------
	// numGames is the main CPU knob. More games means better gradient
	// estimates and more RAM. On a 3600, 128 mixed-size games keeps all 12
	// threads busy without thrashing; raise it if collection is CPU-idle,
	// lower it if you run out of RAM.
	int numGames = 128;

	int tickSkip = 8;
	// One tick less than tickSkip, matching every other RLGym framework. The
	// RLBot client MUST use the same value or the deployed bot will act on a
	// different cadence than it trained with.
	int actionDelay = 7;

	// --- PPO ----------------------------------------------------------------
	int tsPerItr = 50'000;

	// Minibatch is the main VRAM knob. 6 GB is not much, especially if the card
	// is also driving your monitors. 25k leaves headroom; drop to 12'500 if you
	// hit CUDA OOM, and note that OOM often appears several iterations in
	// rather than immediately.
	int miniBatchSize = 25'000;

	int epochs = 1;
	float entropyScale = 0.035f;
	float gaeGamma = 0.99f;
	float policyLR = 1.5e-4f;
	float criticLR = 1.5e-4f;

	// --- Bookkeeping --------------------------------------------------------
	std::filesystem::path checkpointRoot = "checkpoints";
	int64_t tsPerSave = 1'000'000;
	int checkpointsToKeep = 8;
	int64_t randomSeed = -1; // -1 uses the clock

	bool sendMetrics = true;
	std::string wandbProject = "hivemind-rl";
	std::string wandbGroup = "dev";

	bool renderMode = false; // Stream to RocketSimVis instead of training fast
	float renderTimeScale = 1.f;

	bool useGPU = true;

	// Derived helpers
	std::filesystem::path CheckpointFolder() const {
		return checkpointRoot / (target == TrainTarget::Kickoff ? "kickoff" : "general");
	}

	std::string RunName() const {
		return std::string(target == TrainTarget::Kickoff ? "kickoff" : "general");
	}
};

} // namespace Hive
