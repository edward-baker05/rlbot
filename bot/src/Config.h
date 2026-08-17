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
// Self-play
// ---------------------------------------------------------------------------
// By default the policy only ever plays against its current self. That is
// cheap and works, but it lets the policy drift into strategies that beat
// *itself right now* rather than strategies that are actually strong -- both
// sides co-adapt, and a weakness neither side exploits never gets punished.
//
// Training against saved older versions fixes this: periodically the current
// weights are snapshotted, and some fraction of games are played against a
// randomly chosen snapshot instead of the live policy. An exploit that only
// works against the current self stops paying off, because the opponent pool
// does not co-adapt with you.
//
// The skill tracker is the measurement half. It runs evaluation matches
// between the current policy and old versions and maintains an ELO-style
// rating per game mode. This is the only honest answer to "is it actually
// getting better" -- average reward rises in every run, including ones where
// the policy has just found a better way to farm shaping rewards.
struct SelfPlayConfig {
	// Play a fraction of games against saved old versions.
	bool trainAgainstOldVersions = false;

	// Chance per iteration that it trains against an old version instead of
	// itself. Low on purpose: the point is to keep the policy honest, not to
	// stop it improving against a live opponent.
	float trainAgainstOldChance = 0.15f;

	// How often to snapshot the current policy into the version pool.
	//
	// GigaLearn's own default is 25M, which suits multi-billion-step runs. That
	// is far too coarse to observe anything in a short comparison run -- with a
	// 50M budget you would snapshot twice and self-play would barely engage.
	// 5M gives a usable pool quickly; raise it towards 25M for long runs, since
	// a pool full of near-identical recent versions is not much of a test.
	int64_t tsPerVersion = 5'000'000;

	// Version pool size. Older versions past this are dropped.
	int maxOldVersions = 32;

	// --- Skill tracking (measurement) --------------------------------------
	// Worth enabling even without trainAgainstOldVersions, since it is what
	// makes two runs comparable.
	bool trackSkill = false;

	// Arenas used for evaluation matches. These compete with training for CPU,
	// so keep it well under the core count; 8 costs little on a 6-core part.
	int skillArenas = 8;

	// Iterations between evaluation runs. Higher means less overhead and a
	// coarser rating curve.
	int skillUpdateInterval = 20;

	float skillSimTime = 45.f;
	float skillMaxSimTime = 240.f;
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
	SelfPlayConfig selfPlay = {};

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

	// Stop after this many timesteps. 0 means run until you press Q.
	//
	// Exists so two configurations can be compared over an identical budget.
	// Comparing by wall clock is misleading when one config is slower per step
	// -- the slower run would look worse simply for having seen less data.
	int64_t maxSteps = 0;

	// --- Bookkeeping --------------------------------------------------------
	std::filesystem::path checkpointRoot = "checkpoints";

	// Distinguishes runs of the same target. Without it, a second run resumes
	// from the first one's checkpoints, which silently invalidates any
	// comparison between them.
	std::string runLabel = {};

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
	std::string TargetName() const {
		return std::string(target == TrainTarget::Kickoff ? "kickoff" : "general");
	}

	std::string RunName() const {
		return runLabel.empty() ? TargetName() : TargetName() + "-" + runLabel;
	}

	std::filesystem::path CheckpointFolder() const {
		return checkpointRoot / RunName();
	}
};

} // namespace Hive
