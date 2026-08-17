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
// conservative -- it is much easier to diagnose a run that is too small than
// one that OOMs three hours in. See docs/tuning.md before changing these.
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
// Training a slice of games that way stops the policy going off-distribution
// the moment a match stops being perfectly symmetric.
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
// it is the situation the bot is actually in most of the time, and
// over-weighting exotic scenarios produces a bot that can air dribble but
// cannot rotate. The curriculum must agree with the reward function. Spawning
// air dribble scenarios while the reward pays nothing for air dribbling does
// not teach air dribbling -- it just spends samples on a situation the policy
// has no gradient to improve at, and hands it a stream of episodes it can only
// fail.
//
// These weights are matched to RewardPhase::Foundations. When the reward phase
// advances, raise the corresponding scenarios with it.
struct CurriculumWeights {
  float neutralPlay = 35.f; // Ordinary play. Keep this dominant.

  // Car spawned next to the ball, so contact is nearly free.
  //
  // Heavily weighted in phase 1 on purpose: touch rate is the binding
  // constraint (~one touch per 20 seconds), and the phase's main outcome
  // reward cannot shape anything while it almost never fires. This buys the
  // touch reward enough firings to be a real gradient.
  //
  // Possibly lower once touch ratio clears a threshold -- past a point it is
  // training on a situation the bot already gets for free, and the weight may
  // be better spent on NeutralPlayState.
  float ballContact = 20.f;

  float defend = 15.f;

  // Cars end up airborne whether or not we spawn them there, and landing
  // on your wheels is a prerequisite for everything else.
  float recover = 8.f;

  // Ball in the air, cars on the ground. A phase-1 scenario despite the
  // name: the lesson is "the ball is up, go and meet it", which the touch
  // reward pays for directly. Jumping will hopefully emerges from this
  // rather than from being paid to leave the ground.
  float aerial = 5.f;

  // Close ball control on the ground.
  float groundDribble = 5.f;

  // Full kickoffs mixed into general training. The general policy has to play
  // the seconds right after the kickoff model hands over, so it needs to have
  // seen post-kickoff states.
  float kickoff = 8.f;

  // --- Later phases -------------------------------------------------------
  // Zero until the reward function pays for them. CombinedState drops
  // zero-weight setters entirely, so these cost nothing while disabled.
  float airDribble = 0.f; // RewardPhase::Aerial
  float flipReset = 0.f;  // RewardPhase::Aerial
  float demo = 0.f;       // RewardPhase::Teamplay
};

enum class RewardPhase {
  Foundations,
  Possession,
  Aerial,
  Teamplay,
};

struct RewardWeights {
  float velPlayerToBall = 3.f;
  float strongTouch = 50.f;
  float velBallToGoal = 4.f;
  float goal = 30.f;
  float pickupBoost = 5.f;
  float faceBall = 0.1f;
};

struct KickoffRewardWeights {
  float velPlayerToBall = 1.f;
  float strongTouch = 40.f;
  float velBallToGoal = 6.f;
  float goal = 30.f;
};

struct SelfPlayConfig {
  // Play a fraction of games against saved old versions.
  bool trainAgainstOldVersions = false;

  // Chance per iteration that it trains against an old version
  float trainAgainstOldChance = 0.15f;

  // How often to snapshot the current policy into the version pool.
  //
  // GigaLearn's own default is 25M, which suits multi-billion-step runs. That
  // is far too coarse to observe anything in a short comparison run -- with a
  // 50M budget you would snapshot twice and self-play would barely engage.
  // 5M gives a usable pool quickly; raise it towards 25M for long runs, since
  // a pool full of near-identical recent versions is not much of a test.
  int64_t tsPerVersion = 5'000'000;

  // Version pool size
  int maxOldVersions = 32;

  // --- Skill tracking (measurement) --------------------------------------
  // Worth enabling even without trainAgainstOldVersions, since it is what
  // makes two runs comparable.
  bool trackSkill = true;

  // Arenas used for evaluation matches. These compete with training for CPU,
  // so keep it well under the core count; 8 costs little on a 6-core part.
  int skillArenas = 8;

  // Iterations between evaluation runs.
  int skillUpdateInterval = 100;

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
  RewardPhase rewardPhase = RewardPhase::Foundations;

  RewardWeights rewards = {};
  KickoffRewardWeights kickoffRewards = {};
  ModelShape modelShape = {};
  SelfPlayConfig selfPlay = {};

  // Episode limits
  float noTouchTimeoutSeconds = 12.f; // General: end a stalled episode.
  float kickoffTimeoutSeconds = 6.f;  // Kickoff: end if nobody touches it.

  // --- Throughput ---------------------------------------------------------
  // numGames is the main CPU knob. More games means better gradient
  // estimates and more RAM.
  int numGames = 128;

  int tickSkip = 8;
  // One tick less than tickSkip, matching every other RLGym framework. The
  // RLBot client MUST use the same value or the deployed bot will act on a
  // different cadence than it trained with.
  int actionDelay = 7;

  // --- PPO ----------------------------------------------------------------
  int tsPerItr = 50'000;

  // Minibatch is the main VRAM knob. There is 6GB available at the moment.
  int miniBatchSize = 25'000;

  int epochs = 1;
  float entropyScale = 0.035f;
  float gaeGamma = 0.99f;
  float policyLR = 1.5e-4f;
  float criticLR = 1.5e-4f;

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
