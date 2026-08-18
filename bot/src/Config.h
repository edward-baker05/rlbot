#pragma once

#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Hive {

// Everything tunable lives here. Defaults are sized for the training machine
// (Ryzen 3600, 6 cores; RTX 2060, 6 GB VRAM).

// Relative weights for how often each scenario spawns. They control what the
// policy practises, not which model runs -- there is one model. Matched to
// RewardPhase::Foundations; re-derive them from telemetry at each phase gate.
struct CurriculumWeights {
  float neutralPlay = 35.f; // Ordinary play. Keep this dominant.

  // Car spawned next to the ball, so contact is nearly free -- mostly
  // training on a situation the bot already gets for free at this point.
  float ballContact = 10.f;

  float defend = 15.f;

  // Cars end up airborne whether or not we spawn them there, and landing
  // on wheels is a prerequisite for everything else.
  float recover = 8.f;

  // The jump-flip strike: ball at jump height, car already rolling at it with
  // pace, so the only open decisions are when to leave the ground and
  // whether to flip.
  float strike = 15.f;

  // Ball in the air, cars on the ground: the lesson is "the ball is up, go
  // and meet it", which the touch reward pays for directly.
  float aerial = 10.f;

  // Full kickoffs mixed into training; the one policy plays its own
  // kickoffs, so it has to practise them here.
  float kickoff = 8.f;

  // Zero until the reward function pays for them. Zero-weight setters are
  // dropped entirely, so these cost nothing while disabled.
  float airDribble = 0.f;
  float flipReset = 0.f;
  float demo = 0.f;
};

enum class RewardPhase {
  Foundations,
  Possession,
  Aerial,
  Teamplay,
};

// Weights are derived from measured RewardShare telemetry, not guessed --
// see runs/RUNLOG.md for the derivation runs.
struct RewardWeights {
  float velPlayerToBall = 0.5f;

  float touch = 5.f;
  float strongTouch = 75.f;

  // Kept small: it pays on ball velocity the bot mostly did not cause
  // (kickoff momentum, bounces), and as zero-sum noise it dilutes the
  // advantage signal.
  float velBallToGoal = 1.f;

  float goal = 100.f;
  float pickupBoost = 5.f;
  float faceBall = 0.075f;

  // Exponent on the aim multiplier for StrongTouch. 1 = the plain
  // 0.5*(1+cos) curve; higher narrows the cone toward the goal (D6).
  float aimSharpness = 1.f;

  // Note: airborne-upright currently outpays grounded-idle (0.04 vs 0.02),
  // an intentional inversion -- the aerial phase wants the ground bias gone
  // anyway, so it is left rather than rebalanced.
  float airRecovery = 0.04f;
  float grounded = 0.05f;

  // Aerial phase only. Pays TouchHeightReward (height/1500 at contact, zero
  // for a ground touch) as a tiebreaker between floor and air contact, small
  // enough not to outbid actually striking the ball well.
  float touchHeight = 15.f;
};

struct SelfPlayConfig {
  // Play a fraction of games against saved old versions.
  bool trainAgainstOldVersions = true;

  // Chance per iteration that it trains against an old version.
  float trainAgainstOldChance = 0.15f;

  // How often to snapshot the current policy into the version pool.
  // GigaLearn's own default (25M) suits multi-billion-step runs but barely
  // engages in a short comparison run; 5M gives a usable pool quickly.
  // Raise it towards 25M for long runs.
  int64_t tsPerVersion = 5'000'000;

  int maxOldVersions = 32;

  // Skill tracking is what makes two runs comparable, so it is worth
  // enabling even without trainAgainstOldVersions.
  bool trackSkill = true;

  // Evaluation arenas compete with training for CPU; keep well under the
  // core count.
  int skillArenas = 8;

  // Iterations between evaluation runs.
  int skillUpdateInterval = 100;

  float skillSimTime = 45.f;
  float skillMaxSimTime = 240.f;
};

struct TrainConfig {
  // Players per team the observation reserves space for. Changing this
  // changes the observation width, which invalidates every existing
  // checkpoint -- treat it as permanent once a run you care about starts.
  int maxPlayersPerTeam = 1;

  CurriculumWeights curriculum = {};
  RewardPhase rewardPhase = RewardPhase::Foundations;

  RewardWeights rewards = {};
  ModelShape modelShape = {};
  SelfPlayConfig selfPlay = {};

  // End a stalled episode.
  float noTouchTimeoutSeconds = 12.f;

  // numGames is the main CPU knob. More games means better gradient
  // estimates and more RAM.
  int numGames = 128;

  int tickSkip = 8;
  // One tick less than tickSkip, matching every other RLGym framework. The
  // RLBot client MUST use the same value or the deployed bot acts on a
  // different cadence than it trained with.
  int actionDelay = 7;

  // --- PPO ---------------------------------------------------------------
  int tsPerItr = 100'000;

  // Minibatch is the main VRAM knob (6 GB available).
  int miniBatchSize = 25'000;

  // Upstream default.
  int epochs = 2;

  // GigaLearn's entropy is NORMALIZED to [0,1] (divided by log(numActions)),
  // so this scale is not comparable to rlgym-ppo's. Kept low so entropy can
  // actually fall over a run rather than sitting near-uniform; revisit at
  // every phase gate, since too low too early costs exploration.
  float entropyScale = 0.004f;

  float gaeGamma = 0.99f;

  // Upstream default.
  float policyLR = 3e-4f;
  float criticLR = 3e-4f;

  int64_t maxSteps = 0;

  // --- Bookkeeping -------------------------------------------------------
  std::filesystem::path checkpointRoot = "checkpoints";

  // Distinguishes runs. Without it, a second run resumes from the first
  // one's checkpoints, which silently invalidates any comparison between
  // them.
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

  std::string RunName() const {
    return runLabel.empty() ? std::string("main") : "main-" + runLabel;
  }

  std::filesystem::path CheckpointFolder() const {
    return checkpointRoot / RunName();
  }
};

} // namespace Hive
