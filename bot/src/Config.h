#pragma once

#include "policy/Policy.h"

#include <GigaLearnCPP/LearnerConfig.h>

#include <filesystem>
#include <string>

namespace Hive {

// Everything tunable lives here. Defaults are sized for the training machine
// (Ryzen 3600, 6 cores; RTX 2060, 6 GB VRAM).

// Relative weights for how often each scenario spawns. They control what the
// policy practises, not which model runs -- there is one model. Re-derive
// them from telemetry at each phase gate.
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

// --- Reward budgets --------------------------------------------------------
//
// Every reward weight in this project is declared as a BUDGET in goal-units and
// converted to a per-step weight in exactly one place (Hive::GeneralRewardSpecs).
// A goal is 1.0 by definition; nothing else may be written as a bare per-step
// float.
//
// This exists because p1air's `grounded = 0.05` integrated to 9.0 goal-units
// per episode -- nine goals per episode for holding still on the wheels -- and
// nobody noticed, because nobody wrote down the integral. Declaring the
// integral makes that class of mistake unrepresentable.

// tickSkip 8 at RocketSim's 120 Hz.
inline constexpr float STEPS_PER_SECOND = 15.f;

// Working figure for one episode. noTouchTimeout caps a never-touching bot at
// 180 steps and goals end episodes early, so 10 s is a reasonable middle. The
// Episode/Mean Steps metric exists so this is re-derived from telemetry rather
// than staying a guess.
inline constexpr float REFERENCE_EPISODE_SECONDS = 10.f;
inline constexpr float REFERENCE_EPISODE_STEPS =
    STEPS_PER_SECOND * REFERENCE_EPISODE_SECONDS;

// Goal-units earned by holding a behaviour perfectly for one reference episode,
// converted to the per-step weight that earns it.
inline constexpr float RateWeight(float budgetPerEpisode) {
  return budgetPerEpisode / REFERENCE_EPISODE_STEPS;
}

// Goal-units of cost for one second of a condition, converted to per-step.
inline constexpr float PerSecondWeight(float budgetPerSecond) {
  return budgetPerSecond / STEPS_PER_SECOND;
}

// All values are goal-units. See docs/superpowers/specs/2026-08-18-reward-redesign-design.md
// for the derivation of each; none of them is a guess.
struct RewardBudget {
  // --- Rate: earned by holding the behaviour for one reference episode ---

  // Squared, so 0.375 of this is the free coasting floor at 1410 uu/s. The
  // whole-episode maximum equals exactly two ball touches, which is the cap
  // on how much a speed farm can ever be worth.
  float speed = 0.30f;

  float faceBall = 0.20f;

  // Exactly faceBall/3, which is the 2:1 asymmetry (w- = w+/2) expressed
  // through the decomposition. Changing one without the other changes the
  // asymmetry silently.
  float faceBallAxis = 0.20f / 3.f;

  // --- Event: earned per occurrence ---
  float touch = 0.15f;
  float cleanLanding = 0.10f;

  // Cost of a full-speed crash.
  float harshSpeedLoss = 0.10f;

  // --- Per second ---
  // Three seconds on your roof costs 0.30, which is 3x what the best possible
  // landing pays. Recovery is worth more as an avoided loss than as a bonus,
  // which is what stops the landing bonus becoming the objective.
  float wrongSurface = 0.10f;

  // The unit. Not adjustable, and scaling it could not help anyway: rewards
  // are standardized at GAE.cpp:52, and for a rare terminal payoff the
  // signal-to-noise sqrt(p/(1-p)) is independent of magnitude. The only lever
  // on the goal signal is how often goals happen.
  static constexpr float GOAL = 1.f;
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

  RewardBudget rewards = {};
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
