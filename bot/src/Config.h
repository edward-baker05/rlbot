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

  // Car spawned next to the ball, so contact is nearly free. Heavily
  // weighted while touch rate is the binding constraint: outcome rewards
  // cannot shape anything while they almost never fire.
  float ballContact = 20.f;

  float defend = 15.f;

  // Cars end up airborne whether or not we spawn them there, and landing
  // on wheels is a prerequisite for everything else.
  float recover = 8.f;

  // Ball in the air, cars on the ground: the lesson is "the ball is up, go
  // and meet it", which the touch reward pays for directly.
  float aerial = 5.f;

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

// Derived from measured RewardShare telemetry, not guessed -- see
// runs/RUNLOG.md for the derivation runs. The 30M-step baseline with the old
// weights (velPlayerToBall=3, goal=30, no plain touch term) paid 67% of all
// reward mass to velPlayerToBall and ~5% to outcomes; the policy farmed the
// approach term by flip-spamming and never learned to touch the ball.
struct RewardWeights {
  float velPlayerToBall = 0.5f;

  float touch = 5.f;
  float strongTouch = 50.f;

  // Small: it pays on ball velocity the bot mostly did not cause (kickoff
  // momentum, bounces), and as zero-sum noise it dilutes the advantage
  // signal. Measured at 0.67 share at weight 2 with everything else gated.
  //
  // p1probe-h zeroed this to test the "variance pump" hypothesis and
  // FALSIFIED it: removing 44.3% of all reward mass moved GAE/Returns STD by
  // 4% (21.3 -> 20.5) and left the advantage decay identical (-39% vs -40%
  // over 30M). Rewards are standardized (GAE.cpp divides every reward by
  // returnStd), so absolute mass share is not what sets the noise floor.
  // It also made things worse: mass reflowed onto VelPlayerToBall (0.14 ->
  // 0.40) and PickupBoost (0.13 -> 0.22), re-establishing the approach farm.
  // Left at 0.5. Do not re-probe reward weights without a new mechanism.
  float velBallToGoal = 1.f;

  float goal = 150.f;
  float pickupBoost = 10.f;
  float faceBall = 0.05f;

  // --- Air handling (added 2026-08-17, see runs/RUNLOG.md p1age) ----------
  // Both weights are anchored to what a grounded approach step actually pays,
  // measured rather than guessed: at the observed 233 uu/s approach speed,
  // VelPlayerToBall pays 0.0453 per step (Pay/VelPlayerToBall Per Step
  // Grounded). These are set around that number so the ordering per step is
  //
  //   grounded + approaching   0.05 + up to 0.5   <- best
  //   grounded, doing nothing  0.05
  //   airborne, wheels down    0.04
  //   airborne, tumbling       ~0
  //   airborne, inverted       down to -0.04      <- worst
  //
  // with no ties, so being on the ground strictly dominates being in the air.
  // Verify against the Pay/* metrics on the first run and correct from the
  // realized split -- these are a derivation, not a measurement, until then.
  float airRecovery = 0.04f;
  float grounded = 0.02f;
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

  // Upstream default. The inherited 1 starved the policy of updates: with
  // one epoch at half the upstream LR, measured KL divergence and clip
  // fraction were ~0 for entire runs.
  int epochs = 2;

  // GigaLearn's entropy is NORMALIZED to [0,1] (divided by log(numActions)),
  // so this scale is not comparable to rlgym-ppo's. Measured on 30M-step
  // probes: at 0.035 (inherited) and at 0.018 (upstream default), entropy
  // sat at ~0.78 for whole runs with KL ~0 -- the policy never left the
  // uniform distribution and nothing was learned. 0.002 is the largest
  // value probed that lets entropy actually fall (0.77 -> 0.65 over 30M).
  // Revisit at every phase gate; too low too early costs exploration.
  //
  // p1probe-i tried 0.0005 and FALSIFIED the exploration hypothesis below.
  // Entropy did collapse -- 0.729 -> 0.330 over 26M, i.e. ~4.4 effective
  // actions vs ~23 at baseline, the most deterministic policy this project
  // has produced -- and behaviour did not move at all: In Air Ratio 0.916 vs
  // baseline 0.918, Phase/Recover 0.852 vs 0.862. The flip-spam is not
  // exploration noise; a committed policy commits to flipping. Back to 0.002.
  // Caveat on that reading: Policy Entropy is a batch mean and the batch is
  // 92% airborne states, so 0.330 mostly describes the AIRBORNE policy. It
  // does not prove the grounded policy concentrated.
  //
  // Superseded reasoning for the 0.0005 probe: at 0.002 the entropy bonus
  // (scale * entropy = 0.002 * 0.68 = 1.4e-3) is the same order as the whole
  // policy loss (|Policy Loss| ~2e-3 from 20M on, oscillating through zero),
  // so the policy sits at the entropy-regularization fixed point rather than
  // a reward optimum -- which is why entropy bottoms at 0.650 (36M) and
  // drifts back UP to 0.700 over p1-validate's remaining 80M steps. Uniform
  // over the action mask reads 0.83 grounded / 0.95 airborne, so at 0.68 the
  // policy is still roughly half-random, and 42.9% of the 42 actions a
  // grounded car may choose press jump. The flip-spam is the prior showing
  // through, not a learned farm.
  float entropyScale = 0.002f;

  float gaeGamma = 0.99f;

  // Upstream default; the inherited 1.5e-4 compounded the epochs=1 problem.
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
