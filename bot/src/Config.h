#pragma once

#include "env/Obs.h"
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
// Every reward weight in this project is declared as a BUDGET and converted to
// a per-step weight in exactly one place (Hive::GeneralRewardSpecs). Nothing
// may be written as a bare per-step float.
//
// This exists because p1air's `grounded = 0.05` integrated to 9.0 units per
// episode -- nine goals per episode for holding still on the wheels -- and
// nobody noticed, because nobody wrote down the integral. Declaring the
// integral makes that class of mistake unrepresentable.
//
// THE UNIT IS ONE BALL TOUCH, not a goal. A goal was the unit until
// p7approach and it could not be audited: goals arrive 0.116 times per episode
// and 49% of those are `Scenario/Defend` conceding rather than anyone scoring,
// so the ledger had to be reconstructed by hand in every post-mortem. A touch
// occurs 0.16-2 times per episode and is read straight off `Touch/Edge Rate`
// and `Player/Ball Touch Ratio`.
//
// Restating the previous stack in the new unit is what showed the problem.
// p7approach paid 1.83 touch-units for a whole episode of PERFECT approach;
// the guide's stack pays 20.5. An 11x difference that neither spec noticed,
// because the two were denominated in different currencies.

// tickSkip 8 at RocketSim's 120 Hz.
inline constexpr float STEPS_PER_SECOND = 15.f;

// MEASURED, not assumed. p6budget's `Episode/Mean Steps` read 171.0 over 100M
// steps and p7approach's read 166.6. noTouchTimeout caps a never-touching bot
// at 180 steps and goals end episodes early; at a touch ratio near zero almost
// every episode runs to the timeout, which is why this sits just under it.
//
// Re-derive it once the bot can actually reach the ball: episodes will get
// LONGER (touches reset the no-touch timer) until goals start ending them,
// at which point they get much shorter. Both directions silently rescale every
// rate budget below.
inline constexpr float REFERENCE_EPISODE_SECONDS = 11.4f;
inline constexpr float REFERENCE_EPISODE_STEPS =
    STEPS_PER_SECOND * REFERENCE_EPISODE_SECONDS;

// Touch-units earned by holding a behaviour perfectly for one reference
// episode, converted to the per-step weight that earns it.
inline constexpr float RateWeight(float budgetPerEpisode) {
  return budgetPerEpisode / REFERENCE_EPISODE_STEPS;
}

// Touch-units of cost for one second of a condition, converted to per-step.
inline constexpr float PerSecondWeight(float budgetPerSecond) {
  return budgetPerSecond / STEPS_PER_SECOND;
}

// All values are touch-units. This is the early-stage stack from Zealan's
// RLGym-PPO-Guide (making_a_good_bot.md) in the guide's own proportions --
// touch 50, speed-to-ball 5, face-ball 1, air 0.15 -- divided through by the
// touch weight so that a touch is 1.0, then expressed as episode integrals.
//
// Nothing here is this project's invention, deliberately. See
// docs/superpowers/specs/2026-08-18-known-good-baseline-design.md.
struct RewardBudget {
  // --- Event: earned per occurrence ---

  // THE UNIT: one full-power ball touch, 1.0 by definition. Scales with hit
  // force (|delta ball velocity|) and pays exactly ZERO below RLGymCPP's
  // 20 kph floor, which is 555.6 uu/s -- an order of magnitude above the tens
  // of uu/s a dribble carry imparts. Saturates at 130 kph = 3611 uu/s.
  //
  // Realized values will be well under 1.0: p1pay measured typical hit force
  // around 1400 uu/s, which scores 0.39. That is fine and expected -- the
  // budget is what scales it, and `Touch/Strong Value` is what measures it.
  //
  // This replaces p9rel's flat per-step touch, which the bot farmed by
  // carrying the ball for 13% of all steps. It is the guide's middle-stage
  // prescription verbatim: "scale the reward with the strength of the touch...
  // a slight push that barely changes the velocity of the ball will give
  // almost no reward, but a strong shot will give lots of reward."
  //
  // PROVISIONAL. Per roadmap D6 (no magic numbers without measurement), this
  // stands until `Touch/Hit Force` says what a realized touch is actually
  // worth; the first run that reports it is what sets the final value.
  float strongTouch = 1.0f;

  // Arriving at the ball, worth a quarter of a full-power hit. Rising edge, so
  // carrying pays this exactly once (see TouchEdgeReward). Kept small and
  // non-zero because the bot still needs a signal for REACHING the ball --
  // deleting it entirely would leave nothing paying for contact at all until
  // the bot can already hit hard.
  float touchEdge = 0.25f;

  // --- Rate: earned by holding the behaviour for one reference episode ---

  // 5/50 per step over 171 steps. The dominant term by an order of magnitude,
  // which is the point: a bot that cannot reach the ball needs approach paid
  // far more heavily than contact.
  float speedToBall = 17.1f;

  // Exactly speedToBall/5, the guide's ratio. SIGNED, so driving backwards at
  // the ball is punished rather than merely unpaid -- see the pairing note in
  // Rewards.h.
  float faceBall = 3.42f;

  // 0.15/50 per step over 171 steps: 2.5% of the dense budget. Measured to be
  // ~50x too small to pay for the traction and contact a jump costs (p8ref
  // ledger), so it does NOT keep jumping alive and is not expected to. Raising
  // it enough to break even would put ~37% of the budget on floating, which is
  // p1air's do-nothing attractor inverted. Air is a problem for an air-TOUCH
  // term, not for this one.
  float air = 0.513f;
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

  // Where episodes start from. `Random` is RLGymCPP's RandomState with
  // (randBallSpeed, randCarSpeed, carsOnGround) = (true, true, false), which
  // is what Zealan's guide specifies and what every reference bot starts on:
  // fully random positions and velocities, cars airborne on half of spawns.
  //
  // `Curriculum` is this project's 10-entry scenario mix. It has never been
  // validated against anything, and it is the leading suspect for the
  // Early/Late collapse that has stood since p1age -- Early(0-1s) touch rate
  // 0.0081 against Late(4s+) 0.00025, a 32x gap, with the bot ending FARTHER
  // from the ball than it started. A spawn distribution that hands the bot a
  // favourable first second teaches the first second.
  //
  // The curriculum code stays in the tree; phase B measures it against
  // Random rather than assuming it.
  enum class SpawnMode { Random, Curriculum };
  SpawnMode spawn = SpawnMode::Random;

  CurriculumWeights curriculum = {};

  // Whether the action parser masks actions by situation. See Actions.h: the
  // mask more than doubles the grounded jump prior (42.9% vs 20%) relative to
  // every reference implementation. Must match at deployment.
  bool maskActions = false;

  // Which observation the policy sees. `Relative` is `Default` plus car-frame
  // relative geometry for the ball and every other car (see RelativeObs.h);
  // `Default` is RLGymCPP's DefaultObsPadded, which p8ref ran on.
  //
  // Changing this changes the observation WIDTH, so it invalidates every
  // existing checkpoint. Must match at deployment -- a width mismatch throws
  // at load, but a same-width layout mismatch would not, so `verify` checks it
  // explicitly.
  ObsMode obs = ObsMode::Relative;

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
  // Guide, learner_settings.md: "values of 50_000 are good for early learning,
  // but once the bot is actually hitting the ball, it should be increased to
  // 100_000". Halving it from 100k also doubles the number of policy updates
  // per unit of experience, which is what an under-updating run needs.
  int tsPerItr = 50'000;

  // Minibatch is the main VRAM knob (6 GB available).
  int miniBatchSize = 25'000;

  // Upstream default; guide recommends 2 or 3.
  int epochs = 2;

  // NOT the guide's 0.01. GigaLearn's entropy is NORMALIZED to [0,1] (divided
  // by log(numActions)), so this scale is not comparable to rlgym-ppo's, and
  // porting the number verbatim ports a value this project has measured
  // breaking three times.
  //
  // Evidence: 0.035 and 0.018 pinned entropy near-uniform (p1probe-d, -g);
  // 0.01 pinned it again in p7approach (`Policy Entropy` 0.709 -> 0.689 flat
  // over 100M steps, `SB3 Clip Fraction` 0.0038 against a healthy 0.05-0.2,
  // `Policy Relative Entropy Loss` 268 against a documented healthy <=0.1);
  // 0.002 produced this project's only breakthrough (p1probe-f) and the RUNLOG
  // says "Keep 0.002".
  //
  // Exploration does not depend on this alone: the fourth external patch
  // floors every VALID action at 0.02/N probability, which is the only thing
  // that has ever reversed an action extinction.
  float entropyScale = 0.002f;

  // 0.99 at 15 steps/s is a 4.6 s half-life for the VALUE horizon (1/(1-gamma)
  // = 100 steps = 6.7 s). The guide calls this fine for early and middle
  // stages and recommends raising it to a ~15 s half-life later; that is a
  // phase-A change, not this one.
  float gaeGamma = 0.99f;

  // Guide, learner_settings.md: "Bot that can't score yet: 2e-4".
  float policyLR = 2e-4f;
  float criticLR = 2e-4f;

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
