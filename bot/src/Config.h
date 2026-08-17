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
// The curriculum must agree with the reward function. Spawning air dribble
// scenarios while the reward pays nothing for air dribbling does not teach air
// dribbling -- it just spends samples on a situation the policy has no gradient
// to improve at, and hands it a stream of episodes it can only fail.
//
// These weights are matched to RewardPhase::Foundations. When the reward phase
// advances, raise the corresponding scenarios with it.
struct CurriculumWeights {
	float neutralPlay = 45.f;   // Ordinary play. Keep this dominant.
	float defend = 15.f;

	// Cars end up airborne whether or not we spawn them there, and landing
	// on your wheels is a prerequisite for everything else.
	float recover = 8.f;

	// Ball in the air, cars on the ground. A phase-1 scenario despite the
	// name: the lesson is "the ball is up, go and meet it", which the touch
	// reward pays for directly. Jumping emerges from this rather than from
	// being paid to leave the ground.
	float aerial = 5.f;

	// Close ball control on the deck.
	float groundDribble = 5.f;

	// Full kickoffs mixed into general training. The general policy has to play
	// the seconds right after the kickoff model hands over, so it needs to have
	// seen post-kickoff states.
	float kickoff = 8.f;

	// --- Later phases -------------------------------------------------------
	// Zero until the reward function pays for them. CombinedState drops
	// zero-weight setters entirely, so these cost nothing while disabled.
	float airDribble = 0.f;   // RewardPhase::Aerial
	float flipReset = 0.f;    // RewardPhase::Aerial
	float demo = 0.f;         // RewardPhase::Teamplay
};

// ---------------------------------------------------------------------------
// Rewards
// ---------------------------------------------------------------------------
// Read docs/rewards.md before changing any of these. The weights are derived
// from measured per-step reward values, not guessed, and the derivation is what
// keeps the bot from farming instead of playing.
//
// THE RULE THAT MATTERS: never pay per-step for a state the policy fully
// controls. Such a reward is an income stream that requires no skill, and PPO
// will find it. A previous version of this file paid 0.25/step for being
// airborne; the policy duly spent 87% of its life in the air, 98.4% of all
// reward came from shaping terms, and its skill rating fell steadily while the
// reward curve climbed. Every reward below is instead one of:
//
//   TELESCOPING  -- the derivative of a potential, so any closed path in state
//                   space sums to zero. VelocityPlayerToBall is exactly
//                   -d(distance to ball)/dt, so approaching and retreating nets
//                   nothing. Safe at any weight.
//   BOUNDED      -- total obtainable value is capped by a real resource.
//                   PickupBoost is a sqrt-delta, so it pays out at most 1.0 for
//                   an empty-to-full tank and almost nothing for topping up.
//   IMPULSE      -- requires an actual change in the world. StrongTouch needs
//                   the ball's velocity to change, so resting against the ball
//                   pays zero no matter how long you lean on it.
//   TERMINAL     -- the thing you actually want. Goals.
//
// Phases. Rewards are staged: teach one idea, let it consolidate, then add the
// next. Introducing dribble or demo rewards before the bot can reliably strike
// the ball just gives it more ways to earn without improving.
enum class RewardPhase {
	// Get to the ball, hit it hard, hit it towards their goal. Nothing else.
	Foundations,

	// Not yet designed. See docs/rewards.md for what each should add and the
	// measured trigger for advancing. Do not guess at these -- derive them from
	// the run that precedes them, the way Foundations was.
	Possession,
	Aerial,
	Teamplay,
};

// Phase 1: Foundations.
//
// Five components. The bot cannot reliably touch the ball yet, so anything
// beyond "reach it, strike it, aim it" is noise. Composition at a competent
// early policy is roughly 68% outcome / 32% shaping, and the shaping half is
// entirely telescoping or bounded, so the farm ceiling sits near zero.
struct RewardWeights {
	// TELESCOPING, and the backbone of this phase.
	//
	// It looks like a mere shaping term, but in phase 1 it is the car-control
	// curriculum expressed as a reward: to earn it the bot must land on its
	// wheels, orient itself, and drive somewhere deliberate. A fresh policy is
	// airborne and tumbling ~89% of the time, and nothing else here pays for
	// fixing that.
	//
	// This was set to 0.75 on the theory that a large approach reward produces
	// a ball-chaser. Measured over 20M steps that was wrong in a way worth
	// recording: the raw value collapsed from 0.146 to 0.017 and touch rate
	// halved -- the bot simply stopped driving at the ball, and with nothing
	// else dense enough to bootstrap from, learning stalled. Ball-chasing is a
	// phase-2 problem; being unable to reach the ball is a phase-1 one.
	//
	// A high weight is safe here in a way it would not be for an occupancy
	// reward: this is exactly -d(distance to ball)/dt, so any path that returns
	// to its starting point sums to zero. There is nothing to farm, only a bias
	// towards being near the ball.
	float velPlayerToBall = 3.f;

	// IMPULSE. The main outcome signal and the largest single term. Rewards
	// changing the ball's velocity, scaled 0..1 between a gentle nudge and a
	// solid strike, so it teaches "hit it" and "hit it hard" with one gradient.
	float strongTouch = 30.f;

	// Directional outcome: is the ball going the right way? Zero-sum, so it
	// measures progress against the opponent rather than absolute ball motion.
	float velBallToGoal = 3.5f;

	// TERMINAL. Sized against the discount horizon, not against its share of
	// average reward. At gaeGamma 0.99 and 15 steps/sec the horizon is ~100
	// steps (~7s), over which good play accrues ~0.3/step for a total future
	// value near 30. A goal worth 30 therefore roughly doubles the value of the
	// moment it happens -- strong, but not so large it drowns the dense signal
	// that makes early learning possible. Conceding costs the same.
	float goal = 30.f;

	// BOUNDED. A nudge towards collecting boost rather than driving past it.
	// Small on purpose: boost is already instrumentally valuable because it
	// gets you to the ball faster, which the approach reward pays for. This
	// only breaks ties.
	float pickupBoost = 4.f;
};

// Kickoff: from reset to first touch, so roughly 20-40 steps.
//
// Only one thing happens in a kickoff, so the reward is dominated by that one
// thing. Note there is deliberately no boost-conservation term: the episode can
// be extended to its timeout by simply not going for the ball, which would turn
// any per-step term into a reward for dodging the kickoff entirely.
struct KickoffRewardWeights {
	// Denser than in general play -- the episode is far too short for a sparse
	// signal to carry it.
	float velPlayerToBall = 1.f;

	// Winning the touch is essentially the whole objective.
	float strongTouch = 40.f;

	// Which way the ball went afterwards. Only measurable because
	// FirstTouchCondition holds the episode open briefly past contact.
	float velBallToGoal = 6.f;

	// Kickoff goals are rare but real. Same scale as general play.
	float goal = 30.f;
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
	// Which staged reward set to train with. Advance this only on the measured
	// triggers in docs/rewards.md, and start a fresh run when you do -- a
	// policy carried across a reward change is optimising a different objective
	// than the one it was trained on.
	RewardPhase rewardPhase = RewardPhase::Foundations;

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
