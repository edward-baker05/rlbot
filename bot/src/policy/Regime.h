#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <array>
#include <cstdint>

namespace Hive {

// ============================================================================
// Regime -- which of the two models drives the car
// ============================================================================
// The bot runs exactly two policies:
//
//   Kickoff  -- from the kickoff reset until the ball is first touched.
//   General  -- everything else.
//
// This is a partition, not a mixture-of-experts. The distinction matters. A
// general MoE router has to guess at fuzzy boundaries mid-play, and each expert
// only ever trains on its own slice, so neither learns to recover from states
// the other hands it. Those seams are where MoE bots fall apart in a game with
// continuous dynamics.
//
// Kickoff avoids all of that:
//   * It begins at a hard environment reset, from a small set of fixed spawns.
//   * It ends at an unambiguous, observable event -- first ball contact.
//   * The two regimes never interleave; control transfers once, in one
//     direction, at a moment both policies were trained to expect.
//
// So the split costs nothing in coherence and buys a policy that can specialise
// on the handful of deterministic opening positions, which is exactly where a
// generalist wastes capacity.
//
// If you ever find yourself wanting a third regime, re-read the paragraph above
// and check that it satisfies all three properties. Almost nothing else does.
// ============================================================================

enum class Regime : int {
	Kickoff = 0,
	General = 1,

	COUNT
};

constexpr int REGIME_COUNT = static_cast<int>(Regime::COUNT);

const char* RegimeName(Regime r);

// ----------------------------------------------------------------------------
// Kickoff detection
// ----------------------------------------------------------------------------
// Detecting a kickoff from a single frame is easy to get *almost* right, and
// almost-right is the failure mode that silently hands kickoff frames to the
// general model. Two things make a raw per-frame test unreliable:
//
//   1. The ball sits still at the centre spot for the whole 3-2-1 countdown,
//      but it also passes through the centre spot at speed during normal play.
//      Position alone is not enough; the ball must also be at rest.
//
//   2. Immediately after first touch the ball is still near centre for a few
//      ticks, so a position test keeps firing after the kickoff is over.
//
// KickoffTracker fixes both by latching: it enters kickoff only on a genuine
// reset (ball at rest, at centre) and stays latched until first touch or a
// timeout, regardless of what the ball does in between.

struct KickoffThresholds {
	float ballSpeed = 5.f;       // At rest. Real kickoffs are exactly zero.
	float ballRadius = 100.f;    // Distance from centre spot in the XY plane.
	float ballHeight = 130.f;    // Ball is on the ground, not passing overhead.
	float timeoutSeconds = 6.f;  // Give up if nobody touches it.
};

class KickoffTracker {
public:
	explicit KickoffTracker(const KickoffThresholds& t = {}) : t(t) {}

	// Feed every step. Returns the regime that should drive this step.
	// `deltaTime` is seconds since the previous call.
	Regime Update(const RLGC::GameState& state, float deltaTime);

	Regime Current() const { return inKickoff ? Regime::Kickoff : Regime::General; }

	// Call on episode reset / when the bot is (re)spawned.
	void Reset() {
		inKickoff = false;
		elapsed = 0.f;
	}

private:
	// True for a frame that looks like a kickoff has been set up.
	bool LooksLikeKickoffSpawn(const RLGC::GameState& state) const;

	KickoffThresholds t;
	bool inKickoff = false;
	float elapsed = 0.f;
};

// ----------------------------------------------------------------------------
// Play phases -- metrics and curriculum only, never routing
// ----------------------------------------------------------------------------
// These labels describe what the general policy is doing at a given moment.
// They exist so you can (a) see in wandb how often each situation occurs and how
// the policy performs in it, and (b) weight the training curriculum towards
// situations that are rare under random play.
//
// Deliberately NOT used to select a model. One general policy handles all of
// these; the labels are an observability tool, not a control-flow mechanism.

enum class PlayPhase : int {
	Aerial = 0,
	AirDribble,
	GroundDribble,
	Defend,
	Recover,
	Neutral,

	COUNT
};

constexpr int PLAY_PHASE_COUNT = static_cast<int>(PlayPhase::COUNT);

const char* PlayPhaseName(PlayPhase p);

struct PhaseThresholds {
	float airborneZ = 200.f;
	float aerialBallZ = 500.f;
	float airDribbleBallZ = 400.f;
	float ballNearDist = 350.f;
	float dribbleDist = 200.f;
	float dribbleBallZMax = 300.f;
	float defendThirdY = -1700.f;
};

PlayPhase ClassifyPhase(const RLGC::Player& player,
                        const RLGC::GameState& state,
                        const PhaseThresholds& t = {});

// Per-phase counters for training metrics.
struct PhaseCounts {
	std::array<int64_t, PLAY_PHASE_COUNT> counts = {};

	void Add(PlayPhase p) { counts[static_cast<int>(p)]++; }

	int64_t Total() const {
		int64_t total = 0;
		for (auto c : counts)
			total += c;
		return total;
	}
};

} // namespace Hive
