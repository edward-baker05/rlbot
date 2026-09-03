#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Dash {

// A ball trajectory over a fixed window, sampled at RocketSim's fixed 120Hz.
// Offset 0 is always the present; offset i is the predicted state i ticks from
// now. Storage is a ring buffer, so sliding the window forward costs nothing.
struct BallTrajectory {
	std::vector<RocketSim::Vec> posRing;
	std::vector<RocketSim::Vec> velRing;
	int head = 0;
	uint64_t startTick = 0;

	// Tick offsets from the present, or -1 for "not in the window".
	int bounceTick = -1;
	RocketSim::Vec bouncePos = {};
	int goalTick = -1;
	int goalTeam = -1;  // 0 = blue's net, 1 = orange's net, -1 = none

	int Size() const { return (int)posRing.size(); }

	int Wrap(int offset) const {
		const int i = head + offset;
		return i < Size() ? i : i - Size();
	}

	const RocketSim::Vec& PosAt(int offset) const { return posRing[Wrap(offset)]; }
	const RocketSim::Vec& VelAt(int offset) const { return velRing[Wrap(offset)]; }
};

// Predicts where the ball goes if nobody touches it.
//
// The window is pinned to the present: advancing by d ticks drops d states off
// the front and simulates d new ones onto the back, so an untouched ball costs
// exactly tickSkip ball-only ticks per env step. A touch invalidates everything
// and forces a full re-simulation of the window.
//
// Rather than plumbing touch events through, invalidation is detected by
// comparing the live ball against this predictor's own prediction for the
// current tick: any divergence beyond float noise means something interfered.
// That covers touches, goal resets and state setters uniformly.
class BallPredictor {
public:
	static constexpr int TICK_RATE = 120;

	static constexpr int NUM_SAMPLES = 6;

	// t = 0.15, 0.30, 0.55, 0.95, 1.60, 2.60 seconds, snapped to 120Hz ticks.
	// Geometric (ratio ~1.75): prediction accuracy decays with horizon, so the
	// dimensions are spent where the prediction is still true.
	static constexpr std::array<int, NUM_SAMPLES> SAMPLE_TICKS =
		{18, 36, 66, 114, 192, 312};

	// Offsets 0 through the deepest sample. Nothing deeper is ever read, and
	// with a sliding window nothing deeper needs to be simulated either.
	static constexpr int WINDOW_TICKS = SAMPLE_TICKS.back() + 1;

	// Divergence beyond this means the ball was interfered with. RocketSim is
	// deterministic, so an untouched ball matches to float precision; these are
	// far above that noise floor and far below any real touch.
	static constexpr float POS_TOLERANCE = 1.0f;   // uu
	static constexpr float VEL_TOLERANCE = 1.0f;   // uu/s

	// A bounce is a tick where velocity direction turns sharply or speed jumps.
	// Deliberately not "velocity minus gravity*dt": the ball has drag
	// (MutatorConfig::ballDrag), so free flight is not exactly ballistic and an
	// exact test would need to track that constant. Direction change needs no
	// physics constants and works for floor, wall, ceiling and corner alike.
	static constexpr float BOUNCE_COS_THRESHOLD = 0.966f;  // ~15 degrees
	static constexpr float BOUNCE_SPEED_JUMP = 100.f;      // uu/s in one tick

	// Below this speed, velocity direction is noise; a resting ball must not
	// register bounces.
	static constexpr float BOUNCE_MIN_SPEED = 50.f;        // uu/s

	BallPredictor();
	~BallPredictor();

	BallPredictor(const BallPredictor&) = delete;
	BallPredictor& operator=(const BallPredictor&) = delete;

	// Returns a trajectory whose offset 0 is `state`, re-simulating or sliding
	// as needed. The reference is invalidated by the next call.
	const BallTrajectory& Get(const RLGC::GameState& state);

	// Drops the cache. Call on episode reset.
	void Reset();

	// Full window re-simulations. For the performance gate and tests.
	uint64_t SimulationCount() const { return simCount; }

	// Ball-only arena ticks stepped, whether by re-simulation or by sliding.
	// This is the real cost of the predictor.
	uint64_t SimulatedTickCount() const { return tickCount; }

private:
	void Simulate(const RLGC::GameState& state);
	void Advance(int ticks);
	void AppendTick();
	void RefreshBounce();
	bool CacheUsableFor(const RLGC::GameState& state, int& outOffset) const;

	RocketSim::Arena* arena = nullptr;
	BallTrajectory traj = {};

	// Per-tick "this tick is a bounce" flags, in the same ring layout as traj.
	std::vector<uint8_t> bounceRing;

	// Set once a goal is inside the window: past the goal line the simulation
	// is meaningless, so the tail is held still rather than fed to the network
	// as an imaginary continuation.
	bool frozen = false;

	bool hasCache = false;
	uint64_t simCount = 0;
	uint64_t tickCount = 0;
};

}  // namespace Dash
