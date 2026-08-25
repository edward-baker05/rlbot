#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Dash {

// A tick-indexed ball trajectory, sampled at RocketSim's fixed 120Hz.
// Index i is the predicted state i ticks after `startTick`; index 0 is the
// present.
struct BallTrajectory {
	std::vector<RocketSim::Vec> pos;
	std::vector<RocketSim::Vec> vel;
	uint64_t startTick = 0;

	// Tick offsets from startTick, or -1 for "did not happen in the horizon".
	int bounceTick = -1;
	RocketSim::Vec bouncePos = {};
	int goalTick = -1;
	int goalTeam = -1;  // 0 = blue's net, 1 = orange's net, -1 = none
};

// Predicts where the ball goes if nobody touches it.
//
// The trajectory is only invalidated by a touch (or a reset), so a single
// simulation is reused across many env steps. Rather than plumbing touch
// events through, invalidation is detected by comparing the live ball against
// this predictor's own prediction for the current tick: any divergence beyond
// float noise means something interfered. That covers touches, goal resets and
// state setters uniformly, with no event wiring to get out of sync.
class BallPredictor {
public:
	static constexpr int TICK_RATE = 120;

	// 6 seconds. Deliberately more than the deepest sample (2.6s) so the
	// trajectory can be consumed for several seconds before needing a redo.
	static constexpr int SIM_HORIZON_TICKS = 720;

	static constexpr int NUM_SAMPLES = 6;

	// t = 0.15, 0.30, 0.55, 0.95, 1.60, 2.60 seconds, snapped to 120Hz ticks.
	// Geometric (ratio ~1.75): prediction accuracy decays with horizon, so the
	// dimensions are spent where the prediction is still true.
	static constexpr std::array<int, NUM_SAMPLES> SAMPLE_TICKS =
		{18, 36, 66, 114, 192, 312};

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

	// Returns a trajectory valid for `state`, re-simulating only if needed.
	// The reference is invalidated by the next call.
	const BallTrajectory& Get(const RLGC::GameState& state);

	// Drops the cache. Call on episode reset.
	void Reset();

	// Number of full simulations run. For the performance gate and tests.
	uint64_t SimulationCount() const { return simCount; }

private:
	void Simulate(const RLGC::GameState& state);
	bool CacheValidFor(const RLGC::GameState& state) const;

	RocketSim::Arena* arena = nullptr;
	BallTrajectory traj = {};
	bool hasCache = false;
	uint64_t simCount = 0;
};

}  // namespace Dash
