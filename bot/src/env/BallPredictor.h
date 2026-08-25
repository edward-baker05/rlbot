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

	static constexpr int NUM_SAMPLES = 6;

	// t = 0.15, 0.30, 0.55, 0.95, 1.60, 2.60 seconds, snapped to 120Hz ticks.
	// Geometric (ratio ~1.75): prediction accuracy decays with horizon, so the
	// dimensions are spent where the prediction is still true.
	static constexpr std::array<int, NUM_SAMPLES> SAMPLE_TICKS =
		{18, 36, 66, 114, 192, 312};

	// How far past the deepest sample to simulate, so the trajectory can be
	// consumed for a while before it runs out of runway.
	//
	// Sized by measurement. Each re-simulation costs the whole horizon and
	// buys E[min(runway, time-to-next-touch)] ticks of use, so the amortized
	// cost is (deepest + runway) / E[min(runway, T)] -- a curve with a genuine
	// interior minimum, since a short runway re-simulates constantly and a long
	// one simulates tail that a touch throws away unread.
	//
	// predict-bench, on a 3600 under random inputs (T ~ 600 ticks):
	//
	//   runway 300 -> 21.3 ball-only ticks per env-step  (50.9% loss)
	//   runway 408 -> 19.8                               (35-44% loss)
	//
	// The minimum sits near runway 550 and is worth about 3% over 408, which is
	// inside this benchmark's run-to-run spread. 408 it stays -- the 6-second
	// horizon the design started with turns out to have been the right call.
	//
	// Note this is the SHALLOW end of the cost: a trained policy touches the
	// ball far more often than random inputs do, which shortens T and pushes
	// the optimum runway down. Re-measure if the sample schedule changes.
	static constexpr int CACHE_RUNWAY_TICKS = 408;

	static constexpr int SIM_HORIZON_TICKS =
		SAMPLE_TICKS.back() + CACHE_RUNWAY_TICKS;

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
