#include "BallPredictor.h"

using namespace RLGC;

namespace Dash {

BallPredictor::BallPredictor() {
	// Car-less: a ball-only arena is far cheaper to step than a full one, and
	// cars are exactly what we are predicting the absence of.
	arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);

	traj.pos.resize(SIM_HORIZON_TICKS + 1);
	traj.vel.resize(SIM_HORIZON_TICKS + 1);
}

BallPredictor::~BallPredictor() {
	delete arena;
	arena = nullptr;
}

void BallPredictor::Reset() {
	hasCache = false;
}

bool BallPredictor::CacheValidFor(const RLGC::GameState& state) const {
	if (!hasCache)
		return false;

	// A rewound clock means a new episode.
	if (state.lastTickCount < traj.startTick)
		return false;

	const uint64_t offset = state.lastTickCount - traj.startTick;

	// Keep enough runway for the deepest sample.
	if (offset + SAMPLE_TICKS.back() > (uint64_t)SIM_HORIZON_TICKS)
		return false;

	// Did the ball actually go where we said it would?
	const RocketSim::Vec posErr = state.ball.pos - traj.pos[offset];
	if (posErr.Length() > POS_TOLERANCE)
		return false;

	const RocketSim::Vec velErr = state.ball.vel - traj.vel[offset];
	if (velErr.Length() > VEL_TOLERANCE)
		return false;

	return true;
}

void BallPredictor::Simulate(const RLGC::GameState& state) {
	RocketSim::BallState bs = arena->ball->GetState();
	bs.pos = state.ball.pos;
	bs.vel = state.ball.vel;
	bs.angVel = state.ball.angVel;
	arena->ball->SetState(bs);

	traj.startTick = state.lastTickCount;
	traj.bounceTick = -1;
	traj.bouncePos = {};
	traj.goalTick = -1;
	traj.goalTeam = -1;

	traj.pos[0] = bs.pos;
	traj.vel[0] = bs.vel;

	for (int i = 1; i <= SIM_HORIZON_TICKS; i++) {
		arena->Step(1);
		const RocketSim::BallState cur = arena->ball->GetState();
		traj.pos[i] = cur.pos;
		traj.vel[i] = cur.vel;
	}

	hasCache = true;
	simCount++;
}

const BallTrajectory& BallPredictor::Get(const RLGC::GameState& state) {
	if (!CacheValidFor(state))
		Simulate(state);
	return traj;
}

}  // namespace Dash
