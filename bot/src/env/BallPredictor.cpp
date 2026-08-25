#include "BallPredictor.h"

#include <cmath>

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

		if (traj.bounceTick < 0) {
			const RocketSim::Vec& prevVel = traj.vel[i - 1];
			const float prevSpeed = prevVel.Length();
			const float curSpeed = cur.vel.Length();

			if (prevSpeed > BOUNCE_MIN_SPEED && curSpeed > BOUNCE_MIN_SPEED) {
				const float cosAngle =
					prevVel.Dot(cur.vel) / (prevSpeed * curSpeed);
				const bool turned = cosAngle < BOUNCE_COS_THRESHOLD;
				const bool jumped =
					std::abs(curSpeed - prevSpeed) > BOUNCE_SPEED_JUMP;

				if (turned || jumped) {
					traj.bounceTick = i;
					traj.bouncePos = cur.pos;
				}
			}
		}

		if (traj.goalTick < 0 && arena->IsBallScored()) {
			traj.goalTick = i;
			// RS_TEAM_FROM_Y's convention: the net the ball crossed into.
			traj.goalTeam = cur.pos.y > 0 ? 1 : 0;

			// Stop here. Past the goal line the simulation is meaningless --
			// the real game would have reset -- so freeze the remainder rather
			// than feeding the network an imaginary continuation.
			for (int j = i + 1; j <= SIM_HORIZON_TICKS; j++) {
				traj.pos[j] = cur.pos;
				traj.vel[j] = {0, 0, 0};
			}
			break;
		}
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
