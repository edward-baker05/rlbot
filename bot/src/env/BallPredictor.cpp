#include "BallPredictor.h"

#include <cmath>

using namespace RLGC;

namespace Dash {

namespace {

bool IsBounce(const RocketSim::Vec& prevVel, const RocketSim::Vec& vel) {
	const float prevSpeed = prevVel.Length();
	const float speed = vel.Length();

	if (prevSpeed <= BallPredictor::BOUNCE_MIN_SPEED ||
	    speed <= BallPredictor::BOUNCE_MIN_SPEED)
		return false;

	const float cosAngle = prevVel.Dot(vel) / (prevSpeed * speed);
	return cosAngle < BallPredictor::BOUNCE_COS_THRESHOLD ||
	       std::abs(speed - prevSpeed) > BallPredictor::BOUNCE_SPEED_JUMP;
}

}  // namespace

BallPredictor::BallPredictor() {
	// Car-less: a ball-only arena is far cheaper to step than a full one, and
	// cars are exactly what we are predicting the absence of.
	arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);

	traj.posRing.resize(WINDOW_TICKS);
	traj.velRing.resize(WINDOW_TICKS);
	bounceRing.resize(WINDOW_TICKS);
}

BallPredictor::~BallPredictor() {
	delete arena;
	arena = nullptr;
}

void BallPredictor::Reset() {
	hasCache = false;
}

bool BallPredictor::CacheUsableFor(const RLGC::GameState& state,
                                   int& outOffset) const {
	if (!hasCache)
		return false;

	// A rewound clock means a new episode.
	if (state.lastTickCount < traj.startTick)
		return false;

	const uint64_t delta = state.lastTickCount - traj.startTick;
	if (delta >= (uint64_t)traj.Size())
		return false;

	const int offset = (int)delta;

	// Sliding this far would push the goal out of the front of the window,
	// leaving the whole window frozen on a goal that has already happened.
	if (frozen && traj.goalTick < offset)
		return false;

	// Did the ball actually go where we said it would?
	if ((state.ball.pos - traj.PosAt(offset)).Length() > POS_TOLERANCE)
		return false;

	if ((state.ball.vel - traj.VelAt(offset)).Length() > VEL_TOLERANCE)
		return false;

	outOffset = offset;
	return true;
}

void BallPredictor::Simulate(const RLGC::GameState& state) {
	RocketSim::BallState bs = arena->ball->GetState();
	bs.pos = state.ball.pos;
	bs.vel = state.ball.vel;
	bs.angVel = state.ball.angVel;
	arena->ball->SetState(bs);

	traj.head = 0;
	traj.startTick = state.lastTickCount;
	traj.goalTick = -1;
	traj.goalTeam = -1;
	frozen = false;

	traj.posRing[0] = bs.pos;
	traj.velRing[0] = bs.vel;
	bounceRing[0] = 0;

	const int n = traj.Size();
	for (int i = 1; i < n; i++) {
		arena->Step(1);
		tickCount++;

		const RocketSim::BallState cur = arena->ball->GetState();
		traj.posRing[i] = cur.pos;
		traj.velRing[i] = cur.vel;
		bounceRing[i] = IsBounce(traj.velRing[i - 1], cur.vel) ? 1 : 0;

		if (arena->IsBallScored()) {
			frozen = true;
			traj.goalTick = i;
			// RS_TEAM_FROM_Y's convention: the net the ball crossed into.
			traj.goalTeam = cur.pos.y > 0 ? 1 : 0;

			for (int j = i + 1; j < n; j++) {
				traj.posRing[j] = cur.pos;
				traj.velRing[j] = {0, 0, 0};
				bounceRing[j] = 0;
			}
			break;
		}
	}

	RefreshBounce();
	hasCache = true;
	simCount++;
}

void BallPredictor::AppendTick() {
	// The slot holding the present is about to fall off the front, so it is
	// also the slot the new tail state goes into.
	const int slot = traj.head;
	const int prevSlot = slot == 0 ? traj.Size() - 1 : slot - 1;
	const RocketSim::Vec prevVel = traj.velRing[prevSlot];

	RocketSim::Vec pos, vel;
	bool scored = false;

	if (frozen) {
		pos = traj.posRing[prevSlot];
		vel = {0, 0, 0};
	} else {
		arena->Step(1);
		tickCount++;

		const RocketSim::BallState cur = arena->ball->GetState();
		pos = cur.pos;
		vel = cur.vel;
		scored = arena->IsBallScored();
	}

	traj.head = slot + 1 == traj.Size() ? 0 : slot + 1;
	traj.startTick++;

	traj.posRing[slot] = pos;
	traj.velRing[slot] = vel;
	bounceRing[slot] = IsBounce(prevVel, vel) ? 1 : 0;

	if (traj.goalTick >= 0)
		traj.goalTick--;

	if (scored) {
		frozen = true;
		traj.goalTick = traj.Size() - 1;
		traj.goalTeam = pos.y > 0 ? 1 : 0;
	}
}

void BallPredictor::RefreshBounce() {
	traj.bounceTick = -1;
	traj.bouncePos = {};

	for (int i = 1; i < traj.Size(); i++) {
		if (bounceRing[traj.Wrap(i)]) {
			traj.bounceTick = i;
			traj.bouncePos = traj.PosAt(i);
			break;
		}
	}
}

void BallPredictor::Advance(int ticks) {
	for (int i = 0; i < ticks; i++)
		AppendTick();

	RefreshBounce();
}

const BallTrajectory& BallPredictor::Get(const RLGC::GameState& state) {
	int offset = 0;

	if (!CacheUsableFor(state, offset))
		Simulate(state);
	else if (offset > 0)
		Advance(offset);

	return traj;
}

}  // namespace Dash
