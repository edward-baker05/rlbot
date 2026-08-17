#include "StateSetters.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>

using namespace RLGC;
using RocketSim::Math::RandFloat;

namespace Hive {

// Sign convention: +1 means "this team attacks +Y" (blue), -1 means "attacks
// -Y" (orange). Multiply any Y coordinate written from blue's perspective by
// this to mirror it for orange.
static inline float TeamSign(Team team) {
	return (team == Team::BLUE) ? 1.f : -1.f;
}

static inline Vec RandUnitVec() {
	// Rejection-sample inside the unit sphere so directions are uniform.
	// Normalising a uniform cube biases towards the corners.
	for (int i = 0; i < 16; i++) {
		Vec v = RLGC::Math::RandVec(Vec(-1, -1, -1), Vec(1, 1, 1));
		const float len = v.Length();
		if (len > 0.05f && len <= 1.f)
			return v / len;
	}
	return Vec(1, 0, 0);
}

// Point a car at a world-space target, with optional roll/pitch noise.
static inline RotMat LookAt(const Vec& from, const Vec& to, float noise = 0.f) {
	const Vec d = to - from;
	const float yaw = std::atan2(d.y, d.x);
	const float pitch = std::atan2(d.z, std::sqrt(d.x * d.x + d.y * d.y));
	Angle a = Angle(yaw + RandFloat(-noise, noise),
	                pitch + RandFloat(-noise, noise),
	                RandFloat(-noise, noise));
	return a.ToRotMat();
}

// Place a car flat on the ground at (x, y) facing a target.
static inline void PutOnGround(CarState& cs, float x, float y, const Vec& lookTarget, float speed) {
	cs.pos = Vec(x, y, 17.f);
	Angle a = Angle(std::atan2(lookTarget.y - y, lookTarget.x - x), 0, 0);
	cs.rotMat = a.ToRotMat();
	cs.vel = cs.rotMat.forward * speed;
	cs.angVel = Vec(0, 0, 0);
	cs.isOnGround = true;
}

// ----------------------------------------------------------------------------

void AerialState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandFloat(-2800, 2800), RandFloat(-3500, 3500), RandFloat(minBallZ, maxBallZ));
	bs.vel = Vec(RandFloat(-500, 500), RandFloat(-500, 500), RandFloat(-200, 500));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		// Spawn on the ground a workable distance from the ball so the policy
		// has to actually drive-and-jump rather than start already underneath.
		const float dist = RandFloat(1200, 2600);
		const float ang = RandFloat(-M_PI, M_PI);
		PutOnGround(cs,
		            std::clamp(bs.pos.x + std::cos(ang) * dist, -3800.f, 3800.f),
		            std::clamp(bs.pos.y + std::sin(ang) * dist, -4800.f, 4800.f),
		            bs.pos,
		            RandFloat(0, 1200));
		cs.boost = RandFloat(minBoost, maxBoost);
		car->SetState(cs);
	}
}

void AirDribbleState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	const float z = RandFloat(minZ, maxZ);
	const Vec sharedVel = Vec(RandFloat(-700, 700), RandFloat(-300, 1400), RandFloat(200, 700));

	BallState bs = {};
	bs.pos = Vec(RandFloat(-2500, 2500), RandFloat(-3000, 3000), z);
	bs.vel = sharedVel;
	arena->ball->SetState(bs);

	// The first car gets the ball; everyone else is scattered so they do not
	// spawn inside it.
	bool first = true;
	for (Car* car : arena->_cars) {
		CarState cs = {};
		if (first) {
			first = false;
			// Just under the ball, moving with it -- the state you are in
			// immediately after a successful pop.
			cs.pos = bs.pos - Vec(0, 0, CommonValues::BALL_RADIUS + 60.f);
			cs.vel = sharedVel + Vec(RandFloat(-80, 80), RandFloat(-80, 80), RandFloat(-40, 40));
			cs.rotMat = LookAt(cs.pos, bs.pos + sharedVel, 0.15f);
			cs.boost = RandFloat(50, 100);
			cs.isOnGround = false;
			cs.hasJumped = true;
			cs.hasDoubleJumped = true; // No second jump: force air-roll control
		} else {
			PutOnGround(cs, RandFloat(-3500, 3500), RandFloat(-4500, 4500), bs.pos, RandFloat(0, 1000));
			cs.boost = RandFloat(20, 100);
		}
		car->SetState(cs);
	}
}

void FlipResetState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandFloat(-2200, 2200), RandFloat(-3000, 3000), RandFloat(minBallZ, maxBallZ));
	bs.vel = Vec(RandFloat(-300, 300), RandFloat(-300, 300), RandFloat(-150, 150));
	arena->ball->SetState(bs);

	bool first = true;
	for (Car* car : arena->_cars) {
		CarState cs = {};
		if (first) {
			first = false;
			// Below the ball, rising towards it, flip already spent. Reaching
			// the ball's underside with the wheels is what grants the reset.
			cs.pos = bs.pos - Vec(RandFloat(-150, 150),
			                      RandFloat(-150, 150),
			                      carBelowBall + RandFloat(-80, 80));
			cs.vel = Vec(RandFloat(-200, 200), RandFloat(-200, 200), RandFloat(300, 800));
			cs.rotMat = LookAt(cs.pos, bs.pos, 0.3f);
			cs.boost = RandFloat(60, 100);
			cs.isOnGround = false;
			cs.hasJumped = true;
			cs.hasDoubleJumped = true;
			cs.hasFlipped = true; // Flip is gone until a reset restores it
		} else {
			PutOnGround(cs, RandFloat(-3500, 3500), RandFloat(-4500, 4500), bs.pos, RandFloat(0, 1000));
			cs.boost = RandFloat(20, 100);
		}
		car->SetState(cs);
	}
}

void GroundDribbleState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	bool first = true;
	Vec ballPos;
	for (Car* car : arena->_cars) {
		CarState cs = {};
		if (first) {
			first = false;
			const float speed = RandFloat(minSpeed, maxSpeed);
			const float x = RandFloat(-2800, 2800);
			const float y = RandFloat(-3500, 2500);
			const float sign = TeamSign(car->team);

			// Drive towards the opponent's goal, ball balanced on the roof.
			PutOnGround(cs, x, y, Vec(x, sign * CommonValues::BACK_WALL_Y, 0), speed);
			cs.boost = RandFloat(30, 100);

			ballPos = cs.pos + Vec(0, 0, 145.f);
			BallState bs = {};
			bs.pos = ballPos;
			bs.vel = cs.vel + Vec(RandFloat(-60, 60), RandFloat(-60, 60), RandFloat(0, 120));
			arena->ball->SetState(bs);
		} else {
			PutOnGround(cs, RandFloat(-3500, 3500), RandFloat(-4500, 4500), ballPos, RandFloat(0, 1200));
			cs.boost = RandFloat(20, 100);
		}
		car->SetState(cs);
	}
}

void DemoState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	// Ball parked out of the way so the episode is about the cars.
	BallState bs = {};
	bs.pos = Vec(RandFloat(-1500, 1500), RandFloat(-1500, 1500), RandFloat(100, 900));
	bs.vel = RandUnitVec() * RandFloat(0, 800);
	arena->ball->SetState(bs);

	// Lay the cars out along a shared axis, blue on one side, orange the other,
	// all pointed at each other and moving fast.
	const float axisAng = RandFloat(-M_PI, M_PI);
	const Vec axis = Vec(std::cos(axisAng), std::sin(axisAng), 0);
	const Vec midpoint = Vec(RandFloat(-1500, 1500), RandFloat(-2500, 2500), 0);

	int blueN = 0, orangeN = 0;
	for (Car* car : arena->_cars) {
		const bool isBlue = (car->team == Team::BLUE);
		const int n = isBlue ? blueN++ : orangeN++;
		const float side = isBlue ? -1.f : 1.f;

		// Offset each extra car perpendicular to the axis so they do not stack.
		const Vec perp = Vec(-axis.y, axis.x, 0) * (n * 400.f - 200.f);
		const Vec pos = midpoint + axis * (side * separation * 0.5f) + perp;
		const Vec target = midpoint - axis * (side * separation * 0.5f);

		CarState cs = {};
		PutOnGround(cs,
		            std::clamp(pos.x, -3900.f, 3900.f),
		            std::clamp(pos.y, -4900.f, 4900.f),
		            target,
		            RandFloat(minSpeed, maxSpeed));
		cs.boost = RandFloat(30, 100);
		car->SetState(cs);
	}
}

void DefendState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	// Pick a team to be under pressure; mirror everything about that choice.
	const bool pressureOnBlue = RandFloat() > 0.5f;
	const float defSign = pressureOnBlue ? -1.f : 1.f; // Defended goal is at defSign * BACK_WALL_Y
	const Vec goal = Vec(0, defSign * CommonValues::BACK_WALL_Y, 300.f);

	BallState bs = {};
	bs.pos = Vec(RandFloat(-2500, 2500), defSign * RandFloat(500, 3000), RandFloat(100, 1200));
	// Aim the ball at the defended goal, roughly.
	{
		Vec dir = (goal + Vec(RandFloat(-900, 900), 0, RandFloat(-200, 400))) - bs.pos;
		const float len = dir.Length();
		if (len > 1.f)
			dir = dir / len;
		bs.vel = dir * RandFloat(minBallSpeed, maxBallSpeed);
	}
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		const bool defending = (car->team == Team::BLUE) == pressureOnBlue;
		CarState cs = {};
		if (defending) {
			// Goal side of the ball.
			PutOnGround(cs,
			            RandFloat(-1600, 1600),
			            defSign * RandFloat(3800, 5000),
			            bs.pos,
			            RandFloat(0, 900));
			cs.boost = RandFloat(15, 80);
		} else {
			// Attacker following the ball in.
			PutOnGround(cs,
			            bs.pos.x + RandFloat(-1500, 1500),
			            bs.pos.y - defSign * RandFloat(500, 2500),
			            bs.pos,
			            RandFloat(400, 1600));
			cs.boost = RandFloat(20, 100);
		}
		car->SetState(cs);
	}
}

void RecoverState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandFloat(-3000, 3000), RandFloat(-4000, 4000), RandFloat(100, 1200));
	bs.vel = RandUnitVec() * RandFloat(0, 2000);
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		// Airborne, tumbling, pointed nowhere useful, and far from the ball.
		float x, y;
		do {
			x = RandFloat(-3500, 3500);
			y = RandFloat(-4500, 4500);
		} while (Vec(x - bs.pos.x, y - bs.pos.y, 0).Length() < 1500.f);

		cs.pos = Vec(x, y, RandFloat(minZ, maxZ));
		cs.vel = RandUnitVec() * RandFloat(300, 1600);
		cs.angVel = RandUnitVec() * RandFloat(1.f, 5.f);
		cs.rotMat = Angle(RandFloat(-M_PI, M_PI),
		                  RandFloat(-M_PI / 2, M_PI / 2),
		                  RandFloat(-M_PI, M_PI)).ToRotMat();
		cs.boost = RandFloat(0, 60);
		cs.isOnGround = false;
		cs.hasJumped = true;
		cs.hasDoubleJumped = true;
		cs.hasFlipped = true;
		car->SetState(cs);
	}
}

void NeutralPlayState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandFloat(-3000, 3000), RandFloat(-4200, 4200), RandFloat(CommonValues::BALL_RADIUS, 1400));
	bs.vel = RandUnitVec() * RandFloat(0, 2500);
	bs.angVel = RLGC::Math::RandVec(Vec(-3, -3, -3), Vec(3, 3, 3));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		const float sign = TeamSign(car->team);
		// Bias each team towards its own half so the layout resembles real play
		// rather than a scramble.
		PutOnGround(cs,
		            RandFloat(-3600, 3600),
		            sign * RandFloat(-4600, 2500),
		            bs.pos,
		            RandFloat(0, 1700));
		cs.boost = RandFloat(0, 100);
		car->SetState(cs);
	}
}

} // namespace Hive
