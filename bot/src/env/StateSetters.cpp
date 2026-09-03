#include "StateSetters.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>

using namespace RLGC;
using RocketSim::Math::RandFloat;

namespace Dash {

void InfiniteBoostState::ResetArena(Arena* arena) {
	inner->ResetArena(arena);

	lastWasInfinite = RandFloat() < chance;

	MutatorConfig cfg = arena->GetMutatorConfig();
	cfg.boostUsedPerSecond =
		lastWasInfinite ? 0.f : RLConst::BOOST_USED_PER_SECOND;
	arena->SetMutatorConfig(cfg);

	if (lastWasInfinite) {
		for (Car* car : arena->_cars) {
			CarState state = car->GetState();
			state.boost = 100;
			car->SetState(state);
		}
	}
}

void AerialHoverState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	constexpr float X_MAX = 2400.f;
	constexpr float Y_MAX = 3000.f;

	BallState bs = {};
	bs.pos = RLGC::Math::RandVec(Vec(-X_MAX, -Y_MAX, 600.f),
								Vec(X_MAX, Y_MAX, 1600.f));
	bs.vel = RLGC::Math::RandVec(Vec(-500.f, -500.f, -200.f),
								Vec(500.f, 500.f, 300.f));
	bs.angVel = RLGC::Math::RandVec(Vec(-2.f, -2.f, -2.f), Vec(2.f, 2.f, 2.f));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		cs.pos = RLGC::Math::RandVec(Vec(-X_MAX, -Y_MAX, 400.f),
									Vec(X_MAX, Y_MAX, 1200.f));

		Vec dirToBall = (bs.pos - cs.pos).Normalized();
		Angle angle = Angle::FromVec(dirToBall);
		angle.yaw += RandFloat(-0.35f, 0.35f);
		angle.pitch += RandFloat(-0.25f, 0.25f);
		angle.roll += RandFloat(-0.35f, 0.35f);
		angle.NormalizeFix();
		cs.rotMat = angle.ToRotMat();

		float speed = RandFloat(200.f, 800.f);
		cs.vel = dirToBall * speed +
				 RLGC::Math::RandVec(Vec(-150.f, -150.f, -100.f),
									Vec(150.f, 150.f, 150.f));
		cs.angVel =
			RLGC::Math::RandVec(Vec(-1.f, -1.f, -1.f), Vec(1.f, 1.f, 1.f));
		cs.boost = initialBoost;
		car->SetState(cs);
	}
}

void HighBallPopUpState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	constexpr float X_MAX = 2400.f;
	constexpr float Y_MAX = 3000.f;

	BallState bs = {};
	bs.pos = RLGC::Math::RandVec(Vec(-X_MAX * 0.7f, -Y_MAX * 0.7f, 300.f),
								Vec(X_MAX * 0.7f, Y_MAX * 0.7f, 750.f));
	bs.vel = RLGC::Math::RandVec(Vec(-350.f, -350.f, 850.f),
								Vec(350.f, 350.f, 1650.f));
	bs.angVel = RLGC::Math::RandVec(Vec(-2.f, -2.f, -2.f), Vec(2.f, 2.f, 2.f));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		cs.pos = RLGC::Math::RandVec(Vec(-X_MAX, -Y_MAX, 17.f),
									Vec(X_MAX, Y_MAX, 17.f));
		cs.pos.z = 17.f;

		Vec toBallXY = Vec(bs.pos.x - cs.pos.x, bs.pos.y - cs.pos.y, 0.f);
		float yaw = atan2f(toBallXY.y, toBallXY.x);
		Angle angle = Angle(yaw + RandFloat(-0.25f, 0.25f), 0.f, 0.f);
		angle.NormalizeFix();
		cs.rotMat = angle.ToRotMat();

		cs.vel =
			RLGC::Math::RandVec(Vec(-300.f, -300.f, 0.f), Vec(300.f, 300.f, 0.f));
		cs.vel.z = 0.f;
		cs.angVel = {};
		cs.boost = initialBoost;
		car->SetState(cs);
	}
}

void BackboardFollowState::ResetArena(Arena* arena) {
	arena->ResetToRandomKickoff();

	const bool blueAttacks = RandFloat() < 0.5f;
	const float dir = blueAttacks ? 1.f : -1.f;
	const Team attackTeam = blueAttacks ? Team::BLUE : Team::ORANGE;

	const float ballX = RandFloat(-1600.f, 1600.f);
	const float ballY = RandFloat(1000.f, 2600.f) * dir;
	const float ballZ = RandFloat(700.f, 1400.f);

	// Solve the shot from the flight time and the height it should strike the
	// backboard at, so gravity cannot turn it into a ball rolling into the wall.
	const float flightTime = RandFloat(0.9f, 1.4f);
	const float impactZ = RandFloat(750.f, 1500.f);
	const float toWall =
		(CommonValues::BACK_WALL_Y - CommonValues::BALL_RADIUS) - ballY * dir;
	const float gravity = -CommonValues::GRAVITY_Z;

	BallState bs = {};
	bs.pos = Vec(ballX, ballY, ballZ);
	bs.vel = Vec(RandFloat(-250.f, 250.f), (toWall / flightTime) * dir,
				(impactZ - ballZ + 0.5f * gravity * flightTime * flightTime) /
					flightTime);
	bs.angVel = RLGC::Math::RandVec(Vec(-3.f, -3.f, -3.f), Vec(3.f, 3.f, 3.f));
	arena->ball->SetState(bs);

	bool chaserPlaced = false;
	for (Car* car : arena->_cars) {
		CarState cs = {};

		if (car->team == attackTeam && !chaserPlaced) {
			chaserPlaced = true;

			cs.pos = Vec(ballX + RandFloat(-400.f, 400.f),
						ballY - RandFloat(500.f, 1100.f) * dir,
						RandFloat(350.f, RS_MIN(1000.f, ballZ - 150.f)));
			cs.vel = Vec(RandFloat(-200.f, 200.f),
						RandFloat(1100.f, 1700.f) * dir,
						RandFloat(100.f, 400.f));

			Vec dirToBall = (bs.pos - cs.pos).Normalized();
			Angle angle = Angle::FromVec(dirToBall);
			angle.yaw += RandFloat(-0.2f, 0.2f);
			angle.pitch += RandFloat(-0.15f, 0.15f);
			angle.roll += RandFloat(-0.3f, 0.3f);
			angle.NormalizeFix();
			cs.rotMat = angle.ToRotMat();

			cs.angVel = RLGC::Math::RandVec(Vec(-0.5f, -0.5f, -0.5f),
										Vec(0.5f, 0.5f, 0.5f));
			cs.isOnGround = false;
			cs.hasJumped = true;

			// The first touch usually spends the flip, but not always, so both
			// variants are spawned.
			const bool keepsFlip = RandFloat() < 0.35f;
			cs.hasFlipped = !keepsFlip;
			cs.hasDoubleJumped = !keepsFlip;
			cs.airTimeSinceJump =
				keepsFlip ? RandFloat(0.1f, 0.5f)
						: RLConst::DOUBLEJUMP_MAX_DELAY + RandFloat(0.2f, 1.f);

			cs.boost = initialBoost;
		} else {
			const bool defending = car->team != attackTeam;
			cs.pos = Vec(RandFloat(-1400.f, 1400.f),
						defending ? RandFloat(3600.f, 4600.f) * dir
								: RandFloat(0.f, 1500.f) * -dir,
						17.f);

			Vec toBallXY = Vec(bs.pos.x - cs.pos.x, bs.pos.y - cs.pos.y, 0.f);
			float yaw = atan2f(toBallXY.y, toBallXY.x);
			Angle angle = Angle(yaw + RandFloat(-0.25f, 0.25f), 0.f, 0.f);
			angle.NormalizeFix();
			cs.rotMat = angle.ToRotMat();

			cs.vel = Vec(RandFloat(-300.f, 300.f), RandFloat(-300.f, 300.f), 0.f);
			cs.angVel = {};
			cs.boost = RandFloat(20.f, 60.f);
		}

		car->SetState(cs);
	}
}

}  // namespace Dash
