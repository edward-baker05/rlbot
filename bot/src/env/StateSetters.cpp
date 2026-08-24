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

}  // namespace Dash
