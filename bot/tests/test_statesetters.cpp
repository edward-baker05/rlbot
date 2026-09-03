#include "TestCommon.h"
#include "doctest/doctest.h"

#include <env/StateSetters.h>

#include <RLGymCPP/CommonValues.h>

#include <memory>

using namespace Dash;
using namespace RLGC;
using namespace RLGC::CommonValues;

// BackboardFollowState is only useful if the ball it spawns actually reaches
// the attacked backboard in the air, with the chaser behind it and able to
// follow. Simulating is the only honest check: the arithmetic that motivated
// the spawn ranges ignores drag and the corner geometry.
namespace {

struct Spawn {
	Team attackTeam;
	float dir;
	Vec chaserPos;
	Vec chaserVel;
	Vec ballPos;
	Vec ballVel;
	bool chaserAirborne;
	float chaserBoost;
};

Spawn ReadSpawn(Arena* arena) {
	Spawn s = {};
	s.ballPos = arena->ball->GetState().pos;
	s.ballVel = arena->ball->GetState().vel;
	s.dir = s.ballVel.y > 0 ? 1.f : -1.f;
	s.attackTeam = s.dir > 0 ? Team::BLUE : Team::ORANGE;

	for (Car* car : arena->_cars) {
		CarState cs = car->GetState();
		if (car->team == s.attackTeam && !cs.isOnGround) {
			s.chaserPos = cs.pos;
			s.chaserVel = cs.vel;
			s.chaserAirborne = true;
			s.chaserBoost = cs.boost;
		}
	}
	return s;
}

} // namespace

TEST_CASE("BackboardFollowState spawns a live backboard shot") {
	Dash::Test::EnsureRocketSim();

	BackboardFollowState setter(100.f);

	int reachedBackboard = 0;
	constexpr int TRIALS = 60;

	for (int trial = 0; trial < TRIALS; trial++) {
		std::unique_ptr<Arena> arena(Arena::Create(GameMode::SOCCAR));
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
		setter.ResetArena(arena.get());

		Spawn s = ReadSpawn(arena.get());

		// The chaser exists, is airborne, and is behind the ball with boost.
		REQUIRE(s.chaserAirborne);
		CHECK(s.chaserBoost > 50.f);
		CHECK(s.chaserPos.z < s.ballPos.z);
		CHECK(s.chaserPos.z > 250.f);
		CHECK((s.ballPos.y - s.chaserPos.y) * s.dir > 0.f);
		// Chaser is travelling after the ball, not away from it.
		CHECK(s.chaserVel.y * s.dir > 0.f);
		// Ball starts in the attacking half but short of the backboard.
		CHECK(std::abs(s.ballPos.y) < 3000.f);

		// Roll forward and watch for the ball reaching the attacked backboard
		// while still off the floor.
		float bounceZ = -1.f;
		float bounceTime = -1.f;
		float prevVelY = s.ballVel.y;
		for (int tick = 0; tick < 300; tick++) {
			arena->Step(1);
			BallState bs = arena->ball->GetState();
			if (bounceTime < 0.f && bs.vel.y * s.dir < 0.f &&
				prevVelY * s.dir > 0.f) {
				bounceTime = tick * TICK_TIME;
				bounceZ = bs.pos.z;
			}
			prevVelY = bs.vel.y;
		}

		// The shot is solved from a target flight time and impact height, so
		// every spawn should strike the backboard high, not roll into the wall.
		REQUIRE(bounceTime > 0.f);
		CHECK(bounceZ > 600.f);
		CHECK(bounceZ < CommonValues::CEILING_Z);
		CHECK(bounceTime > 0.7f);
		CHECK(bounceTime < 1.8f);
		reachedBackboard++;
	}

	CHECK(reachedBackboard == TRIALS);
}

TEST_CASE("BackboardFollowState attacks both nets") {
	Dash::Test::EnsureRocketSim();

	BackboardFollowState setter(100.f);
	int blueAttacks = 0;

	for (int trial = 0; trial < 60; trial++) {
		std::unique_ptr<Arena> arena(Arena::Create(GameMode::SOCCAR));
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
		setter.ResetArena(arena.get());

		Spawn s = ReadSpawn(arena.get());
		if (s.attackTeam == Team::BLUE)
			blueAttacks++;

		// The defender sits back in front of the net under threat.
		for (Car* car : arena->_cars) {
			if (car->team == s.attackTeam)
				continue;
			CarState cs = car->GetState();
			CHECK(cs.pos.y * s.dir > 3000.f);
			CHECK(cs.isOnGround);
		}
	}

	CHECK(blueAttacks > 10);
	CHECK(blueAttacks < 50);
}
