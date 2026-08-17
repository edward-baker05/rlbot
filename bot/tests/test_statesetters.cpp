#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/StateSetters.h>

#include <RLGymCPP/CommonValues.h>

#include <algorithm>
#include <cmath>

using namespace Hive;

namespace {

struct ArenaFixture {
	Arena* arena;
	ArenaFixture() {
		Hive::Test::EnsureRocketSim();
		arena = Arena::Create(GameMode::SOCCAR);
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}
	~ArenaFixture() { delete arena; }
};

bool InField(const Vec& p, float margin = 0.f) {
	return std::abs(p.x) < 4096.f + margin &&
	       std::abs(p.y) < 5120.f + 880.f + margin && // goals recess past the back wall
	       p.z > -1.f && p.z < 2044.f + margin;
}

template <typename Setter>
void CheckBasicInvariants(Setter& setter, int iterations = 100) {
	ArenaFixture f;
	for (int i = 0; i < iterations; i++) {
		setter.ResetArena(f.arena);
		CHECK(InField(f.arena->ball->GetState().pos));
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			CHECK(InField(cs.pos));
			CHECK(cs.boost >= 0.f);
			CHECK(cs.boost <= 100.f);
		}
	}
}

} // namespace

TEST_CASE("NeutralPlayState basic invariants") { NeutralPlayState s; CheckBasicInvariants(s); }
TEST_CASE("BallContactState basic invariants") { BallContactState s; CheckBasicInvariants(s); }
TEST_CASE("DefendState basic invariants")      { DefendState s; CheckBasicInvariants(s); }
TEST_CASE("RecoverState basic invariants")     { RecoverState s; CheckBasicInvariants(s); }
TEST_CASE("AerialState basic invariants")      { AerialState s; CheckBasicInvariants(s); }
TEST_CASE("AirDribbleState basic invariants")  { AirDribbleState s; CheckBasicInvariants(s); }
TEST_CASE("FlipResetState basic invariants")   { FlipResetState s; CheckBasicInvariants(s); }
TEST_CASE("DemoState basic invariants")        { DemoState s; CheckBasicInvariants(s); }

TEST_CASE("AerialState puts the ball in its configured height band with cars grounded") {
	AerialState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		CHECK(ball.pos.z >= s.minBallZ - 1.f);
		CHECK(ball.pos.z <= s.maxBallZ + 1.f);
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			CHECK(cs.pos.z < 100.f);
			CHECK(cs.boost >= s.minBoost - 1.f);
		}
	}
}

TEST_CASE("BallContactState spawns a car within its configured distance of the ball") {
	BallContactState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		float best = 1e9f;
		for (Car* car : f.arena->_cars)
			best = std::min(best, (car->GetState().pos - ball.pos).Length());
		CHECK(best <= s.maxDist + 100.f);
	}
}

TEST_CASE("FlipResetState puts the car airborne below the ball") {
	FlipResetState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		CHECK(ball.pos.z >= s.minBallZ - 1.f);
		bool anyAirborneBelow = false;
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			if (cs.pos.z > 300.f && cs.pos.z < ball.pos.z)
				anyAirborneBelow = true;
		}
		CHECK(anyAirborneBelow);
	}
}
