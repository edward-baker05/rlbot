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

// THE FAILURE THIS TEST EXISTS FOR: the mutator config belongs to the ARENA and
// outlives the episode. If InfiniteBoostState only wrote `boostUsedPerSecond`
// on the infinite episodes and never restored it, the first infinite episode
// would make that arena infinite forever -- and the run would simply look like
// a bot that had solved its own boost problem.
TEST_CASE("InfiniteBoostState restores the normal drain rate every reset") {
	Hive::Test::EnsureRocketSim();

	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	SUBCASE("chance 1 gives every episode a full, non-draining tank") {
		Hive::InfiniteBoostState setter(new Hive::NeutralPlayState(), 1.f);
		for (int i = 0; i < 5; i++) {
			setter.ResetArena(arena);
			CHECK(setter.LastWasInfinite());
			CHECK(arena->GetMutatorConfig().boostUsedPerSecond == 0.f);
			for (Car* car : arena->_cars)
				CHECK(car->GetState().boost == doctest::Approx(100.f));
		}
	}

	SUBCASE("chance 0 never touches the drain rate") {
		Hive::InfiniteBoostState setter(new Hive::NeutralPlayState(), 0.f);
		for (int i = 0; i < 5; i++) {
			setter.ResetArena(arena);
			CHECK(!setter.LastWasInfinite());
			CHECK(arena->GetMutatorConfig().boostUsedPerSecond ==
			      doctest::Approx(RLConst::BOOST_USED_PER_SECOND));
		}
	}

	SUBCASE("an infinite episode does not leak into the next normal one") {
		// Force infinite, then force normal on the same arena.
		Hive::InfiniteBoostState infinite(new Hive::NeutralPlayState(), 1.f);
		infinite.ResetArena(arena);
		REQUIRE(arena->GetMutatorConfig().boostUsedPerSecond == 0.f);

		Hive::InfiniteBoostState normal(new Hive::NeutralPlayState(), 0.f);
		normal.ResetArena(arena);
		CHECK(arena->GetMutatorConfig().boostUsedPerSecond ==
		      doctest::Approx(RLConst::BOOST_USED_PER_SECOND));
	}

	SUBCASE("the share is roughly the configured chance") {
		Hive::InfiniteBoostState setter(new Hive::NeutralPlayState(), 0.25f);
		int infinite = 0;
		for (int i = 0; i < 2000; i++) {
			setter.ResetArena(arena);
			infinite += setter.LastWasInfinite() ? 1 : 0;
		}
		// 2000 draws at p=0.25 has sd ~0.0097, so 0.20-0.30 is ~5 sigma.
		CHECK(infinite / 2000.f > 0.20f);
		CHECK(infinite / 2000.f < 0.30f);
	}

	delete arena;
}
