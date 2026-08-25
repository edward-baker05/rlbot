#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/BallPredictor.h>

#include <RLGymCPP/Gamestates/GameState.h>

using namespace Dash;

namespace {

// A ball in free flight, high enough that it will not hit anything for a while.
RLGC::GameState MakeFlyingState(uint64_t tick = 0) {
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 1200};
	s.ball.vel = {300, 500, 200};
	s.ball.angVel = {0, 0, 0};
	s.lastTickCount = tick;
	return s;
}

} // namespace

TEST_CASE("BallPredictor produces a full-horizon trajectory") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;
	RLGC::GameState s = MakeFlyingState();

	const BallTrajectory& t = pred.Get(s);

	CHECK(t.pos.size() == BallPredictor::SIM_HORIZON_TICKS + 1);
	CHECK(t.vel.size() == BallPredictor::SIM_HORIZON_TICKS + 1);
	CHECK(t.startTick == 0);

	// Slice 0 is the present, so it must equal the state we handed in.
	CHECK(t.pos[0].x == doctest::Approx(s.ball.pos.x));
	CHECK(t.pos[0].y == doctest::Approx(s.ball.pos.y));
	CHECK(t.pos[0].z == doctest::Approx(s.ball.pos.z));
}

TEST_CASE("BallPredictor prediction matches a real arena stepped forward") {
	Dash::Test::EnsureRocketSim();

	// Ground truth: an arena with no cars, stepped by hand.
	RocketSim::Arena* truth = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	RLGC::GameState s = MakeFlyingState();
	truth->ball->SetState([&]{
		RocketSim::BallState bs = truth->ball->GetState();
		bs.pos = s.ball.pos;
		bs.vel = s.ball.vel;
		bs.angVel = s.ball.angVel;
		return bs;
	}());

	BallPredictor pred;
	const BallTrajectory& t = pred.Get(s);

	// 240 ticks is long enough to include the first ground bounce.
	truth->Step(240);
	const RocketSim::Vec truthPos = truth->ball->GetState().pos;

	CHECK(t.pos[240].x == doctest::Approx(truthPos.x).epsilon(0.001));
	CHECK(t.pos[240].y == doctest::Approx(truthPos.y).epsilon(0.001));
	CHECK(t.pos[240].z == doctest::Approx(truthPos.z).epsilon(0.001));

	delete truth;
}

TEST_CASE("BallPredictor reuses the cached trajectory when the ball is untouched") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	const uint64_t simsAfterFirst = pred.SimulationCount();
	REQUIRE(simsAfterFirst == 1);

	// Advance 8 ticks along the predicted path -- exactly what an untouched
	// ball does over one env step at tickSkip 8.
	RLGC::GameState next = s;
	next.lastTickCount = 1008;
	next.ball.pos = first.pos[8];
	next.ball.vel = first.vel[8];

	pred.Get(next);
	CHECK(pred.SimulationCount() == simsAfterFirst); // no re-simulation
}

TEST_CASE("BallPredictor re-simulates when the ball diverges from prediction") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// A touch: same position, very different velocity.
	RLGC::GameState touched = s;
	touched.lastTickCount = 1008;
	touched.ball.vel = {-1500, -900, 400};

	pred.Get(touched);
	CHECK(pred.SimulationCount() == 2);
}

TEST_CASE("BallPredictor re-simulates when the cached horizon is exhausted") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// Jump far enough ahead that fewer than the deepest sample (312 ticks)
	// remain in the cache.
	const int offset = BallPredictor::SIM_HORIZON_TICKS - 100;
	RLGC::GameState late = s;
	late.lastTickCount = 1000 + offset;
	late.ball.pos = first.pos[offset];
	late.ball.vel = first.vel[offset];

	pred.Get(late);
	CHECK(pred.SimulationCount() == 2);
}

TEST_CASE("BallPredictor::Reset forces the next call to re-simulate") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	RLGC::GameState next = s;
	next.lastTickCount = 1008;
	next.ball.pos = first.pos[8];
	next.ball.vel = first.vel[8];

	pred.Reset();
	pred.Get(next);
	CHECK(pred.SimulationCount() == 2);
}
