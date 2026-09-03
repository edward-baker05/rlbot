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

// An arena holding the same ball, for stepping ground truth by hand.
RocketSim::Arena* MakeTruthArena(const RLGC::GameState& s) {
	RocketSim::Arena* truth =
		RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	RocketSim::BallState bs = truth->ball->GetState();
	bs.pos = s.ball.pos;
	bs.vel = s.ball.vel;
	bs.angVel = s.ball.angVel;
	truth->ball->SetState(bs);
	return truth;
}

// Move `s` forward along its own predicted path, as an untouched ball does.
RLGC::GameState Advanced(const RLGC::GameState& s, const BallTrajectory& t,
                         int ticks) {
	RLGC::GameState next = s;
	next.lastTickCount = s.lastTickCount + ticks;
	next.ball.pos = t.PosAt(ticks);
	next.ball.vel = t.VelAt(ticks);
	return next;
}

} // namespace

TEST_CASE("BallPredictor fills the whole prediction window") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;
	RLGC::GameState s = MakeFlyingState();

	const BallTrajectory& t = pred.Get(s);

	CHECK(t.Size() == BallPredictor::WINDOW_TICKS);
	CHECK(t.startTick == 0);

	// Offset 0 is the present, so it must equal the state we handed in.
	CHECK(t.PosAt(0).x == doctest::Approx(s.ball.pos.x));
	CHECK(t.PosAt(0).y == doctest::Approx(s.ball.pos.y));
	CHECK(t.PosAt(0).z == doctest::Approx(s.ball.pos.z));
}

TEST_CASE("BallPredictor prediction matches a real arena stepped forward") {
	Dash::Test::EnsureRocketSim();

	RLGC::GameState s = MakeFlyingState();
	RocketSim::Arena* truth = MakeTruthArena(s);

	BallPredictor pred;
	const BallTrajectory& t = pred.Get(s);

	// 240 ticks is long enough to include the first ground bounce.
	truth->Step(240);
	const RocketSim::Vec truthPos = truth->ball->GetState().pos;

	CHECK(t.PosAt(240).x == doctest::Approx(truthPos.x).epsilon(0.001));
	CHECK(t.PosAt(240).y == doctest::Approx(truthPos.y).epsilon(0.001));
	CHECK(t.PosAt(240).z == doctest::Approx(truthPos.z).epsilon(0.001));

	delete truth;
}

TEST_CASE("BallPredictor reuses the cached trajectory when the ball is untouched") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// One env step at tickSkip 8.
	pred.Get(Advanced(s, first, 8));
	CHECK(pred.SimulationCount() == 1); // no re-simulation
}

TEST_CASE("BallPredictor slides the window instead of re-simulating it") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	const uint64_t ticksAfterFirst = pred.SimulatedTickCount();
	REQUIRE(ticksAfterFirst == BallPredictor::WINDOW_TICKS - 1);

	// Advancing 8 ticks must cost 8 ball-only ticks, not another whole window.
	pred.Get(Advanced(s, first, 8));
	CHECK(pred.SimulatedTickCount() == ticksAfterFirst + 8);
}

TEST_CASE("BallPredictor samples stay anchored to the present across a slide") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	RocketSim::Arena* truth = MakeTruthArena(s);

	const BallTrajectory& first = pred.Get(s);
	const RLGC::GameState next = Advanced(s, first, 8);

	const BallTrajectory& t = pred.Get(next);

	// The shallowest sample must now be the ball 8 + 18 ticks after the
	// original state, not 18.
	truth->Step(8 + BallPredictor::SAMPLE_TICKS[0]);
	const RocketSim::Vec expected = truth->ball->GetState().pos;
	const RocketSim::Vec got = t.PosAt(BallPredictor::SAMPLE_TICKS[0]);

	CHECK(got.x == doctest::Approx(expected.x).epsilon(0.001));
	CHECK(got.y == doctest::Approx(expected.y).epsilon(0.001));
	CHECK(got.z == doctest::Approx(expected.z).epsilon(0.001));
	CHECK(t.startTick == next.lastTickCount);

	delete truth;
}

TEST_CASE("BallPredictor stays anchored across many consecutive slides") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	RocketSim::Arena* truth = MakeTruthArena(s);

	// 60 env steps of an untouched ball: far past where the old fixed horizon
	// would have gone stale, and through several bounces.
	RLGC::GameState cur = s;
	for (int i = 0; i < 60; i++)
		cur = Advanced(cur, pred.Get(cur), 8);

	const BallTrajectory& t = pred.Get(cur);
	REQUIRE(pred.SimulationCount() == 1);

	truth->Step(60 * 8 + BallPredictor::SAMPLE_TICKS.back());
	const RocketSim::Vec expected = truth->ball->GetState().pos;
	const RocketSim::Vec got = t.PosAt(BallPredictor::SAMPLE_TICKS.back());

	CHECK(got.x == doctest::Approx(expected.x).epsilon(0.001));
	CHECK(got.y == doctest::Approx(expected.y).epsilon(0.001));
	CHECK(got.z == doctest::Approx(expected.z).epsilon(0.001));

	delete truth;
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

TEST_CASE("BallPredictor re-simulates when the present is past the window") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	RocketSim::Arena* truth = MakeTruthArena(s);
	pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// A correctly predicted ball, but further ahead than the window reaches,
	// so sliding cannot get there.
	const int jump = BallPredictor::WINDOW_TICKS + 10;
	truth->Step(jump);
	RLGC::GameState late = s;
	late.lastTickCount = 1000 + jump;
	late.ball.pos = truth->ball->GetState().pos;
	late.ball.vel = truth->ball->GetState().vel;

	pred.Get(late);
	CHECK(pred.SimulationCount() == 2);

	delete truth;
}

TEST_CASE("BallPredictor::Reset forces the next call to re-simulate") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	const RLGC::GameState next = Advanced(s, first, 8);

	pred.Reset();
	pred.Get(next);
	CHECK(pred.SimulationCount() == 2);
}

TEST_CASE("BallPredictor detects the first ground bounce") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	// Thrown straight down with no horizontal motion: the first bounce is
	// on the floor, directly under the launch point.
	//
	// The downward velocity is not incidental. Arena::Step sleeps the ball
	// whenever its linear AND angular velocity are both exactly zero
	// (Arena.cpp:722), so a ball released from rest hangs in mid-air
	// indefinitely rather than falling.
	RLGC::GameState s = {};
	s.ball.pos = {500, -700, 1000};
	s.ball.vel = {0, 0, -500};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);

	REQUIRE(t.bounceTick > 0);
	// ~907uu of travel from 500uu/s under 650uu/s^2 is a bit over a second.
	CHECK(t.bounceTick < 200);
	CHECK(t.bouncePos.x == doctest::Approx(500).epsilon(0.01));
	CHECK(t.bouncePos.y == doctest::Approx(-700).epsilon(0.01));
	// Contact happens at roughly one ball radius above the floor.
	CHECK(t.bouncePos.z < 150.f);

	// The bounce must reverse vertical velocity.
	CHECK(t.VelAt(t.bounceTick - 1).z < 0);
	CHECK(t.VelAt(t.bounceTick).z > 0);
}

TEST_CASE("BallPredictor counts the bounce down as the window slides") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = {};
	s.ball.pos = {500, -700, 1000};
	s.ball.vel = {0, 0, -500};
	s.lastTickCount = 0;

	const BallTrajectory& first = pred.Get(s);
	const int bounceTick = first.bounceTick;
	const RocketSim::Vec bouncePos = first.bouncePos;
	REQUIRE(bounceTick > 8);

	const BallTrajectory& t = pred.Get(Advanced(s, first, 8));

	REQUIRE(pred.SimulationCount() == 1);
	CHECK(t.bounceTick == bounceTick - 8);
	CHECK(t.bouncePos.x == doctest::Approx(bouncePos.x));
	CHECK(t.bouncePos.z == doctest::Approx(bouncePos.z));
}

TEST_CASE("BallPredictor reports no bounce for a ball at rest") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 93};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);
	CHECK(t.bounceTick == -1);
}

TEST_CASE("BallPredictor detects a goal and which net") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	// Rolling hard at the orange net (+y).
	RLGC::GameState s = {};
	s.ball.pos = {0, 3000, 93};
	s.ball.vel = {0, 3000, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);

	REQUIRE(t.goalTick > 0);
	CHECK(t.goalTeam == 1);  // orange's net
	CHECK(t.goalTick < BallPredictor::WINDOW_TICKS);

	// Past the goal line the trajectory is frozen, not an imaginary
	// continuation of play.
	CHECK(t.VelAt(BallPredictor::WINDOW_TICKS - 1).Length()
	      == doctest::Approx(0));
	CHECK(t.PosAt(BallPredictor::WINDOW_TICKS - 1).y
	      == doctest::Approx(t.PosAt(t.goalTick).y));
}

TEST_CASE("BallPredictor counts the goal down as the window slides") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = {};
	s.ball.pos = {0, 3000, 93};
	s.ball.vel = {0, 3000, 0};
	s.lastTickCount = 0;

	const BallTrajectory& first = pred.Get(s);
	const int goalTick = first.goalTick;
	REQUIRE(goalTick > 8);

	const BallTrajectory& t = pred.Get(Advanced(s, first, 8));

	REQUIRE(pred.SimulationCount() == 1);
	CHECK(t.goalTick == goalTick - 8);
	CHECK(t.goalTeam == 1);
}

TEST_CASE("BallPredictor reports no goal for a ball going nowhere") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 93};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);
	CHECK(t.goalTick == -1);
	CHECK(t.goalTeam == -1);
}
