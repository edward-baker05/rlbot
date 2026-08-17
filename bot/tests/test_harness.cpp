#include "doctest/doctest.h"
#include "TestCommon.h"

using namespace RLGC;

TEST_CASE("harness runs and RocketSim initializes") {
	CHECK(1 + 1 == 2);
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	CHECK(arena != nullptr);
	delete arena;
}

// The eval subcommand's scorekeeping hangs off this callback; a wrong team
// attribution would silently invert every eval result.
TEST_CASE("goal score callback fires and credits the scoring team") {
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	struct GoalFlag { bool scored = false; Team team = Team::BLUE; } flag;
	arena->SetGoalScoreCallback(
		[](Arena*, Team team, void* user) {
			auto* g = static_cast<GoalFlag*>(user);
			g->scored = true;
			g->team = team;
		},
		&flag);

	arena->ResetToRandomKickoff();

	// Fire the ball into the orange goal (+y): blue scores.
	BallState bs = arena->ball->GetState();
	bs.pos = Vec(0, 4500, 200);
	bs.vel = Vec(0, 3000, 0);
	arena->ball->SetState(bs);
	arena->Step(120);

	CHECK(flag.scored);
	CHECK(flag.team == Team::BLUE);
	delete arena;
}
