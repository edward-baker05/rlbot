#include "doctest/doctest.h"

#include <policy/RolloutPlanner.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/CommonValues.h>

using namespace Hive;
using namespace RLGC;

TEST_SUITE("RolloutPlanner") {

TEST_CASE("GenerateCandidates produces valid bounded actions") {
	PlannerConfig cfg;
	cfg.numCandidates = 32;
	RolloutPlanner planner(cfg);

	Player p = {};
	p.isOnGround = true;
	p.rotMat = Angle(0, 0, 0).ToRotMat();

	Action base = {};
	base.throttle = 0.5f;
	base.steer = -0.2f;
	base.boost = 0.f;
	base.jump = 0.f;

	auto candidates = planner.GenerateCandidates(base, p);
	CHECK(!candidates.empty());
	CHECK(candidates.size() <= 32);

	// The first candidate must be the base action
	CHECK(candidates[0].throttle == doctest::Approx(0.5f));
	CHECK(candidates[0].steer == doctest::Approx(-0.2f));

	for (const auto& c : candidates) {
		CHECK(c.throttle >= -1.0f);
		CHECK(c.throttle <= 1.0f);
		CHECK(c.steer >= -1.0f);
		CHECK(c.steer <= 1.0f);
		CHECK(c.pitch >= -1.0f);
		CHECK(c.pitch <= 1.0f);
		CHECK(c.yaw >= -1.0f);
		CHECK(c.yaw <= 1.0f);
		CHECK(c.roll >= -1.0f);
		CHECK(c.roll <= 1.0f);
		CHECK((c.boost == 0.f || c.boost == 1.f));
		CHECK((c.jump == 0.f || c.jump == 1.f));
		CHECK((c.handbrake == 0.f || c.handbrake == 1.f));
	}
}

TEST_CASE("EvaluateRollout rewards goal-directed velocity and penalizes own goals") {
	PlannerConfig cfg;
	RolloutPlanner planner(cfg);

	RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	RocketSim::Car* car = arena->AddCar(RocketSim::Team::BLUE);

	// Scenario 1: Ball moving fast toward opponent goal (+Y for Blue)
	RocketSim::BallState ballAttacking = {};
	ballAttacking.pos = Vec(0, 2000, 100);
	ballAttacking.vel = Vec(0, 1500, 0); // 1500 uu/s towards opponent goal
	arena->ball->SetState(ballAttacking);

	RocketSim::CarState carState = {};
	carState.pos = Vec(0, 1500, 17);
	carState.isOnGround = true;
	carState.rotMat = Angle(0, 0, 0).ToRotMat();
	car->SetState(carState);

	float scoreAttacking = planner.EvaluateRollout(
		arena, car, RocketSim::Team::BLUE, Vec(0, 1000, 100), /*hadContact=*/true, /*goalScored=*/false, RocketSim::Team::BLUE);

	// Scenario 2: Ball moving fast toward own goal (-Y for Blue)
	RocketSim::BallState ballDefendingDanger = {};
	ballDefendingDanger.pos = Vec(0, -3000, 100);
	ballDefendingDanger.vel = Vec(0, -1500, 0); // 1500 uu/s towards own goal
	arena->ball->SetState(ballDefendingDanger);

	float scoreOwnGoalDanger = planner.EvaluateRollout(
		arena, car, RocketSim::Team::BLUE, Vec(0, -2000, 100), /*hadContact=*/false, /*goalScored=*/false, RocketSim::Team::BLUE);

	CHECK(scoreAttacking > scoreOwnGoalDanger);

	// Scenario 3: Goal scored for blue vs against blue
	float scoreGoalBlue = planner.EvaluateRollout(
		arena, car, RocketSim::Team::BLUE, Vec(0, 0, 0), true, true, RocketSim::Team::BLUE);
	float scoreGoalOrange = planner.EvaluateRollout(
		arena, car, RocketSim::Team::BLUE, Vec(0, 0, 0), true, true, RocketSim::Team::ORANGE);

	CHECK(scoreGoalBlue > scoreGoalOrange);
	CHECK(scoreGoalOrange < 0.f);

	delete arena;
}

TEST_CASE("PlanAction runs and selects an action without crashing") {
	PlannerConfig cfg;
	cfg.horizonTicks = 8;
	cfg.numCandidates = 16;
	cfg.temperature = 0.0f; // Argmax mode
	RolloutPlanner planner(cfg);

	RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	arena->AddCar(RocketSim::Team::BLUE);
	arena->AddCar(RocketSim::Team::ORANGE);
	arena->ResetToRandomKickoff();

	GameState state(arena);
	Player player = state.players[0];

	Action baseAction = {};
	baseAction.throttle = 1.0f;
	baseAction.steer = 0.0f;

	Action planned = planner.PlanAction(state, player, baseAction);

	CHECK(planned.throttle >= -1.0f);
	CHECK(planned.throttle <= 1.0f);
	CHECK(planned.steer >= -1.0f);
	CHECK(planned.steer <= 1.0f);

	delete arena;
}

}
