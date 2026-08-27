#include "doctest/doctest.h"
#include "TestCommon.h"

#include <Config.h>
#include <env/Rewards.h>
#include <env/TimeoutCondition.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <memory>
#include <string>
#include <vector>

using namespace Dash;
using namespace RLGC;
using namespace RLGC::CommonValues;

// Sign errors are this project's most expensive bug: nothing crashes, the run
// looks alive, and the bot spends a billion steps learning the opposite of what
// was intended. These pin down the direction of each reward, not its value.

namespace {

Player MakePlayer(int index, uint32_t carId, Team team, Vec pos) {
	Player player = {};
	player.index = index;
	player.carId = carId;
	player.team = team;
	player.pos = pos;
	player.vel = Vec(0, 0, 0);
	player.boost = 50;
	player.isOnGround = true;
	player.rotMat = RotMat(Vec(0, 1, 0), Vec(-1, 0, 0), Vec(0, 0, 1));
	return player;
}

struct Scenario {
	GameState prev;
	GameState state;

	// GameState::prev and Player::prev are raw pointers into the members above,
	// so the links have to be rebuilt after any copy.
	void Link() {
		state.prev = &prev;
		for (size_t i = 0; i < state.players.size(); i++)
			state.players[i].prev = &prev.players[i];
	}
};

Scenario MakeScenario(Vec ballPos, Vec ballVel, Vec prevBallPos,
					  Vec prevBallVel) {
	Scenario scenario;

	scenario.prev.ball.pos = prevBallPos;
	scenario.prev.ball.vel = prevBallVel;
	scenario.prev.players = {MakePlayer(0, 1, Team::BLUE, Vec(0, -3000, 17)),
							 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3000, 17))};

	scenario.state.ball.pos = ballPos;
	scenario.state.ball.vel = ballVel;
	scenario.state.players = scenario.prev.players;

	scenario.Link();
	return scenario;
}

// Mirrors EnvSet's ordering: PreStep over the whole stack, then GetAllRewards
// over the whole stack, weighted contributions summed per player.
std::vector<float> StackTotals(const TrainConfig &cfg, Scenario &scenario) {
	scenario.Link();
	std::vector<RewardSpec> specs = GeneralRewardSpecs(cfg);
	std::vector<std::unique_ptr<Reward>> rewards;
	for (const RewardSpec &spec : specs)
		rewards.emplace_back(spec.make());

	for (auto &reward : rewards)
		reward->Reset(scenario.prev);
	for (auto &reward : rewards)
		reward->PreStep(scenario.prev);
	for (auto &reward : rewards)
		reward->GetAllRewards(scenario.prev, false);

	for (auto &reward : rewards)
		reward->PreStep(scenario.state);

	std::vector<float> totals(scenario.state.players.size(), 0.f);
	for (size_t i = 0; i < rewards.size(); i++) {
		std::vector<float> output =
			rewards[i]->GetAllRewards(scenario.state, false);
		for (size_t p = 0; p < totals.size(); p++)
			totals[p] += output[p] * specs[i].weight;
	}
	return totals;
}

float WeightedContribution(const TrainConfig &cfg, const std::string &name,
						   Scenario &scenario, int playerIdx) {
	scenario.Link();
	for (const RewardSpec &spec : GeneralRewardSpecs(cfg)) {
		if (spec.name != name)
			continue;

		std::unique_ptr<Reward> reward(spec.make());
		reward->Reset(scenario.prev);
		reward->PreStep(scenario.prev);
		reward->GetAllRewards(scenario.prev, false);
		reward->PreStep(scenario.state);
		return reward->GetAllRewards(scenario.state, false)[playerIdx] *
			   spec.weight;
	}

	FAIL("no reward spec named ", name);
	return 0.f;
}

constexpr int BLUE = 0;
constexpr int ORANGE = 1;

// Ball in the blue half, rolling hard at the blue net and inside both posts.
Scenario BallAtBlueNet() {
	return MakeScenario(Vec(0, -4000, 100), Vec(0, -3000, 0),
						Vec(0, -3800, 100), Vec(0, -3000, 0));
}

// The same ball, same place, travelling away from the blue net instead.
Scenario BallAwayFromBlueNet() {
	return MakeScenario(Vec(0, -4000, 100), Vec(0, 3000, 0),
						Vec(0, -4200, 100), Vec(0, 3000, 0));
}

float OwnGoalThreat(const TrainConfig &cfg, Scenario &scenario, int playerIdx) {
	return WeightedContribution(cfg, "Own Goal Threat", scenario, playerIdx);
}

// Blue in a genuine aerial: on the ground in the previous state so the climb
// latches, airborne and touching with the ball above the height gate.
Scenario AerialTouch(Vec ballVel, Vec prevBallVel) {
	Scenario scenario;

	scenario.prev.ball.pos = Vec(0, 0, 1000);
	scenario.prev.ball.vel = prevBallVel;
	scenario.prev.players = {MakePlayer(0, 1, Team::BLUE, Vec(0, -200, 17)),
							 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3000, 17))};

	scenario.state = scenario.prev;
	scenario.state.ball.vel = ballVel;

	Player &blue = scenario.state.players[BLUE];
	blue.pos = Vec(0, -200, 800);
	blue.isOnGround = false;
	blue.ballTouchedStep = true;

	scenario.Link();
	return scenario;
}

float AirTouch(const TrainConfig &cfg, Scenario &scenario) {
	return WeightedContribution(cfg, "Air Touch", scenario, BLUE);
}

// One step of a shot sequence: blue attacking the orange net.
GameState ShotStep(Vec ballVel, int toucher) {
	GameState state;
	state.deltaTime = 8.f / 120.f;
	state.ball.pos = Vec(0, 4000, 300);
	state.ball.vel = ballVel;
	state.players = {MakePlayer(0, 1, Team::BLUE, Vec(0, 3500, 17)),
					 MakePlayer(1, 2, Team::ORANGE, Vec(0, 4600, 17))};
	if (toucher >= 0) {
		state.players[toucher].ballTouchedStep = true;
		state.lastTouchCarID = state.players[toucher].carId;
	}
	return state;
}

// Weighted payout per step for one reward driven over a scripted sequence,
// in EnvSet's PreStep-then-GetAllRewards order.
std::vector<float> RunSequence(const TrainConfig &cfg, const std::string &name,
							   const std::vector<GameState> &steps,
							   int playerIdx) {
	for (const RewardSpec &spec : GeneralRewardSpecs(cfg)) {
		if (spec.name != name)
			continue;

		std::unique_ptr<Reward> reward(spec.make());
		reward->Reset(steps.front());

		std::vector<float> out;
		for (const GameState &state : steps) {
			reward->PreStep(state);
			out.push_back(reward->GetAllRewards(state, false)[playerIdx] *
						  spec.weight);
		}
		return out;
	}

	FAIL("no reward spec named ", name);
	return {};
}

constexpr Vec ON_TARGET_FAST = Vec(0, 3000, 0);
constexpr Vec OFF_TARGET_FAST = Vec(3000, 0, 0);

// Blue striking an on-target shot, off a step where the ball was not one.
Scenario ShotScenario() {
	Scenario scenario;
	scenario.prev = ShotStep(OFF_TARGET_FAST, -1);
	scenario.state = ShotStep(ON_TARGET_FAST, BLUE);
	scenario.Link();
	return scenario;
}

} // namespace

TEST_CASE("air touch rewards aim, and never rewards missing") {
	TrainConfig cfg = {};

	Scenario aimedHard = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario aimedFeather = AerialTouch(Vec(0, 10, 0), Vec(0, 0, 0));
	Scenario awayHard = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	Scenario awayFeather = AerialTouch(Vec(0, -10, 0), Vec(0, 0, 0));
	Scenario missed = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	missed.state.players[BLUE].ballTouchedStep = false;

	const float missedR = AirTouch(cfg, missed);
	const float awayHardR = AirTouch(cfg, awayHard);
	const float awayFeatherR = AirTouch(cfg, awayFeather);
	const float aimedFeatherR = AirTouch(cfg, aimedFeather);

	CHECK(missedR == doctest::Approx(0.f));

	// The escape hatch: a touch under the old 50uu/s guard skipped the
	// alignment factor entirely, so feathering the ball away from the net
	// scored exactly the same as striking it at the net.
	CHECK(awayFeatherR < aimedFeatherR);

	// And an unfloored negative alignment made whiffing worth more than any
	// badly aimed hit, which is its own reason to miss the ball.
	CHECK(awayHardR > missedR);
	CHECK(awayHardR == doctest::Approx(awayFeatherR));
}

// Power is DirectionalTouchReward's job, not this reward's, so the incentive to
// strike rather than feather only exists in the stack as a whole.
TEST_CASE("the stack prefers a powerful aerial touch to a feathered one") {
	TrainConfig cfg = {};

	Scenario hard = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario feather = AerialTouch(Vec(0, 10, 0), Vec(0, 0, 0));

	CHECK(StackTotals(cfg, hard)[BLUE] > StackTotals(cfg, feather)[BLUE]);
}

TEST_CASE("a badly aimed aerial touch beats whiffing in the stack") {
	TrainConfig cfg = {};

	Scenario awayHard = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	Scenario missed = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	missed.state.players[BLUE].ballTouchedStep = false;

	CHECK(StackTotals(cfg, awayHard)[BLUE] > StackTotals(cfg, missed)[BLUE]);
}

TEST_CASE("air touch has no cliff across the old 50uu/s guard") {
	TrainConfig cfg = {};

	Scenario under = AerialTouch(Vec(0, 49, 0), Vec(0, 0, 0));
	Scenario over = AerialTouch(Vec(0, 51, 0), Vec(0, 0, 0));

	CHECK(AirTouch(cfg, under) == doctest::Approx(AirTouch(cfg, over)));
}

TEST_CASE("a shot on target pays once, not once per re-touch") {
	TrainConfig cfg = {};

	const std::vector<GameState> steps = {
		ShotStep(OFF_TARGET_FAST, -1),   // arms
		ShotStep(ON_TARGET_FAST, BLUE),  // the shot
		ShotStep(ON_TARGET_FAST, -1),    // in flight
		ShotStep(ON_TARGET_FAST, BLUE),  // blue catches its own shot
	};

	const std::vector<float> paid =
		RunSequence(cfg, "On Target", steps, BLUE);

	CHECK(paid[0] == doctest::Approx(0.f));
	CHECK(paid[1] > 0.f);
	CHECK(paid[2] == doctest::Approx(0.f));
	CHECK(paid[3] == doctest::Approx(0.f));
}

TEST_CASE("a shot on target pays again after the opponent intervenes") {
	TrainConfig cfg = {};

	const std::vector<GameState> steps = {
		ShotStep(OFF_TARGET_FAST, -1),     // arms
		ShotStep(ON_TARGET_FAST, BLUE),    // the shot
		ShotStep(OFF_TARGET_FAST, ORANGE), // orange saves it
		ShotStep(ON_TARGET_FAST, BLUE),    // blue shoots again
	};

	const std::vector<float> paid =
		RunSequence(cfg, "On Target", steps, BLUE);

	CHECK(paid[1] > 0.f);
	CHECK(paid[3] > 0.f);
}

TEST_CASE("timeout fires after maxTime of accumulated steps") {
	GameState state;
	state.deltaTime = 8.f / 120.f;

	TimeoutCondition condition(1.0f);
	condition.Reset(state);

	int steps = 0;
	while (steps < 1000) {
		steps++;
		if (condition.IsTerminal(state))
			break;
	}

	CHECK(steps == 15);
	CHECK(condition.IsTruncation());
}

TEST_CASE("onTarget's team argument names the goal being shot at") {
	Scenario scenario = BallAtBlueNet();

	CHECK(onTarget(scenario.state, Team::BLUE));
	CHECK_FALSE(onTarget(scenario.state, Team::ORANGE));
}

TEST_CASE("a ball closing on our own net is a punishment") {
	TrainConfig cfg = {};
	Scenario threatened = BallAtBlueNet();

	CHECK(OwnGoalThreat(cfg, threatened, BLUE) < 0.f);
	CHECK(OwnGoalThreat(cfg, threatened, ORANGE) == doctest::Approx(0.f));
}

TEST_CASE("a ball leaving our own net is not punished") {
	TrainConfig cfg = {};
	Scenario cleared = BallAwayFromBlueNet();

	CHECK(OwnGoalThreat(cfg, cleared, BLUE) == doctest::Approx(0.f));
}

// The bug this replaces paid out per step regardless of speed, so a slow ball
// left to trickle in scored the same as a rocket -- and scored more the longer
// it was left alone.
TEST_CASE("own goal threat scales with closing speed, not with time") {
	TrainConfig cfg = {};

	Scenario slow = MakeScenario(Vec(0, -4000, 100), Vec(0, -300, 0),
								 Vec(0, -3980, 100), Vec(0, -300, 0));
	Scenario fast = BallAtBlueNet();

	CHECK(OwnGoalThreat(cfg, fast, BLUE) < OwnGoalThreat(cfg, slow, BLUE));
	CHECK(OwnGoalThreat(cfg, slow, BLUE) < 0.f);
}

// The gate this replaces zeroed the punishment whenever our own team touched
// last, which made hitting it in ourselves cheaper than conceding a shot.
TEST_CASE("own goal threat does not care who touched last") {
	TrainConfig cfg = {};

	Scenario blueTouched = BallAtBlueNet();
	blueTouched.state.lastTouchCarID = blueTouched.state.players[BLUE].carId;
	blueTouched.prev.lastTouchCarID = blueTouched.state.lastTouchCarID;

	Scenario orangeTouched = BallAtBlueNet();
	orangeTouched.state.lastTouchCarID =
		orangeTouched.state.players[ORANGE].carId;
	orangeTouched.prev.lastTouchCarID = orangeTouched.state.lastTouchCarID;

	CHECK(OwnGoalThreat(cfg, blueTouched, BLUE) ==
		  doctest::Approx(OwnGoalThreat(cfg, orangeTouched, BLUE)));
	CHECK(OwnGoalThreat(cfg, blueTouched, BLUE) < 0.f);
}

TEST_CASE("the whole stack prefers the ball leaving our own net") {
	TrainConfig cfg = {};

	Scenario threatened = BallAtBlueNet();
	Scenario cleared = BallAwayFromBlueNet();

	const float threatenedTotal = StackTotals(cfg, threatened)[BLUE];
	const float clearedTotal = StackTotals(cfg, cleared)[BLUE];

	CHECK(threatenedTotal < clearedTotal);
}

// Train.cpp's StepCallback calls GetReward a SECOND time, on the same live
// reward objects EnvSet just stepped, to build the RewardMass metrics. A reward
// that mutates its own state in GetReward therefore reports zero forever after,
// while training itself looks fine -- which is exactly what the shot latch did.
TEST_CASE("GetReward is repeatable for the same state") {
	TrainConfig cfg = {};

	Scenario shot = ShotScenario();
	Scenario aerial = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario threatened = BallAtBlueNet();

	for (Scenario *scenario : {&shot, &aerial, &threatened}) {
		scenario->Link();
		for (const RewardSpec &spec : GeneralRewardSpecs(cfg)) {
			std::unique_ptr<Reward> reward(spec.make());
			reward->Reset(scenario->prev);
			reward->PreStep(scenario->prev);
			reward->GetAllRewards(scenario->prev, false);
			reward->PreStep(scenario->state);

			const std::vector<float> first =
				reward->GetAllRewards(scenario->state, false);
			const std::vector<float> second =
				reward->GetAllRewards(scenario->state, false);

			REQUIRE(first.size() == second.size());
			for (size_t i = 0; i < first.size(); i++) {
				INFO("reward: " << spec.name << " player " << i);
				CHECK(first[i] == doctest::Approx(second[i]));
			}
		}
	}
}

// Guards the test above from going vacuous: it only proves anything while the
// scenario actually makes the latched reward fire.
TEST_CASE("the shot scenario actually pays out, twice over") {
	TrainConfig cfg = {};
	Scenario shot = ShotScenario();

	CHECK(WeightedContribution(cfg, "On Target", shot, BLUE) > 0.f);
	CHECK(WeightedContribution(cfg, "On Target", shot, BLUE) > 0.f);
}
