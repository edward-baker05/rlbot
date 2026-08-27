#include "doctest/doctest.h"
#include "TestCommon.h"

#include <Config.h>
#include <env/Rewards.h>

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

} // namespace

TEST_CASE("onTarget's team argument names the goal being shot at") {
	Scenario scenario = BallAtBlueNet();

	CHECK(onTarget(scenario.state, Team::BLUE));
	CHECK_FALSE(onTarget(scenario.state, Team::ORANGE));
}

TEST_CASE("a ball on target at our own net is a punishment") {
	TrainConfig cfg = {};
	Scenario threatened = BallAtBlueNet();

	const float blue =
		WeightedContribution(cfg, "Own Goal Threat", threatened, BLUE);
	const float orange =
		WeightedContribution(cfg, "Own Goal Threat", threatened, ORANGE);

	CHECK(blue < 0.f);
	CHECK(orange == doctest::Approx(0.f));
}

TEST_CASE("the whole stack prefers the ball leaving our own net") {
	TrainConfig cfg = {};

	Scenario threatened = BallAtBlueNet();
	Scenario cleared = BallAwayFromBlueNet();

	const float threatenedTotal = StackTotals(cfg, threatened)[BLUE];
	const float clearedTotal = StackTotals(cfg, cleared)[BLUE];

	CHECK(threatenedTotal < clearedTotal);
}
