#include "TestCommon.h"
#include "doctest/doctest.h"

#include <Config.h>
#include <env/Rewards.h>
#include <env/TimeoutCondition.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace Dash;
using namespace RLGC;
using namespace RLGC::CommonValues;

// Sign errors are this project's most expensive bug: nothing crashes, the run
// looks alive, and the bot spends a billion steps learning the opposite of what
// was intended. These pin down the direction of each reward and how its rungs
// rank against each other, never a hand-computed magnitude.

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

constexpr int BLUE = 0;
constexpr int ORANGE = 1;

// Same driving order as WeightedContribution, for a reward not in the stack.
float BareContribution(Reward *owned, Scenario &scenario, int playerIdx) {
	std::unique_ptr<Reward> reward(owned);
	scenario.Link();
	reward->Reset(scenario.prev);
	reward->PreStep(scenario.prev);
	reward->GetAllRewards(scenario.prev, false);
	reward->PreStep(scenario.state);
	return reward->GetAllRewards(scenario.state, false)[playerIdx];
}

std::vector<float> BareSequence(Reward *owned,
								const std::vector<GameState> &steps,
								int playerIdx) {
	std::unique_ptr<Reward> reward(owned);
	reward->Reset(steps.front());

	std::vector<float> out;
	for (const GameState &state : steps) {
		reward->PreStep(state);
		out.push_back(reward->GetAllRewards(state, false)[playerIdx]);
	}
	return out;
}

// Like BareSequence, but links state.prev/player.prev between consecutive
// steps, for rewards (AerialDistanceReward) that read them.
struct Chain {
	std::vector<GameState> steps;

	void Link() {
		for (size_t i = 1; i < steps.size(); i++) {
			steps[i].prev = &steps[i - 1];
			for (size_t p = 0; p < steps[i].players.size(); p++)
				steps[i].players[p].prev = &steps[i - 1].players[p];
		}
	}
};

std::vector<float> ChainRewards(Reward *owned, Chain &chain, int playerIdx) {
	std::unique_ptr<Reward> reward(owned);
	chain.Link();
	reward->Reset(chain.steps.front());

	std::vector<float> out;
	for (GameState &state : chain.steps) {
		reward->PreStep(state);
		out.push_back(reward->GetAllRewards(state, false)[playerIdx]);
	}
	return out;
}

// A single AerialDistanceReward step: blue at bluePos, ball at ballPos,
// blue optionally touching. Orange sits out of the way throughout.
GameState AerialDistanceStep(Vec bluePos, Vec ballPos, bool blueTouches) {
	GameState state;
	state.players = {MakePlayer(0, 1, Team::BLUE, bluePos),
					 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3000, 17))};
	state.ball.pos = ballPos;
	if (blueTouches)
		state.players[BLUE].ballTouchedStep = true;
	return state;
}

// Ball in the blue half, rolling hard at the blue net and inside both posts.
// The orange touch is what ConditionalVelocityBallToGoalReward gates on.
Scenario BallAtBlueNet() {
	Scenario scenario = MakeScenario(Vec(0, -4000, 100), Vec(0, -3000, 0),
									 Vec(0, -3800, 100), Vec(0, -3000, 0));
	scenario.prev.lastTouchCarID = scenario.prev.players[ORANGE].carId;
	scenario.state.lastTouchCarID = scenario.state.players[ORANGE].carId;
	return scenario;
}

// The same ball, same place, travelling away from the blue net instead.
Scenario BallAwayFromBlueNet() {
	Scenario scenario = MakeScenario(Vec(0, -4000, 100), Vec(0, 3000, 0),
									 Vec(0, -4200, 100), Vec(0, 3000, 0));
	scenario.prev.lastTouchCarID = scenario.prev.players[ORANGE].carId;
	scenario.state.lastTouchCarID = scenario.state.players[ORANGE].carId;
	return scenario;
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

float AirTouch(Scenario &scenario) {
	return BareContribution(new ImprovedAirTouchReward(), scenario, BLUE);
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
	Scenario aimed = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario away = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	Scenario missed = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	missed.state.players[BLUE].ballTouchedStep = false;

	CHECK(AirTouch(missed) == doctest::Approx(0.f));
	CHECK(AirTouch(aimed) > 0.f);
	CHECK(AirTouch(away) < 0.f);
}

// The aerial drills spawn cars already airborne, which have no launch height to
// measure a climb from. Seeding it from the spawn z tripped MAX_GROUND_LAUNCH_Z
// and silenced every AerialReward for the rest of the episode.
TEST_CASE("air rewards survive an airborne spawn") {
	Scenario scenario = MakeScenario(Vec(0, 0, 1200), Vec(0, 0, 0),
									 Vec(0, 0, 1200), Vec(0, 0, 0));

	for (GameState *gs : {&scenario.prev, &scenario.state}) {
		Player &blue = gs->players[BLUE];
		blue.pos = Vec(0, -850, 640);
		blue.vel = Vec(0, 700, 400);
		blue.isOnGround = false;
		blue.hasFlipped = true;
	}
	scenario.state.players[BLUE].pos = Vec(0, -790, 690);
	scenario.Link();

	CHECK(BareContribution(new AirVelToBallReward(650.f, 1800.f), scenario,
						   BLUE) > 0.f);
	CHECK(BareContribution(new AirFaceBallReward(650.f, 1800.f), scenario,
						   BLUE) > 0.f);
}

// Blue chases a ball that is pulling away from it, the shape of the first half
// of a double touch. The gap to the ball grew, which used to blank the reward
// outright; only the car's own approach counts now, so where the ball was a
// step ago cannot switch it off.
TEST_CASE("air face ball ignores what the ball is doing") {
	auto faceBall = [](Vec prevBallPos, Vec blueVel) {
		Scenario scenario = MakeScenario(Vec(0, 0, 1200), Vec(0, 3000, 0),
										 prevBallPos, Vec(0, 3000, 0));

		for (GameState *gs : {&scenario.prev, &scenario.state}) {
			Player &blue = gs->players[BLUE];
			blue.pos = Vec(0, -850, 640);
			blue.vel = blueVel;
			blue.isOnGround = false;
			blue.hasFlipped = true;
		}
		scenario.state.players[BLUE].pos = Vec(0, -790, 690);
		scenario.Link();

		return BareContribution(new AirFaceBallReward(650.f, 1800.f), scenario,
								BLUE);
	};

	const Vec chasing = Vec(0, 700, 400);
	const float receding = faceBall(Vec(0, -400, 1200), chasing);
	const float closing = faceBall(Vec(0, 400, 1200), chasing);

	CHECK(receding > 0.f);
	CHECK(receding == doctest::Approx(closing));

	// The guard that replaced it: nose on the ball while flying away from it.
	CHECK(faceBall(Vec(0, 400, 1200), Vec(0, -700, -400)) ==
		  doctest::Approx(0.f));
}

// The flip side of that seeding: a car that genuinely left the ground high up a
// wall is still not aerialing, so wall rides cannot farm the air rewards.
TEST_CASE("air rewards ignore a wall launch") {
	Scenario scenario = MakeScenario(Vec(0, 0, 1200), Vec(0, 0, 0),
									 Vec(0, 0, 1200), Vec(0, 0, 0));

	Player &prevBlue = scenario.prev.players[BLUE];
	prevBlue.pos = Vec(4000, -850, 900);
	prevBlue.vel = Vec(0, 700, 400);
	prevBlue.isOnGround = true;

	Player &blue = scenario.state.players[BLUE];
	blue.pos = Vec(3900, -790, 950);
	blue.vel = Vec(0, 700, 400);
	blue.isOnGround = false;
	scenario.Link();

	CHECK(BareContribution(new AirVelToBallReward(650.f, 1800.f), scenario,
						   BLUE) == doctest::Approx(0.f));
}

// Power is DirectionalTouchReward's job, not this reward's, so the incentive to
// strike rather than feather only exists in the stack as a whole.
TEST_CASE("the stack prefers a powerful aerial touch to a feathered one") {
	TrainConfig cfg = {};

	Scenario hard = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario feather = AerialTouch(Vec(0, 10, 0), Vec(0, 0, 0));

	CHECK(StackTotals(cfg, hard)[BLUE] > StackTotals(cfg, feather)[BLUE]);
}

TEST_CASE("a shot on target pays once, and re-arms on the opponent's touch") {
	const std::vector<GameState> steps = {
		ShotStep(OFF_TARGET_FAST, -1),     // arms
		ShotStep(ON_TARGET_FAST, BLUE),    // the shot
		ShotStep(ON_TARGET_FAST, -1),      // in flight
		ShotStep(ON_TARGET_FAST, BLUE),    // blue catches its own shot
		ShotStep(OFF_TARGET_FAST, ORANGE), // orange saves it
		ShotStep(ON_TARGET_FAST, BLUE),    // blue shoots again
	};

	const std::vector<float> paid =
		BareSequence(new ShotOnTargetReward(), steps, BLUE);

	CHECK(paid[0] == doctest::Approx(0.f));
	CHECK(paid[1] > 0.f);
	CHECK(paid[2] == doctest::Approx(0.f));
	CHECK(paid[3] == doctest::Approx(0.f));
	CHECK(paid[5] > 0.f);
}

TEST_CASE("timeout fires after maxTime of accumulated steps") {
	GameState state;
	state.deltaTime = 8.f / 120.f;

	const float maxTime = 1.0f;
	TimeoutCondition condition(maxTime);
	condition.Reset(state);

	int steps = 0;
	while (steps < 1000) {
		steps++;
		if (condition.IsTerminal(state))
			break;
	}

	CHECK(steps == (int)std::ceil(maxTime / state.deltaTime));
	CHECK(condition.IsTruncation());
}

TEST_CASE("onTarget's team argument names the goal being shot at") {
	Scenario scenario = BallAtBlueNet();

	CHECK(onTarget(scenario.state, Team::BLUE));
	CHECK_FALSE(onTarget(scenario.state, Team::ORANGE));
}

TEST_CASE("the whole stack prefers the ball leaving our own net") {
	TrainConfig cfg = {};

	Scenario threatened = BallAtBlueNet();
	Scenario cleared = BallAwayFromBlueNet();

	const float threatenedTotal = StackTotals(cfg, threatened)[BLUE];
	const float clearedTotal = StackTotals(cfg, cleared)[BLUE];

	CHECK(threatenedTotal <= clearedTotal);
}

// Train.cpp's StepCallback calls GetReward a SECOND time, on the same live
// reward objects EnvSet just stepped, to build the per-reward metrics. A reward
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

// A big pad always fills to 100, so "landed below full" is what separates the
// two pad sizes. Getting that backwards would pay the flat small-pad reward
// for every big pad in the game.
static float PadPickup(float prevBoost, float boost) {
	Scenario scenario =
		MakeScenario(Vec(0, 0, 93), Vec(0, 0, 0), Vec(0, 0, 93), Vec(0, 0, 0));
	scenario.prev.players[BLUE].boost = prevBoost;
	scenario.state.players[BLUE].boost = boost;
	scenario.Link();

	PadAwarePickupBoostReward reward;
	reward.Reset(scenario.prev);
	return reward.GetAllRewards(scenario.state, false)[BLUE];
}

TEST_CASE("a small pad pays a flat premium, a big pad pays the sqrt curve") {
	const float fromEmpty = PadPickup(0.f, 12.f);
	const float fromHalf = PadPickup(60.f, 72.f);

	CHECK(fromEmpty ==
		  doctest::Approx(PadAwarePickupBoostReward::SMALL_PAD_REWARD));
	CHECK(fromHalf == doctest::Approx(fromEmpty));
	CHECK(PadPickup(30.f, 100.f) > PadPickup(92.f, 100.f));
	CHECK(PadPickup(92.f, 100.f) < fromEmpty);
	CHECK(PadPickup(50.f, 50.f) == doctest::Approx(0.f));
	CHECK(PadPickup(50.f, 20.f) == doctest::Approx(0.f));
}

// The port had the height axis wrong: the Python source reads position.z (up),
// the port read pos.y (the goal-to-goal axis) instead.
TEST_CASE("aerial distance's touch reward reads height, not field position") {
	Chain grounded;
	grounded.steps = {AerialDistanceStep(Vec(0, 4000, 17), Vec(0, 4000, 17), false),
					  AerialDistanceStep(Vec(0, 4000, 17), Vec(0, 4000, 17), true)};

	Chain airborne;
	airborne.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
					  AerialDistanceStep(Vec(0, 0, 2000), Vec(0, 0, 2000), true)};

	CHECK(ChainRewards(new AerialDistanceReward(), grounded, BLUE).back() ==
		  doctest::Approx(0.f));
	CHECK(ChainRewards(new AerialDistanceReward(), airborne, BLUE).back() > 0.f);
}

// Python clears its own last_touch_agent when the tracked player lands, which
// forces the next touch back through the height-reward branch. The port
// checked lastTouchCarID (GameState's own field, which the reward can't
// clear) instead, so a second touch after landing paid the zeroed chain
// distance rather than a fresh height reward.
TEST_CASE("the chain resets when the carrier lands, even without another touch") {
	Chain chain;
	chain.steps = {
		AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
		AerialDistanceStep(Vec(0, 0, 1500), Vec(0, 0, 1500), true),
		AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
		AerialDistanceStep(Vec(0, 0, 1500), Vec(0, 0, 1500), true),
	};

	std::vector<float> paid = ChainRewards(new AerialDistanceReward(), chain, BLUE);

	CHECK(paid[1] > 0.f);
	CHECK(paid[3] == doctest::Approx(paid[1]));
}

// distances/rewards are sized to player count and were indexed by carId, an
// arena-lifetime id that has no relation to array position -- an
// out-of-bounds read/write, not just a misattributed reward.
TEST_CASE("carId does not have to match player index for the reward to land "
		 "on the right player") {
	GameState start;
	start.players = {MakePlayer(0, 100000, Team::BLUE, Vec(0, 0, 17)),
					 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3000, 17))};
	start.ball.pos = Vec(0, 0, 17);

	GameState touch;
	touch.players = {MakePlayer(0, 100000, Team::BLUE, Vec(0, 0, 1500)),
					 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3000, 17))};
	touch.ball.pos = Vec(0, 0, 1500);
	touch.players[BLUE].ballTouchedStep = true;

	Chain chain;
	chain.steps = {start, touch};

	std::vector<float> paid = ChainRewards(new AerialDistanceReward(), chain, BLUE);

	CHECK(paid[1] > 0.f);
}

// w1 == w2 == 0 does not chain in C++ the way it does in Python: it parses as
// (w1 == w2) == 0, true whenever the weights merely differ, so a zeroed car
// weight fell back to weighting car and ball equally. Weighting the ball only,
// the car's own height must not move the payout.
TEST_CASE("a zeroed distance weight is not clobbered back to equal weights") {
	Chain lowCar;
	lowCar.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
					AerialDistanceStep(Vec(0, 0, 1000), Vec(0, 0, 3000), true)};

	Chain highCar;
	highCar.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
					 AerialDistanceStep(Vec(0, 0, 2000), Vec(0, 0, 3000), true)};

	auto ballHeightOnly = [] {
		AerialDistanceReward *reward = new AerialDistanceReward();
		reward->carDistanceWeight = 0.f;
		reward->ballDistanceWeight = 1.f;
		return reward;
	};

	CHECK(ChainRewards(ballHeightOnly(), lowCar, BLUE).back() ==
		  doctest::Approx(ChainRewards(ballHeightOnly(), highCar, BLUE).back()));
}

// Blue attacks +y, so an air dribble carried upfield has to outrank the same
// distance carried back towards blue's own net. The floor keeps the wrong-way
// chain worth something rather than killing the mechanic outright.
TEST_CASE("an aerial chain pays more when the ball travels at the net") {
	auto carry = [](float dy) {
		GameState open =
			AerialDistanceStep(Vec(0, 0, 1500), Vec(0, 0, 1500), true);

		GameState carried =
			AerialDistanceStep(Vec(0, dy, 1500), Vec(0, dy, 1500), false);
		carried.ball.vel = Vec(0, dy, 0);

		GameState close = AerialDistanceStep(Vec(0, 2 * dy, 1500),
											 Vec(0, 2 * dy, 1500), true);
		close.ball.vel = Vec(0, dy, 0);

		Chain chain;
		chain.steps = {open, carried, close};
		return ChainRewards(new AerialDistanceReward(), chain, BLUE).back();
	};

	CHECK(carry(600.f) > carry(-600.f));
	CHECK(carry(-600.f) > 0.f);
}

// Blue cradles the ball; orange sits at oppDist along +y. The flick branches
// need blue airborne with its flip spent, which HasFlipOrJump() reads off
// hasFlipped and airTimeSinceJump rather than any single flag.
GameState CradleStep(float ballZ, Vec carried, bool onGround, bool hasFlip,
					 float oppDist) {
	GameState state;
	Player blue = MakePlayer(0, 1, Team::BLUE, Vec(0, -60, 17));
	blue.vel = carried;
	blue.isOnGround = onGround;
	blue.hasFlipped = !onGround && !hasFlip;
	blue.airTimeSinceJump = onGround ? 0.f : 0.2f;

	state.players = {blue, MakePlayer(1, 2, Team::ORANGE, Vec(0, oppDist, 17))};
	state.ball.pos = Vec(0, 0, ballZ);
	state.ball.vel = carried;
	return state;
}

float DribbleFlick(GameState state) {
	Community::DribbleFlickReward reward;
	reward.Reset(state);
	return reward.GetReward(state.players[BLUE], state, false);
}

// The rungs are the point: getting the ball into the zone at all, then holding
// it there, then flicking out of it. Each has to outrank the one below or the
// ladder does not lead anywhere.
TEST_CASE("dribble flick climbs from reaching the zone to carrying to "
		  "flicking") {
	const Vec carried = Vec(0, 1000, 0);

	const float inZone = DribbleFlick(CradleStep(250.f, Vec(), true, true, 5000.f));
	const float carrying = DribbleFlick(CradleStep(140.f, carried, true, true, 5000.f));
	const float withFlip = DribbleFlick(CradleStep(140.f, carried, false, true, 200.f));
	const float flicked = DribbleFlick(CradleStep(140.f, carried, false, false, 200.f));

	const float away =
		DribbleFlick(CradleStep(140.f, Vec(0, -1000, 0), true, true, 5000.f));

	CHECK(inZone > 0.f);
	CHECK(carrying > inZone);
	CHECK(withFlip > carrying);
	CHECK(flicked > withFlip);
	CHECK(flicked <= 1.f);
	// Carrying away from the net loses the directional rung, but the term
	// rewards -- it never punishes.
	CHECK(away == doctest::Approx(inZone));
}

// Once an opponent closes, holding the carry has to be the worst option on the
// board -- below dropping it, which pays nothing.
TEST_CASE("a challenged carry held on the ground is punished") {
	CHECK(DribbleFlick(CradleStep(140.f, Vec(0, 1000, 0), true, true, 200.f)) <
		  0.f);
}

// Blue cradles at `carryVel`, then the ball separates at `releasePos`/
// `releaseVel`. FlickReward pays across the window after the carry ends, so
// both steps have to run through one instance. `dodged` picks which airborne
// input broke the carry, since only a flip is a flick.
float Flick(Vec carryVel, Vec releasePos, Vec releaseVel, float oppDist,
			bool dodged = true) {
	GameState carry = CradleStep(140.f, carryVel, true, true, oppDist);

	GameState release = carry;
	release.ball.pos = releasePos;
	release.ball.vel = releaseVel;
	release.players[BLUE].isOnGround = false;
	release.players[BLUE].hasFlipped = dodged;
	release.players[BLUE].hasDoubleJumped = !dodged;

	FlickReward reward;
	reward.Reset(carry);
	for (int i = 0; i < reward.minCarrySteps; i++) {
		reward.PreStep(carry);
		reward.GetReward(carry.players[BLUE], carry, false);
	}
	reward.PreStep(release);
	return reward.GetReward(release.players[BLUE], release, false);
}

// Blue holds the carry for `groundSteps` on the ground and `airborneSteps`
// riding up with the ball, then pops it towards the net.
float CarryThenRelease(int groundSteps, int airborneSteps, bool dodged = true) {
	std::vector<GameState> steps;
	for (int i = 0; i < groundSteps; i++)
		steps.push_back(CradleStep(140.f, Vec(), true, true, 5000.f));

	for (int i = 1; i <= airborneSteps; i++) {
		GameState climbing = CradleStep(140.f, Vec(), false, true, 5000.f);
		climbing.players[BLUE].pos.z += 100.f * i;
		climbing.ball.pos.z += 100.f * i;
		steps.push_back(climbing);
	}

	GameState release = steps.back();
	release.ball.pos = Vec(0, 100, release.ball.pos.z + 160.f);
	release.ball.vel = Vec(0, 1500, 0);
	release.players[BLUE].isOnGround = false;
	release.players[BLUE].hasFlipped = dodged;
	release.players[BLUE].hasDoubleJumped = !dodged;
	steps.push_back(release);

	return BareSequence(new FlickReward(), steps, BLUE).back();
}

// The whole point of the reward: a pop straight up carries nothing towards the
// net, and letting the ball roll off imparts no velocity of its own.
TEST_CASE("a flick pays for power towards the net") {
	const float atGoal = Flick(Vec(), Vec(0, 100, 300), Vec(0, 1500, 0), 5000.f);
	const float harder = Flick(Vec(), Vec(0, 100, 300), Vec(0, 3000, 0), 5000.f);
	const float straightUp =
		Flick(Vec(), Vec(0, 100, 300), Vec(0, 0, 1500), 5000.f);
	const float dropped =
		Flick(Vec(0, 1000, 0), Vec(0, 400, 100), Vec(0, 1000, 0), 5000.f);

	CHECK(atGoal > 0.f);
	CHECK(harder > atGoal);
	CHECK(harder <= 1.f);
	CHECK(straightUp == doctest::Approx(0.f).epsilon(0.01));
	CHECK(dropped == doctest::Approx(0.f));
}

TEST_CASE("a flick under pressure outscores the same flick unopposed") {
	const float free = Flick(Vec(), Vec(0, 100, 300), Vec(0, 1500, 0), 5000.f);
	const float pressured = Flick(Vec(), Vec(0, 100, 300), Vec(0, 1500, 0), 200.f);

	CHECK(free == doctest::Approx(pressured * FlickReward().freeFlickScale));
}

// A ball that merely bounced off the roof was never carried, and one the car
// rides up with is an air dribble. The jump that starts a real flick is the
// case in between, and has to survive.
TEST_CASE("only a carry held on the ground arms the flick") {
	const FlickReward reward;

	CHECK(CarryThenRelease(reward.minCarrySteps - 1, 0) ==
		  doctest::Approx(0.f));
	CHECK(CarryThenRelease(reward.minCarrySteps, 0) > 0.f);
	CHECK(CarryThenRelease(reward.minCarrySteps, reward.maxAirSteps) > 0.f);
	CHECK(CarryThenRelease(reward.minCarrySteps, reward.maxAirSteps + 1) ==
		  doctest::Approx(0.f));
}

// A double jump out of a carry pops the ball just as hard, so power and
// direction alone cannot tell the two apart. Flipping is what makes it a flick.
TEST_CASE("only a dodge out of the carry pays") {
	const FlickReward reward;

	CHECK(Flick(Vec(), Vec(0, 100, 300), Vec(0, 1500, 0), 5000.f, false) ==
		  doctest::Approx(0.f));
	CHECK(Flick(Vec(), Vec(0, 100, 300), Vec(0, 1500, 0), 5000.f, true) > 0.f);
	CHECK(CarryThenRelease(reward.minCarrySteps, 0, false) ==
		  doctest::Approx(0.f));
}

// The separation spans several steps. Paying the peak once, rather than a sum
// over every step the ball is still flying, is what keeps this an event.
TEST_CASE("a flick pays its peak once across the window") {
	GameState carry = CradleStep(140.f, Vec(), true, true, 5000.f);

	GameState release = carry;
	release.ball.pos = Vec(0, 100, 300);
	release.ball.vel = Vec(0, 1500, 0);
	release.players[BLUE].isOnGround = false;
	release.players[BLUE].hasFlipped = true;

	FlickReward reward;
	reward.Reset(carry);
	for (int i = 0; i < reward.minCarrySteps; i++) {
		reward.PreStep(carry);
		reward.GetReward(carry.players[BLUE], carry, false);
	}

	reward.PreStep(release);
	const float first = reward.GetReward(release.players[BLUE], release, false);

	// Train.cpp reads every reward a second time to fill its metrics row.
	CHECK(reward.GetReward(release.players[BLUE], release, false) ==
		  doctest::Approx(first));

	float total = first;
	for (int i = 0; i < reward.windowSteps + 2; i++) {
		reward.PreStep(release);
		total += reward.GetReward(release.players[BLUE], release, false);
	}

	CHECK(first > 0.f);
	CHECK(total == doctest::Approx(first));
}

// PossessionReward moved off GetAllRewards, so its whole-arena pass now runs in
// PreStep. The sign is the part that matters: whoever reaches the ball first is
// positive and their opponent is the exact negation.
TEST_CASE("possession favours the car that gets there first, symmetrically") {
	GameState state;
	state.players = {MakePlayer(0, 1, Team::BLUE, Vec(0, -400, 17)),
					 MakePlayer(1, 2, Team::ORANGE, Vec(0, 3500, 17))};
	state.ball.pos = Vec(0, 0, BALL_RADIUS);

	PossessionReward reward;
	reward.Reset(state);
	reward.PreStep(state);
	std::vector<float> paid = reward.GetAllRewards(state, false);

	CHECK(paid[BLUE] > 0.f);
	CHECK(paid[ORANGE] == doctest::Approx(-paid[BLUE]));

	// Train.cpp evaluates this reward a second time for its metrics row.
	CHECK(reward.GetReward(state.players[BLUE], state, false) ==
		  doctest::Approx(paid[BLUE]));
	CHECK(reward.GetReward(state.players[ORANGE], state, false) ==
		  doctest::Approx(paid[ORANGE]));
}

// A team with nobody left to contest cannot be scored against on time-to-ball,
// so the whole arena pays zero rather than an arbitrary sign.
TEST_CASE("possession pays nothing when only one team is present") {
	GameState state;
	state.players = {MakePlayer(0, 1, Team::BLUE, Vec(0, -400, 17))};
	state.ball.pos = Vec(0, 0, BALL_RADIUS);

	PossessionReward reward;
	reward.Reset(state);
	reward.PreStep(state);

	CHECK(reward.GetAllRewards(state, false)[BLUE] == doctest::Approx(0.f));
}
