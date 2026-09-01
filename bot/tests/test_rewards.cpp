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

// The reward returns a positive magnitude and is wired with a negative weight,
// so the sign is reapplied here.
constexpr float OWN_GOAL_THREAT_WEIGHT = -0.005f;

float OwnGoalThreat(Scenario &scenario, int playerIdx) {
	return OWN_GOAL_THREAT_WEIGHT *
		   BareContribution(new OwnGoalThreatPunishment(), scenario, playerIdx);
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
	Scenario awayHard = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	Scenario missed = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	missed.state.players[BLUE].ballTouchedStep = false;

	CHECK(AirTouch(cfg, missed) == doctest::Approx(0.f));
	CHECK(AirTouch(cfg, aimedHard) > 0.f);
	CHECK(AirTouch(cfg, awayHard) < 0.f);
}

// Documented, not endorsed: t4 replaced this guard with a 0.1 alignment floor,
// and which is right is still open.
TEST_CASE("air touch skips the alignment factor below 50uu/s") {
	TrainConfig cfg = {};

	Scenario aimedFeather = AerialTouch(Vec(0, 10, 0), Vec(0, 0, 0));
	Scenario awayFeather = AerialTouch(Vec(0, -10, 0), Vec(0, 0, 0));

	CHECK(AirTouch(cfg, awayFeather) ==
		  doctest::Approx(AirTouch(cfg, aimedFeather)));
}

// Power is DirectionalTouchReward's job, not this reward's, so the incentive to
// strike rather than feather only exists in the stack as a whole.
TEST_CASE("the stack prefers a powerful aerial touch to a feathered one") {
	TrainConfig cfg = {};

	Scenario hard = AerialTouch(Vec(0, 1000, 0), Vec(0, 0, 0));
	Scenario feather = AerialTouch(Vec(0, 10, 0), Vec(0, 0, 0));

	CHECK(StackTotals(cfg, hard)[BLUE] > StackTotals(cfg, feather)[BLUE]);
}

// The cost of the guard above: a hard touch away from the net is punished
// harder than not reaching the ball at all, so whiffing beats a bad hit.
TEST_CASE("a badly aimed aerial touch loses to whiffing in the stack") {
	TrainConfig cfg = {};

	Scenario awayHard = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	Scenario missed = AerialTouch(Vec(0, -1000, 0), Vec(0, 0, 0));
	missed.state.players[BLUE].ballTouchedStep = false;

	CHECK(StackTotals(cfg, awayHard)[BLUE] < StackTotals(cfg, missed)[BLUE]);
}

TEST_CASE("a shot on target pays once, not once per re-touch") {
	const std::vector<GameState> steps = {
		ShotStep(OFF_TARGET_FAST, -1),	// arms
		ShotStep(ON_TARGET_FAST, BLUE), // the shot
		ShotStep(ON_TARGET_FAST, -1),	// in flight
		ShotStep(ON_TARGET_FAST, BLUE), // blue catches its own shot
	};

	const std::vector<float> paid =
		BareSequence(new ShotOnTargetReward(), steps, BLUE);

	CHECK(paid[0] == doctest::Approx(0.f));
	CHECK(paid[1] > 0.f);
	CHECK(paid[2] == doctest::Approx(0.f));
	CHECK(paid[3] == doctest::Approx(0.f));
}

TEST_CASE("a shot on target pays again after the opponent intervenes") {
	const std::vector<GameState> steps = {
		ShotStep(OFF_TARGET_FAST, -1),	   // arms
		ShotStep(ON_TARGET_FAST, BLUE),	   // the shot
		ShotStep(OFF_TARGET_FAST, ORANGE), // orange saves it
		ShotStep(ON_TARGET_FAST, BLUE),	   // blue shoots again
	};

	const std::vector<float> paid =
		BareSequence(new ShotOnTargetReward(), steps, BLUE);

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
	Scenario threatened = BallAtBlueNet();

	CHECK(OwnGoalThreat(threatened, BLUE) < 0.f);
	CHECK(OwnGoalThreat(threatened, ORANGE) == doctest::Approx(0.f));
}

TEST_CASE("a ball leaving our own net is not punished") {
	Scenario cleared = BallAwayFromBlueNet();

	CHECK(OwnGoalThreat(cleared, BLUE) == doctest::Approx(0.f));
}

// The bug this replaces paid out per step regardless of speed, so a slow ball
// left to trickle in scored the same as a rocket -- and scored more the longer
// it was left alone.
TEST_CASE("own goal threat scales with closing speed, not with time") {
	Scenario slow = MakeScenario(Vec(0, -4000, 100), Vec(0, -300, 0),
								 Vec(0, -3980, 100), Vec(0, -300, 0));
	Scenario fast = BallAtBlueNet();

	CHECK(OwnGoalThreat(fast, BLUE) < OwnGoalThreat(slow, BLUE));
	CHECK(OwnGoalThreat(slow, BLUE) < 0.f);
}

// The gate this replaces zeroed the punishment whenever our own team touched
// last, which made hitting it in ourselves cheaper than conceding a shot.
TEST_CASE("own goal threat does not care who touched last") {
	Scenario blueTouched = BallAtBlueNet();
	blueTouched.state.lastTouchCarID = blueTouched.state.players[BLUE].carId;
	blueTouched.prev.lastTouchCarID = blueTouched.state.lastTouchCarID;

	Scenario orangeTouched = BallAtBlueNet();

	CHECK(OwnGoalThreat(blueTouched, BLUE) ==
		  doctest::Approx(OwnGoalThreat(orangeTouched, BLUE)));
	CHECK(OwnGoalThreat(blueTouched, BLUE) < 0.f);
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

// The shot latch is where that bug was found, and it is unwired now, so the
// stack sweep above no longer covers it.
TEST_CASE("the shot latch pays out, and repeats for the same state") {
	Scenario shot = ShotScenario();
	shot.Link();

	std::unique_ptr<Reward> reward(new ShotOnTargetReward());
	reward->Reset(shot.prev);
	reward->PreStep(shot.prev);
	reward->GetAllRewards(shot.prev, false);
	reward->PreStep(shot.state);

	const float first = reward->GetAllRewards(shot.state, false)[BLUE];
	const float second = reward->GetAllRewards(shot.state, false)[BLUE];

	CHECK(first > 0.f);
	CHECK(second == doctest::Approx(first));
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

TEST_CASE("a small pad pays the same whether the tank is empty or part-full") {
	const float fromEmpty = PadPickup(0.f, 12.f);
	const float fromHalf = PadPickup(60.f, 72.f);

	CHECK(fromEmpty ==
		  doctest::Approx(PadAwarePickupBoostReward::SMALL_PAD_REWARD));
	CHECK(fromHalf == doctest::Approx(fromEmpty));
}

TEST_CASE("a small pad beats what the sqrt curve could ever pay for one") {
	CHECK(PadPickup(0.f, 12.f) > std::sqrt(0.12f));
}

TEST_CASE("a big pad keeps the sqrt curve, and so does topping off to full") {
	CHECK(PadPickup(30.f, 100.f) == doctest::Approx(1.f - std::sqrt(0.30f)));
	CHECK(PadPickup(92.f, 100.f) == doctest::Approx(1.f - std::sqrt(0.92f)));
	CHECK(PadPickup(92.f, 100.f) < PadAwarePickupBoostReward::SMALL_PAD_REWARD);
}

TEST_CASE("no boost gained pays nothing") {
	CHECK(PadPickup(50.f, 50.f) == doctest::Approx(0.f));
	CHECK(PadPickup(50.f, 20.f) == doctest::Approx(0.f));
}

// AerialDistanceReward is unwired (see GeneralRewardSpecs), but is exercised
// bare here since it ported the height axis wrong: the Python source reads
// position.z (up), the port read pos.y (the goal-to-goal axis) instead.
TEST_CASE("aerial distance's touch reward reads height, not field position") {
	Chain grounded;
	grounded.steps = {AerialDistanceStep(Vec(0, 4000, 17), Vec(0, 4000, 17), false),
					  AerialDistanceStep(Vec(0, 4000, 17), Vec(0, 4000, 17), true)};

	Chain airborne;
	airborne.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
					  AerialDistanceStep(Vec(0, 0, 2000), Vec(0, 0, 2000), true)};

	float groundedPaid =
		ChainRewards(new AerialDistanceReward(), grounded, BLUE).back();
	float airbornePaid =
		ChainRewards(new AerialDistanceReward(), airborne, BLUE).back();

	CHECK(groundedPaid == doctest::Approx(0.f));
	CHECK(airbornePaid == doctest::Approx((2000.f - 256.f) / BACK_WALL_Y));
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

TEST_CASE("a repeat touch while still airborne pays for distance travelled "
		 "since the last touch") {
	Chain chain;
	chain.steps = {
		AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
		AerialDistanceStep(Vec(0, 0, 1000), Vec(0, 0, 1000), true),
		AerialDistanceStep(Vec(500, 0, 1000), Vec(300, 0, 1000), false),
		AerialDistanceStep(Vec(500, 0, 1000), Vec(300, 0, 1000), true),
	};

	std::vector<float> paid = ChainRewards(new AerialDistanceReward(), chain, BLUE);

	CHECK(paid[1] > 0.f);
	CHECK(paid[3] == doctest::Approx((500.f + 300.f) / BACK_WALL_Y));
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
// (w1 == w2) == 0, true whenever the weights merely differ, and the port also
// mutated the member weights permanently instead of using local fallbacks.
TEST_CASE("an asymmetric weight is not clobbered back to equal weights") {
	Chain chain;
	chain.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
				  AerialDistanceStep(Vec(0, 0, 1000), Vec(0, 0, 3000), true)};

	AerialDistanceReward *reward = new AerialDistanceReward();
	reward->carDistanceWeight = 0.f;
	reward->ballDistanceWeight = 1.f;

	float paid = ChainRewards(reward, chain, BLUE).back();

	// Weighting ball height only: (0*1000 + 1*3000)/1 - 256 = 2744.
	CHECK(paid == doctest::Approx((3000.f - 256.f) / BACK_WALL_Y));
}

// Train.cpp's metrics probe calls GetReward once more on every reward it does
// not special-case. That threw while the chain lived in GetAllRewards, and a
// second call must not re-pay or re-advance the chain either.
TEST_CASE("the metrics probe's extra GetReward call is safe") {
	Chain chain;
	chain.steps = {AerialDistanceStep(Vec(0, 0, 17), Vec(0, 0, 17), false),
				   AerialDistanceStep(Vec(0, 0, 1500), Vec(0, 0, 1500), true),
				   AerialDistanceStep(Vec(400, 0, 1500), Vec(200, 0, 1500), true)};
	chain.Link();

	AerialDistanceReward reward;
	reward.Reset(chain.steps.front());

	std::vector<float> paid;
	for (GameState &state : chain.steps) {
		reward.PreStep(state);
		paid.push_back(reward.GetAllRewards(state, false)[BLUE]);
		CHECK(reward.GetReward(state.players[BLUE], state, false) ==
			  doctest::Approx(paid.back()));
	}

	CHECK(paid[1] == doctest::Approx((1500.f - 256.f) / BACK_WALL_Y));
	CHECK(paid[2] == doctest::Approx((400.f + 200.f) / BACK_WALL_Y));
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

float FlickReward(GameState state) {
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

	const float inZone = FlickReward(CradleStep(250.f, Vec(), true, true, 5000.f));
	const float carrying = FlickReward(CradleStep(140.f, carried, true, true, 5000.f));
	const float withFlip = FlickReward(CradleStep(140.f, carried, false, true, 200.f));
	const float flicked = FlickReward(CradleStep(140.f, carried, false, false, 200.f));

	CHECK(inZone == doctest::Approx(0.1f));
	CHECK(carrying == doctest::Approx(0.3f));
	CHECK(withFlip > carrying);
	CHECK(flicked > withFlip);
	CHECK(flicked <= 1.f);
}

// Once an opponent closes, holding the carry pays nothing at all -- not even
// the zone rung -- which is what leaves flicking as the only paying option.
TEST_CASE("a challenged carry held on the ground earns nothing") {
	CHECK(FlickReward(CradleStep(140.f, Vec(0, 1000, 0), true, true, 200.f)) ==
		  doctest::Approx(0.f));
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
