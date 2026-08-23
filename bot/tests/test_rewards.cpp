#include <cmath>

#include "doctest/doctest.h"

#include <Config.h>
#include <env/Rewards.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <string>
#include <vector>

using namespace Hive;

TEST_CASE("specs and built rewards agree in count and weight") {
	TrainConfig cfg = {};
	auto specs = GeneralRewardSpecs(cfg);
	auto built = BuildGeneralRewards(cfg);

	REQUIRE(specs.size() == built.size());
	for (size_t i = 0; i < specs.size(); i++) {
		CHECK(specs[i].weight == built[i].weight);
		CHECK(!specs[i].name.empty());
		CHECK(built[i].reward != nullptr);
	}
	for (auto& wr : built)
		delete wr.reward;
}

TEST_CASE("spec names are unique") {
	auto specs = GeneralRewardSpecs(TrainConfig{});
	for (size_t i = 0; i < specs.size(); i++)
		for (size_t j = i + 1; j < specs.size(); j++)
			CHECK(specs[i].name != specs[j].name);
}

// `Player p = {}` value-initializes through CarState's constructor, which sets
// isOnGround = true. Every test below sets it explicitly rather than relying on
// that, because the default is the opposite of what most of these cases want.

TEST_CASE("SpeedToBallReward pays for closing, and nothing for retreating") {
	SpeedToBallReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	// Same z as the ball, so toBall is exactly horizontal.
	p.pos = {0, 0, 93};

	const float V = RLGC::CommonValues::CAR_MAX_SPEED;

	// Straight at the ball at max speed: the term's ceiling.
	p.vel = {0, V, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Half speed at the ball: exactly half. The term is LINEAR in closing
	// speed.
	p.vel = {0, V / 2.f, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.5f).epsilon(1e-4));

	// Perpendicular: closing speed is zero.
	p.vel = {V, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// THE RECTIFICATION. Driving away pays nothing -- it is not punished.
	// Upstream's VelocityPlayerToBallReward returns -1 here. The guide is
	// explicit that moving away should not be punished, and the signed form
	// also lets a circling bot generate large +/- values that cancel to
	// nothing, which is what p1air's RewardShare 0.482 at ~zero net was.
	p.vel = {0, -V, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Motionless pays nothing.
	p.vel = {0, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Sitting on the ball: no direction to close on, and no divide by zero.
	p.pos = s.ball.pos;
	p.vel = {0, V, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

// THE REGRESSION TEST FOR p7approach.
//
// Rectifying this term is what left the p7approach stack with no state that
// could ever be penalised, and the argmax of such a stack is "carry speed in a
// straight line and never turn" -- turning is the only action that costs
// speed. Measured: `Action/Steer Nonzero` 0.160 -> 0.087 over 100M steps while
// `Jump When Grounded Upright` went 0.755 -> 0.878.
//
// If this test ever needs `RS_MAX(0.f, ...)` to pass, the stack has silently
// lost its only cost and something else must supply one first.
TEST_CASE("FaceBall is SIGNED: pointing away is punished, not merely unpaid") {
	RLGC::FaceBallReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	p.pos = {0, 0, 93};

	// Nose on the ball.
	p.rotMat.forward = {0, 1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Side on: no opinion.
	p.rotMat.forward = {1, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Nose away. This must be NEGATIVE.
	p.rotMat.forward = {0, -1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(-1.f).epsilon(1e-4));
	CHECK(r.GetReward(p, s, false) < 0.f);

	// 60 degrees off, the sign is what matters.
	p.rotMat.forward = {0.866f, -0.5f, 0};
	CHECK(r.GetReward(p, s, false) < 0.f);
}

// THE REGRESSION TEST FOR p9rel.
//
// A flat per-step touch reward IS a dribble reward. Once the relative
// observation made the bot competent enough to carry the ball, it did: steps
// per contact sequence went 1.16 -> 1.98, contact occurred on 13% of ALL steps,
// and `RewardShare/Touch` reached 0.741. Roadmap D4 bans possession rewards;
// this is how one arrives by accident.
TEST_CASE("carrying the ball pays once, not once per step") {
	TouchEdgeReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	// Not touching.
	p.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// First step of contact, no history: a genuine new touch.
	p.ballTouchedStep = true;
	CHECK(r.GetReward(p, s, false) == 1.f);

	// Second consecutive step of contact pays NOTHING. This is the whole
	// difference from the p8ref/p9rel term.
	RLGC::Player prev = p;
	prev.ballTouchedStep = true;
	p.prev = &prev;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Re-arriving after losing contact pays again.
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 1.f);

	// A 180-step carry is worth exactly one touch, not 180.
	float carried = 0.f;
	RLGC::Player cur = {}, last = {};
	cur.ballTouchedStep = true;
	for (int i = 0; i < 180; i++) {
		cur.prev = i ? &last : nullptr;
		carried += r.GetReward(cur, s, false);
		last = cur;
	}
	CHECK(carried == doctest::Approx(1.f));
}

// Direction is the whole point. p11 showed a force-only term cannot stop the
// poke farm: Touch/Hit Force fell 878 -> 551, below StrongTouch's own 555.6
// floor, so the average touch earned nothing from it while TouchEdge doubled.
TEST_CASE("TouchGoalAccel pays for moving the ball toward the right net") {
	// Exponent 1 isolates DIRECTION, which is what this case is about. The
	// convexity gets its own case below.
	TouchGoalAccelReward r(1.f);

	RLGC::GameState prev = {};
	RLGC::GameState s = {};
	s.prev = &prev;
	// Ball at the centre circle, so "toward orange" is +y and "toward blue" -y.
	prev.ball.pos = s.ball.pos = {0, 0, 93};

	RLGC::Player blue = {};
	blue.team = Team::BLUE;
	blue.ballTouchedStep = true;

	auto rewardFor = [&](Vec before, Vec after) {
		prev.ball.vel = before;
		s.ball.vel = after;
		return r.GetReward(blue, s, false);
	};

	const float FULL = RLGC::Math::KPHToVel(130);

	// Blue attacks the orange net at +y. Speeding the ball that way pays.
	CHECK(rewardFor({0, 0, 0}, {0, FULL, 0}) == doctest::Approx(1.f).epsilon(0.02));
	CHECK(rewardFor({0, 0, 0}, {0, FULL / 2, 0}) == doctest::Approx(0.5f).epsilon(0.02));

	// THE SIGN. Putting it toward your own net is NEGATIVE -- no term in this
	// project has ever expressed that.
	CHECK(rewardFor({0, 0, 0}, {0, -FULL, 0}) == doctest::Approx(-1.f).epsilon(0.02));

	// A poke sideways moves the ball but not toward either net: ~zero. This is
	// the farm p11 converged on.
	CHECK(rewardFor({0, 0, 0}, {FULL, 0, 0}) == doctest::Approx(0.f).epsilon(0.02));

	// Orange gets the mirror image for the identical ball motion.
	RLGC::Player orange = blue;
	orange.team = Team::ORANGE;
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, FULL, 0};
	CHECK(r.GetReward(orange, s, false) == doctest::Approx(-1.f).epsilon(0.02));

	// No touch, no reward, however the ball is moving.
	blue.ballTouchedStep = false;
	CHECK(rewardFor({0, 0, 0}, {0, FULL, 0}) == 0.f);
}

// THE p13strike THESIS. A LINEAR touch term is indifferent to concentration:
// the goal-directed delta-v needed to score is fixed by the length of the
// field, so five 400 uu/s pokes pay exactly what one 2000 uu/s strike pays.
// Every other term in the stack breaks that tie toward the pokes, because a
// poke leaves the ball inside re-contact range -- and `Touch/Hit Force` duly
// fell 878 (p10) -> 551 (p11) -> 422 (p12), below RLGymCPP's own 20 kph
// "weak touch" floor of 555.6 uu/s.
TEST_CASE("TouchGoalAccel is convex, so one strike beats five pokes") {
	RLGC::GameState prev = {};
	RLGC::GameState s = {};
	s.prev = &prev;
	prev.ball.pos = s.ball.pos = {0, 0, 93};

	RLGC::Player blue = {};
	blue.team = Team::BLUE;
	blue.ballTouchedStep = true;

	auto pay = [&](TouchGoalAccelReward& r, float kph) {
		prev.ball.vel = {0, 0, 0};
		s.ball.vel = {0, RLGC::Math::KPHToVel(kph), 0};
		return r.GetReward(blue, s, false);
	};

	TouchGoalAccelReward linear(1.f);
	TouchGoalAccelReward convex(2.f);

	// The indifference, stated: linearly, five 20 kph pokes equal one 100 kph
	// strike exactly. This is the equilibrium p12 converged to.
	CHECK(5.f * pay(linear, 20.f) == doctest::Approx(pay(linear, 100.f)).epsilon(0.02));

	// Convex, the same five pokes are worth a fifth of the strike.
	CHECK(5.f * pay(convex, 20.f) ==
	      doctest::Approx(pay(convex, 100.f) / 5.f).epsilon(0.02));

	// The unit is unchanged: a maximal 130 kph goal-directed strike is still
	// exactly 1.0, so every budget written before p13 still reads in the same
	// currency.
	CHECK(pay(convex, 130.f) == doctest::Approx(1.f).epsilon(0.02));

	// 80 kph is what a "strong touch" should mean; 20 kph is RLGymCPP's
	// library default and is decidedly weak. Convexity must separate them by
	// much more than the 4x a linear term gives.
	CHECK(pay(linear, 80.f) / pay(linear, 20.f) == doctest::Approx(4.f).epsilon(0.05));
	CHECK(pay(convex, 80.f) / pay(convex, 20.f) == doctest::Approx(16.f).epsilon(0.05));

	// A soft floor, not a hard one: a weak touch is worth little but NOT zero,
	// so there is still a gradient pointing up. A hard cutoff at 80 kph would
	// read identically zero today -- the mean touch is 15.2 kph -- which is
	// p11's inert-term failure amplified.
	CHECK(pay(convex, 20.f) > 0.f);
	CHECK(pay(convex, 20.f) < 0.05f);

	// Sign survives the power: own-net deliveries are punished convexly too.
	CHECK(pay(convex, -80.f) == doctest::Approx(-pay(convex, 80.f)).epsilon(0.02));
}

TEST_CASE("TouchGoalAccel semi-zero-sum penalizes opponent touches at 50%") {
	std::unique_ptr<RLGC::Reward> r(new RLGC::ZeroSumReward(new TouchGoalAccelReward(1.f), 0.f, 0.5f));

	RLGC::GameState prev = {};
	RLGC::GameState s = {};
	s.prev = &prev;
	prev.ball.pos = s.ball.pos = {0, 0, 93};

	RLGC::Player blue = {};
	blue.team = Team::BLUE;
	blue.carId = 0;

	RLGC::Player orange = {};
	orange.team = Team::ORANGE;
	orange.carId = 1;

	s.players = {blue, orange};

	const float FULL = RLGC::Math::KPHToVel(130);

	// Scenario 1: Blue strikes ball toward Orange net (+y) at FULL speed
	s.players[0].ballTouchedStep = true;
	s.players[1].ballTouchedStep = false;
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, FULL, 0};

	auto rewards1 = r->GetAllRewards(s, false);
	// Blue receives +1.0; Orange is penalized 50% (-0.5)
	CHECK(rewards1[0] == doctest::Approx(1.f).epsilon(0.02));
	CHECK(rewards1[1] == doctest::Approx(-0.5f).epsilon(0.02));

	// Scenario 2: Half-power strike by Blue
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, FULL / 2.f, 0};
	auto rewards2 = r->GetAllRewards(s, false);
	CHECK(rewards2[0] == doctest::Approx(0.5f).epsilon(0.02));
	CHECK(rewards2[1] == doctest::Approx(-0.25f).epsilon(0.02));

	// Scenario 3: Orange strikes ball toward Blue net (-y) at FULL speed
	s.players[0].ballTouchedStep = false;
	s.players[1].ballTouchedStep = true;
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, -FULL, 0};
	auto rewards3 = r->GetAllRewards(s, false);
	// Blue is penalized 50% (-0.5); Orange receives +1.0
	CHECK(rewards3[0] == doctest::Approx(-0.5f).epsilon(0.02));
	CHECK(rewards3[1] == doctest::Approx(1.f).epsilon(0.02));

	// Scenario 4: Blue own-goal touch toward Blue net (-y) at FULL speed
	s.players[0].ballTouchedStep = true;
	s.players[1].ballTouchedStep = false;
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, -FULL, 0};
	auto rewards4 = r->GetAllRewards(s, false);
	// Blue receives -1.0; Orange receives +0.5 bonus
	CHECK(rewards4[0] == doctest::Approx(-1.f).epsilon(0.02));
	CHECK(rewards4[1] == doctest::Approx(0.5f).epsilon(0.02));

	// Scenario 5: Neither car touches ball
	s.players[0].ballTouchedStep = false;
	s.players[1].ballTouchedStep = false;
	prev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, FULL, 0};
	auto rewards5 = r->GetAllRewards(s, false);
	CHECK(rewards5[0] == 0.f);
	CHECK(rewards5[1] == 0.f);
}

TEST_CASE("GeneralRewardSpecs builds TouchGoalAccel as ZeroSumReward with configured opponent scale") {
	TrainConfig cfg = {};
	cfg.rewards.touchGoalAccelOpponentScale = 0.5f;
	auto specs = GeneralRewardSpecs(cfg);

	RLGC::Reward* reward = nullptr;
	for (auto& s : specs) {
		if (s.name == "TouchGoalAccel") {
			reward = s.make();
			break;
		}
	}
	REQUIRE(reward != nullptr);

	auto* zeroSum = dynamic_cast<RLGC::ZeroSumReward*>(reward);
	REQUIRE(zeroSum != nullptr);
	CHECK(zeroSum->opponentScale == doctest::Approx(0.5f));
	CHECK(zeroSum->teamSpirit == doctest::Approx(0.f));

	delete reward;
}


// The min() is the anti-farm device and the reason this term is safe to add to
// a bot that already reaches high balls by driving up the wall.
TEST_CASE("AirTouch pays nothing for a wall shot, however high") {
	// Exponent 1 isolates the min() -- the anti-wall-shot device -- which is
	// what this case is about. The height convexity gets its own case below.
	AirTouchReward r(1.f);
	RLGC::GameState s = {};
	RLGC::Player p = {};
	p.ballTouchedStep = true;

	// Straight at the orange net, so the direction factor is exactly 1 and
	// this case still isolates the min().
	auto rewardFor = [&](float airTime, float ballZ) {
		p.airTime = airTime;
		s.ball.pos = {0, 0, ballZ};
		s.ball.vel = {0, 1000, 0};
		return r.GetReward(p, s, false);
	};

	// A car on a wall is isOnGround, so airTime is 0. Ball at the ceiling and
	// it still pays EXACTLY zero -- this is the "plat wall-shot" the guide
	// names, and the behaviour this bot already runs.
	CHECK(rewardFor(0.f, RLGC::CommonValues::CEILING_Z) == 0.f);

	// Floating forever at ground level also pays nothing.
	CHECK(rewardFor(5.f, 93.f) == doctest::Approx(93.f / RLGC::CommonValues::CEILING_Z));
	CHECK(rewardFor(5.f, 0.f) == 0.f);

	// A genuine aerial: 1.2 s aloft, ball at 1000. min(0.686, 0.489) = 0.489.
	CHECK(rewardFor(1.2f, 1000.f) == doctest::Approx(0.489f).epsilon(0.02));

	// Saturates at the guide's 1.75 s: more air time is not worth more, so
	// floating is not the strategy.
	CHECK(rewardFor(1.75f, RLGC::CommonValues::CEILING_Z) == doctest::Approx(1.f));
	CHECK(rewardFor(10.f, RLGC::CommonValues::CEILING_Z) == doctest::Approx(1.f));

	// No touch, no reward.
	p.ballTouchedStep = false;
	CHECK(rewardFor(1.75f, 1500.f) == 0.f);
}

// Not optional at p13's budget. The p12 form paid on every contact STEP, so an
// air carry at ceiling height would have earned ~170 touch-units per second --
// p9rel's dribble farm, relocated to the air. It stayed harmless only because
// the budget was too small for anything to happen.
// AirTouch was DIRECTION-BLIND, and at 19.1% of reward mass that made it the
// largest term in the stack with no opinion about which net the ball was
// heading for. Carrying the ball back into your own half paid exactly what
// carrying it at their net paid.
//
// TouchGoalAccel could not supply the missing signal, because convexity
// suppresses it quadratically and an air dribble is a sequence of very soft
// touches: at x = 0.05 the direction term pays 0.11 touch-units against
// AirTouch's 3.18 per contact.
TEST_CASE("an air touch toward the wrong net pays nothing") {
	AirTouchReward r(1.f);
	RLGC::GameState s = {};
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	p.team = Team::BLUE;
	p.airTime = 1.75f;
	s.ball.pos = {0, 0, RLGC::CommonValues::CEILING_Z};

	// Straight at the orange net: full value.
	s.ball.vel = {0, 1500, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-3));

	// Straight back at its own net: nothing at all.
	s.ball.vel = {0, -1500, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// And the same touch is full value for the other team.
	p.team = Team::ORANGE;
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-3));
}

TEST_CASE("the air direction factor is smooth, so aerials are not a knife edge") {
	AirTouchReward r(1.f);
	RLGC::GameState s = {};
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	p.team = Team::BLUE;
	p.airTime = 1.75f;
	s.ball.pos = {0, 0, RLGC::CommonValues::CEILING_Z};

	// Sideways: half. There is gradient everywhere between 0 and 1, so a
	// slightly misaimed aerial is worth slightly less rather than nothing --
	// the aerial game has to survive this change.
	s.ball.vel = {1500, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.5f).epsilon(1e-3));

	// 60 degrees off: 0.5 + 0.5*cos(60) = 0.75.
	s.ball.vel = {1500.f * 0.8660254f, 1500.f * 0.5f, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.75f).epsilon(1e-3));

	// Monotone in the angle.
	s.ball.vel = {0, 1500, 0};
	const float straight = r.GetReward(p, s, false);
	s.ball.vel = {600, 1500, 0};
	const float slight = r.GetReward(p, s, false);
	s.ball.vel = {1500, 600, 0};
	const float wide = r.GetReward(p, s, false);
	CHECK(straight > slight);
	CHECK(slight > wide);
	CHECK(wide > 0.f);
}

// The direction curve is a config field so it can be sharpened on evidence
// rather than argued about. At exponent 1 a purely SIDEWAYS carry still keeps
// half its payment, which may well be too generous -- but that is a question
// for AirTouch/Direction Factor to answer, not for a guess to settle.
TEST_CASE("the air direction exponent sharpens the curve without moving its ends") {
	RLGC::GameState s = {};
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	p.team = Team::BLUE;
	p.airTime = AirTouchReward::MAX_AIR_TIME;
	s.ball.pos = {0, 0, RLGC::CommonValues::CEILING_Z};

	AirTouchReward soft(1.f, 1.f);
	AirTouchReward sharp(1.f, 3.f);

	// The ends are fixed by construction: straight at the net is always full
	// value, straight backwards is always nothing, whatever the exponent.
	s.ball.vel = {0, 1500, 0};
	CHECK(soft.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-3));
	CHECK(sharp.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-3));

	s.ball.vel = {0, -1500, 0};
	CHECK(soft.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));
	CHECK(sharp.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Sideways is where the exponent does its work: 0.5 against 0.125.
	s.ball.vel = {1500, 0, 0};
	CHECK(soft.GetReward(p, s, false) == doctest::Approx(0.5f).epsilon(1e-3));
	CHECK(sharp.GetReward(p, s, false) == doctest::Approx(0.125f).epsilon(1e-3));

	// And it stays monotone, so there is still a gradient to turn toward the
	// net from anywhere.
	s.ball.vel = {1500, 600, 0};
	const float wide = sharp.GetReward(p, s, false);
	s.ball.vel = {600, 1500, 0};
	const float near = sharp.GetReward(p, s, false);
	CHECK(near > wide);
	CHECK(wide > 0.f);
}

TEST_CASE("an air CARRY pays once, not once per step") {
	// Carried at the opponent's net, so the direction factor is 1 and this
	// case still isolates the rising edge.
	AirTouchReward r(1.f);
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, RLGC::CommonValues::CEILING_Z};
	s.ball.vel = {0, 1000, 0};

	RLGC::Player prev = {};
	prev.ballTouchedStep = true;

	RLGC::Player p = {};
	p.ballTouchedStep = true;
	p.airTime = AirTouchReward::MAX_AIR_TIME;

	// First contact of a sequence: prev is null (episode reset) or not
	// touching. Pays in full.
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));

	// Still glued to the ball on the next step: pays nothing more. Same rule
	// and same reason as TouchEdgeReward.
	p.prev = &prev;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Let go and re-connect, and it is a new contact again.
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));
}

// p13strike measured the linear form failing: at ball height ~350 with 0.9 s
// aloft, min(0.52, 0.171) = 0.171, so a plain jump-touch collected the term and
// `RewardShare/AirTouch` rose to 0.047 -- ABOVE its 0.030 target -- while
// `Touch/Above 450` FELL 0.037 -> 0.015. The term paid for the behaviour that
// replaced aerials.
TEST_CASE("AirTouch height is convex, so higher pays disproportionately") {
	RLGC::GameState s = {};
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	p.airTime = AirTouchReward::MAX_AIR_TIME; // never let air time bind

	// Goal-ward, so the direction factor is 1 and this case isolates height.
	auto pay = [&](AirTouchReward& r, float z) {
		s.ball.pos = {0, 0, z};
		s.ball.vel = {0, 1000, 0};
		return r.GetReward(p, s, false);
	};

	AirTouchReward linear(1.f);
	AirTouchReward convex(2.f);

	// A jump-touch at z 350 is STILL PAID. A jump taken to reach a high ball is
	// a real aerial, just a small one, and gating it to zero would remove the
	// gradient that leads to a bigger one. It simply must not be worth what a
	// genuine aerial is worth.
	CHECK(pay(convex, 350.f) > 0.f);

	// z 800 against z 300: 2.7x linear, 7.1x convex.
	CHECK(pay(linear, 800.f) / pay(linear, 300.f) == doctest::Approx(2.67f).epsilon(0.02));
	CHECK(pay(convex, 800.f) / pay(convex, 300.f) == doctest::Approx(7.11f).epsilon(0.02));

	// Learnable from where the bot actually is: the gradient at the current
	// operating point must be a real fraction of the target one. Same
	// derivation that set TouchGoalAccel's exponent.
	CHECK(pay(convex, 300.f) / pay(convex, 800.f) > 0.10f);

	// The ceiling is still exactly 1.0, so the budget unit is unchanged.
	CHECK(pay(convex, RLGC::CommonValues::CEILING_Z) == doctest::Approx(1.f));

	// And a wall shot is still worth zero however high, because the min() with
	// air time is untouched.
	p.airTime = 0.f;
	CHECK(pay(convex, RLGC::CommonValues::CEILING_Z) == 0.f);
}

TEST_CASE("Air pays for being airborne") {
	RLGC::AirReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.isOnGround = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	p.isOnGround = false;
	CHECK(r.GetReward(p, s, false) == 1.f);
}

TEST_CASE("budget conversion is the only route to a per-step weight") {
	TrainConfig cfg = {};
	auto specs = GeneralRewardSpecs(cfg);

	auto weightOf = [&](const std::string& name) {
		for (auto& s : specs)
			if (s.name == name)
				return s.weight;
		FAIL("no spec named " << name);
		return 0.f;
	};

	// Rate budgets are episode integrals, so the per-step weight must be the
	// budget divided by the reference episode -- never written directly.
	CHECK(weightOf("SpeedToBall") ==
	      doctest::Approx(cfg.rewards.speedToBall / REFERENCE_EPISODE_STEPS));
	CHECK(weightOf("FaceBall") ==
	      doctest::Approx(cfg.rewards.faceBall / REFERENCE_EPISODE_STEPS));
	CHECK(weightOf("Air") ==
	      doctest::Approx(cfg.rewards.air / REFERENCE_EPISODE_STEPS));
	CHECK(weightOf("SaveBoost") ==
	      doctest::Approx(cfg.rewards.saveBoost / REFERENCE_EPISODE_STEPS));

	// Event budgets are per occurrence, so they are the weight unchanged.
	CHECK(weightOf("TouchEdge") == doctest::Approx(cfg.rewards.touchEdge));
	CHECK(weightOf("PickupBoost") == doctest::Approx(cfg.rewards.pickupBoost));
	CHECK(weightOf("TouchGoalAccel") == doctest::Approx(cfg.rewards.touchGoalAccel));
	CHECK(weightOf("Goal") == doctest::Approx(cfg.rewards.goal));
	CHECK(weightOf("AirTouch") == doctest::Approx(cfg.rewards.airTouch));
}

// The currency is one ball touch. `strongTouch = 3.0` means a MAXIMAL strike
// (3611 uu/s of delta-v) is worth three of them, which is the right shape --
// a maximal hit is a big deal. What must stay true is the ordering: connecting
// beats merely arriving.
TEST_CASE("connecting outranks arriving") {
	const RewardBudget b = {};
	CHECK(b.touchEdge < b.touchGoalAccel);
	CHECK(b.touchEdge < b.airTouch);
}

// The intuition that a goal should dominate is the one the guide argues
// against hardest: "A giant goal reward will drown out every other reward you
// have." In self-play a goal is +1 for one car and -1 for the other, so its
// mean is zero and scaling it scales variance the critic cannot predict.
//
// Restated for p13: the raw budget numbers are no longer comparable to each
// other, because touchGoalAccel is now convex and denominated in MAXIMAL
// strikes that essentially never occur. Everything below is therefore checked
// in REALIZED terms, against p12goal's measured rates.
namespace {
// p15manual, last 60 iterations at 690M steps. p12's rates are three runs
// stale and every one of them moved: the mean touch is 6.7x more
// goal-directed, episodes are 1.6x longer and contact is half as frequent.
// Sizing a budget against them would be sizing it for a bot that no longer
// exists, which is the mistake this block was written to prevent.
//
// All of these are RAW per-step means from `Rewards/*`, not `RewardShare/*`.
// The share column averages an already-normalized ratio, so it is biased low
// for spiky terms -- it disagrees with the raw means by up to 2.9x on exactly
// the event terms these budgets turn on. See RewardMass/* in Train.cpp.
// Re-measured 2026-08-20 from RewardMass/* over 831-839M, which is the FIRST
// unbiased read this project has had. The previous goal rate here (3.05e-4)
// came from RewardShare/Goal and was wrong by 5.3x -- goals arrive 1.07 times
// per episode, not 0.19 -- which is how `goal` came to be sized at 50 and ended
// up holding 36% of all reward mass.
constexpr float P15_EPISODE_STEPS = 392.f;
constexpr float P15_EDGE_RATE = 0.02018f;         // Touch/Edge Rate
constexpr float P15_SPEED_TO_BALL = 0.34250f;     // raw mean
constexpr float P15_FACE_BALL = 0.58290f;         // raw mean
constexpr float P15_TOUCH_GOAL_ACCEL = 1.661e-3f;
constexpr float P15_TOUCH_EDGE = 0.019330f;
constexpr float P15_PICKUP_BOOST = 4.5721e-3f;
constexpr float P15_SAVE_BOOST = 0.33200f;
constexpr float P15_AIR = 0.38818f;
constexpr float P15_FLIP_SPEED = 5.885e-3f;
constexpr float P15_WRONG_SURFACE_RATE = 0.01300f;

// Goals per step from RewardMass/Goal: 1.04 per episode.
constexpr float P15_GOAL_RATE = 2.648e-3f;

// AirTouch's raw mean depends on the HEIGHT EXPONENT, so this is projected for
// exponent 1 from the exponent-2 rate measured at 1025M, scaled by
// heightFrac^1 / heightFrac^2 = 1/heightFrac at the measured touch height.
constexpr float P15_AIR_TOUCH = 2.99e-3f;

// ShotOnTarget's realized rate was measured while the term was a FLAT plateau.
// The strength factor multiplies it by the mean strength of an on-target
// touch, which nothing has measured yet -- Shot/Strength is published for
// exactly this, and this constant should be replaced by it after one run.
// Named rather than folded in, so the guess stays visible.
constexpr float P15_SHOT_ON_TARGET_FLAT = 2.0067e-3f;
// Measured on the p17cal probe, no longer assumed.
constexpr float ASSUMED_SHOT_STRENGTH = 0.5247f;
constexpr float P15_SHOT_ON_TARGET = P15_SHOT_ON_TARGET_FLAT * ASSUMED_SHOT_STRENGTH;

// Where the bot ACTUALLY touches the ball, measured at 1025M. The aerial
// budget has to pay for itself here, not at the height we wish it played at.
constexpr float P15_TOUCH_HEIGHT = 191.f;

// What continuing is worth, which is the only quantity the goal budget has to
// beat: Critic/V All 0.259 x GAE/Returns STD 51.66, at gamma 0.99.
constexpr float P15_STATE_VALUE = 13.4f;

// The measured price of leaving the ground, in touch-units: Critic/TD Delta
// Jump -0.01119 against NoJump 0.00027, so -0.01146 standardized, times
// GAE/Returns STD 59.2. p12 measured 0.81 against a stack an order of
// magnitude smaller, so in relative terms a takeoff has gone from expensive to
// very nearly free. That is what p15's air and flip numbers are made of.
constexpr float TAKEOFF_COST = 0.68f;

// A touch worth calling strong, per the 80 kph target, scored convexly. This
// is what ONE GOOD HIT is worth, and it is 27x the mean -- which is the whole
// reason the two must never be confused when sizing a budget.
constexpr float STRONG_TOUCH_VALUE = (80.f / 130.f) * (80.f / 130.f); // 0.379

// Per-episode realized mass, in touch-units, for the blocks the budgets are
// balanced between. Rate terms integrate their raw mean over the episode;
// event terms are already per-step rates.
float RatePerEp(float budget, float rawMean) {
	return RateWeight(budget) * rawMean * P15_EPISODE_STEPS;
}

float EventPerEp(float budget, float rawRate) {
	return budget * rawRate * P15_EPISODE_STEPS;
}

// The whole realized ledger at measured rates. AirTouch is counted UNDIRECTED,
// which over-counts it now that the direction factor is in -- deliberately, so
// every share bound below is conservative.
float LedgerPerEp(const RewardBudget& b) {
	return EventPerEp(b.touchGoalAccel, P15_TOUCH_GOAL_ACCEL) +
		   EventPerEp(b.goal, P15_GOAL_RATE) +
		   EventPerEp(b.shotOnTarget, P15_SHOT_ON_TARGET) +
		   EventPerEp(b.touchEdge, P15_TOUCH_EDGE) +
		   EventPerEp(b.pickupBoost, P15_PICKUP_BOOST) +
		   EventPerEp(b.flipSpeed, P15_FLIP_SPEED) +
		   EventPerEp(b.airTouch, P15_AIR_TOUCH) +
		   RatePerEp(b.speedToBall, P15_SPEED_TO_BALL) +
		   RatePerEp(b.faceBall, P15_FACE_BALL) +
		   RatePerEp(b.saveBoost, P15_SAVE_BOOST) +
		   RatePerEp(b.air, P15_AIR) +
		   PerSecondWeight(b.wrongSurface) * P15_WRONG_SURFACE_RATE * P15_EPISODE_STEPS;
}
} // namespace

TEST_CASE("the goal reward is decisive but not dominant") {
	const RewardBudget b = {};

	// THE MARGINAL CONDITION, and the only one that decides whether the bot
	// shoots at an open net: scoring pays `goal` and ENDS the episode,
	// forfeiting the remaining stream. So the comparison is against the
	// measured continuation value, not against a whole episode's touches.
	//
	// The earlier form of this test compared `goal` to the whole-episode touch
	// stream, which was written when goals were believed to arrive 0.19 times
	// per episode. They arrive 1.07 times. At roughly one goal per episode the
	// undiscounted whole-episode comparison is the wrong question entirely.
	CHECK(b.goal > P15_STATE_VALUE);

	// And not so large that it drowns the shaping it exists to break ties
	// between. In self-play a goal is +1 for one car and -1 for the other, so
	// its mean is exactly zero and its whole contribution is variance the
	// critic cannot predict -- scaling it scales noise, not signal. At 50 it
	// held 36% of reward mass, which is the guide's named failure.
	CHECK(EventPerEp(b.goal, P15_GOAL_RATE) < 0.30f * LedgerPerEp(b));

	// THE SHOT-FARM GUARD. Convexity pays a lot for one good hit, and the
	// failure mode it invites is blasting the ball goalward over and over
	// without scoring. One strong touch must not out-earn a goal by enough to
	// make that trade attractive.
	CHECK(b.touchGoalAccel * STRONG_TOUCH_VALUE < 3.f * b.goal);
}

// The accuracy half of the finishing block. TouchGoalAccel projects onto the
// direction to the goal CENTRE, so a shot from 4000 uu that misses the goal
// ENTIRELY still earns 83% of a perfect one -- less than the 19% it loses from
// a 10% drop in power. Power had a gradient and accuracy had none, which is
// the whole reason shots go near the post.
TEST_CASE("aiming is worth about as much as hitting hard") {
	const RewardBudget b = {};

	const float meanTouch = b.touchGoalAccel * (P15_TOUCH_GOAL_ACCEL / P15_EDGE_RATE);

	// On target against half a goal-width wide, per contact sequence. This has
	// to be a real fraction of what the touch itself pays or the accuracy
	// gradient is again lost in the power gradient.
	const float halfWide = std::exp(-0.5f);
	const float aimingDelta = b.shotOnTarget * ASSUMED_SHOT_STRENGTH * (1.f - halfWide);
	CHECK(aimingDelta > 0.25f * meanTouch);

	// An on-target shot is a better outcome than an average touch, so it
	// should pay more than one. But it must not out-earn a genuine strike,
	// which is both powerful AND usually on target -- the two stack, and
	// placement must not replace power.
	// Compared at REALIZED value: the budget is no longer what a shot pays,
	// because the strength factor scales it.
	const float shotPays = b.shotOnTarget * ASSUMED_SHOT_STRENGTH;
	CHECK(shotPays > meanTouch);
	CHECK(shotPays < b.touchGoalAccel * STRONG_TOUCH_VALUE);

	// The farm bound: every contact turning into an on-target shot is the
	// objective, not an exploit, but the term still must not become the
	// stack's argmax at the rate actually measured.
	CHECK(EventPerEp(b.shotOnTarget, P15_SHOT_ON_TARGET) < 0.20f * LedgerPerEp(b));
}

// THE INVERSION, and it is the point of p13strike. Every previous run in this
// project paid more for BEING NEAR the ball than for what the ball did:
// RewardShare SpeedToBall + FaceBall was 0.761 (p8ref), 0.876 (p10), 0.778
// (p11), 0.606 (p12), and no run ever tested reducing it.
TEST_CASE("the ball now outranks the chase") {
	const RewardBudget b = {};

	const float proximityPerEp = RatePerEp(b.speedToBall, P15_SPEED_TO_BALL) +
								 RatePerEp(b.faceBall, P15_FACE_BALL) +
								 EventPerEp(b.touchEdge, P15_EDGE_RATE);

	// Realized, at p15's OWN touch strengths -- not at the strength the run is
	// trying to buy. Sizing a budget off the outcome you want rather than the
	// one you have is precisely how p12's AirTouch shipped at 0.008 share.
	const float ballPerEp = EventPerEp(b.touchGoalAccel, P15_TOUCH_GOAL_ACCEL);

	CHECK(ballPerEp > proximityPerEp);

	// Approach is not deleted, but the floor is 0.15 rather than p13's 0.25.
	// That floor existed to protect a capability that was days old and
	// fragile: Velocity Alignment had only just left its 1/pi null. It is not
	// fragile now -- p15 measures alignment 0.563 against the 0.318 null,
	// FaceBall/Mean Cos 0.52, and the bot ends episodes CLOSER to the ball
	// than it starts (Late Ball Dist 1537 against Early 3992), having ended
	// them farther away in every run up to p14. Holding 0.25 now would force
	// the approach budget up for no behavioural reason.
	CHECK(proximityPerEp > 0.15f * ballPerEp);
}

// p12 measured the price of a takeoff directly, under something close to a
// randomized trial: ~91% of sampled jumps come from the exploration floor,
// which mixes uniformly over valid actions independently of state.
TEST_CASE("an aerial pays for itself, with margin") {
	const RewardBudget b = {};

	// AT THE MEASURED TOUCH HEIGHT, which is the whole point. The earlier form
	// of this test asserted break-even at a ball height of 800 -- a height the
	// bot reached when the test was written and does not reach now -- so it
	// passed while the term was 3.9x BELOW break-even in play, and the air
	// game duly switched itself off. That is exactly the error
	// scripts/solve_budgets.py exists to prevent: sizing a budget against the
	// outcome you want rather than the one you have.
	const float heightFrac = P15_TOUCH_HEIGHT / RLGC::CommonValues::CEILING_Z;
	const float paidNow = b.airTouch * std::pow(heightFrac, b.airTouchHeightExponent);
	CHECK(paidNow > TAKEOFF_COST);

	// Deliberate over-payment: this behaviour has now been found and lost
	// THREE times (p12 at 4% below break-even, p14, and again at 1025M), each
	// time at a margin near or below 1.0.
	CHECK(paidNow > 1.5f * TAKEOFF_COST);

	// The convexity is what made the collapse self-reinforcing: at exponent 2
	// a falling touch height cuts the payment QUADRATICALLY, so the term
	// weakens faster than the behaviour does and there is no way back.
	// Learnability at the current operating point, same instrument as the
	// touch exponent below.
	const float target = 800.f / RLGC::CommonValues::CEILING_Z;
	const float gradientFrac =
		std::pow(heightFrac / target, b.airTouchHeightExponent - 1.f);
	CHECK(gradientFrac > 0.5f);

	// And it still cannot be collected from a wall, which is what makes the
	// budget safe at all. Asserted behaviourally above.
	CHECK(b.airTouch > b.touchEdge);

	// Below the FINISHING block, so "get it high" is never the stack's
	// loudest opinion again.
	const float airPerEp = EventPerEp(b.airTouch, P15_AIR_TOUCH);
	const float finishPerEp = EventPerEp(b.goal, P15_GOAL_RATE) +
							  EventPerEp(b.shotOnTarget, P15_SHOT_ON_TARGET);
	CHECK(airPerEp < finishPerEp);
}

// A flip is a TRAVEL primitive worth +500 uu/s. The v1 term measured closing
// speed on the ball at the dodge's rising edge, which pays a forward dodge
// aimed at the ball and pays a side dodge zero, so it could only reinforce the
// contact flips the bot already had. v2 measures the car's own speed across
// the dodge and is blind inside MIN_BALL_DIST.
TEST_CASE("a travel flip is worth taking but cannot become the run's argmax") {
	const RewardBudget b = {};

	// A flip must clear what a takeoff costs, or the bot is right to refuse.
	CHECK(b.flipSpeed > TAKEOFF_COST);

	// The dodge commits FLIP_PITCHLOCK_TIME (1.0 s) of no pitch control and
	// the jump until landing. Over that second a perfect approach earns this,
	// and the flip has to beat it to be the better use of the time.
	const float approachPerSecond =
		RateWeight(b.speedToBall) * P15_SPEED_TO_BALL * STEPS_PER_SECOND;
	CHECK(b.flipSpeed > approachPerSecond);

	// The ceiling, against the MEASURED flip rate rather than a speculative
	// cycle time. p15 realizes 0.0096 flips-worth per step at 780-790M, which
	// is the rate AFTER the behaviour established -- so this is what the term
	// actually spends, not what it could spend in the worst case.
	const float flipPerEp = EventPerEp(b.flipSpeed, P15_FLIP_SPEED);
	CHECK(flipPerEp < 0.5f * EventPerEp(b.touchGoalAccel, P15_TOUCH_GOAL_ACCEL));
}

// The dense terms are still the guide's shape relative to each other, but the
// BLOCK is smaller. What has to be true is that the per-step weight actually
// fell -- the budget NUMBER moving is meaningless on its own, because
// REFERENCE_EPISODE_SECONDS moves with the measured episode, and reading the
// number instead of the weight is exactly how this project spent four runs
// over-delivering every rate budget by 2.28x.
TEST_CASE("the approach budget really was cut, in per-step terms") {
	const RewardBudget b = {};

	// p12goal: speedToBall 17.1 over 171 reference steps.
	constexpr float P12_SPEED_WEIGHT = 17.1f / 171.f; // 0.1000
	constexpr float P12_FACE_WEIGHT = 3.42f / 171.f;  // 0.0200
	constexpr float P12_AIR_WEIGHT = 0.513f / 171.f;  // 0.0030

	CHECK(RateWeight(b.speedToBall) < 0.5f * P12_SPEED_WEIGHT);
	CHECK(RateWeight(b.faceBall) < 0.5f * P12_FACE_WEIGHT);

	// Air is NOT part of the cut: its target share is unchanged, so its weight
	// should be within a factor of two of p12's. The budget number moving is
	// the reference-length correction and nothing else.
	CHECK(RateWeight(b.air) > 0.5f * P12_AIR_WEIGHT);
	CHECK(RateWeight(b.air) < 2.0f * P12_AIR_WEIGHT);

	// The reference episode is the MEASURED one. p15 ran 622-step episodes
	// against a constant that still said 390, so every rate budget was
	// silently delivering 1.6x its declared integral.
	// Within 5%, not 1%: episode length drifts every run and demanding an
	// exact match would force a rate-budget churn each time.
	CHECK(REFERENCE_EPISODE_STEPS == doctest::Approx(P15_EPISODE_STEPS).epsilon(0.05));
}

TEST_CASE("the touch exponent is convex but still learnable") {
	const RewardBudget b = {};

	CHECK(b.touchAccelExponent > 1.f);

	const float ratio = 15.2f / 80.f;
	const float gradientFrac = std::pow(ratio, b.touchAccelExponent - 1.f);
	CHECK(gradientFrac > 0.10f);
}

TEST_CASE("the spec list is the thirteen designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs) {
		names.push_back(s.name);
		CHECK(s.weight > 0.f);
	}

	const std::vector<std::string> expected = {"TouchGoalAccel", "Goal", "ShotOnTarget",
	                                           "Save", "TouchEdge", "SpeedToBall",
	                                           "FaceBall", "SaveBoost", "PickupBoost",
	                                           "FlipSpeed", "AirTouch", "Air",
	                                           "WrongSurface"};
	CHECK(names == expected);
}

// Every one of these was in the stack at some point and each is a standing
// decision to leave out. Goal is noise until the bot can cause one; the rest
// are tuning rewards the guide's troubleshooting section says to remove, and
// WrongSurface held 30% of p7approach's reward mass on its own.
//
// Boost is NOT on this list any more: p10touch measured `Player/Boost` at 7.3
// out of 100 while the bot was trying to air dribble, and the guide prescribes
// a boost economy at exactly this stage.
TEST_CASE("no continuous ball-to-goal or tuning terms are in the stack") {
	auto specs = GeneralRewardSpecs(TrainConfig{});
	for (auto& s : specs) {
		// WrongSurface came OFF this list on 2026-08-20. The ban was never
		// about the term, it was about p7approach's 0.292 share -- and that
		// share was a symptom of a starved stack, not a large weight. p7 ran
		// PerSecondWeight(0.10) against a bot that could not touch the ball,
		// so every event term was near zero and a tiny penalty was 29% of
		// almost nothing. At p15's rates the same weight would be 0.05% of the
		// ledger. The share is now bounded by a test rather than by exclusion.
		CHECK(s.name != "CleanLanding");
		CHECK(s.name != "HarshSpeedLoss");
		CHECK(s.name != "VelBallToGoal");
	}
}

// The boost economy, added in p11 after p10touch measured `Player/Boost` at
// 7.3 out of 100 while the bot was trying to air dribble.
// Wheels-up against a surface: the recovery tax. Deliberately surface-blind
// rather than orientation-based -- `isOnGround` is >=3 wheels on ANY surface,
// so driving up a wall is on-wheels and pays nothing, while sitting on the
// roof or scraping the chassis pays every step.
TEST_CASE("a car wheels-up against a surface is punished every step it stays there") {
	WrongSurfaceReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.worldContact.hasContact = true;
	p.isOnGround = false;
	CHECK(r.GetReward(p, s, false) == -1.f);

	// Still on its roof next step: this is a RATE, so dawdling costs more.
	CHECK(r.GetReward(p, s, false) == -1.f);
}

TEST_CASE("a car on its wheels pays nothing, on any surface") {
	WrongSurfaceReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.worldContact.hasContact = true;
	p.isOnGround = true;
	CHECK(r.GetReward(p, s, false) == 0.f);
}

// The failure mode that would matter most: p15 spends 29% of its steps
// airborne and has only just learned to flip for speed. A term that taxed air
// time would undo both.
TEST_CASE("free flight is not punished, this is not an air tax") {
	WrongSurfaceReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.worldContact.hasContact = false;
	p.isOnGround = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Measured, not assumed: p6budget found this firing on 2.3% of airborne
	// steps, because free flight makes no chassis contact.
}

TEST_CASE("the recovery tax bites per event but cannot matter in aggregate") {
	const RewardBudget b = {};

	const float perSecond = PerSecondWeight(b.wrongSurface) * STEPS_PER_SECOND;

	// It has to beat the takeoff that put the car there, or landing badly is
	// cheaper than the jump and the term changes nothing.
	CHECK(perSecond > TAKEOFF_COST);

	// But a second of it must not out-price the thing the bot is here to do.
	const float meanTouch = b.touchGoalAccel * (P15_TOUCH_GOAL_ACCEL / P15_EDGE_RATE);
	CHECK(perSecond < meanTouch);

	// THE p7approach GUARD, and the reason this term is allowed back. p7 ran
	// it at 0.292 of reward mass. At p15's measured wrong-contact rate this
	// budget is under 3% of what the ball itself is worth, and if
	// RewardShare/WrongSurface is ever seen above 0.05 at runtime the term
	// goes straight back out.
	const float wrongPerEp = PerSecondWeight(b.wrongSurface) *
							 P15_WRONG_SURFACE_RATE * P15_EPISODE_STEPS;
	const float ballPerEp = EventPerEp(b.touchGoalAccel, P15_TOUCH_GOAL_ACCEL);
	CHECK(wrongPerEp < 0.03f * ballPerEp);
}

TEST_CASE("SaveBoost pays sqrt of the tank, so the first drops are worth most") {
	RLGC::SaveBoostReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	auto at = [&](float boost) { p.boost = boost; return r.GetReward(p, s, false); };

	CHECK(at(0.f) == doctest::Approx(0.f));
	CHECK(at(100.f) == doctest::Approx(1.f));
	CHECK(at(25.f) == doctest::Approx(0.5f).epsilon(1e-4));

	// Concave: 0 -> 50 is worth more than 50 -> 100. This is the whole reason
	// the guide specifies sqrt rather than a linear term.
	CHECK((at(50.f) - at(0.f)) > (at(100.f) - at(50.f)));
}

TEST_CASE("TieredPickupBoostReward guarantees a base floor for small pads and full scale for big pads") {
	TieredPickupBoostReward r;
	RLGC::GameState s = {};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;

	auto gain = [&](float from, float to) {
		prev.boost = from;
		cur.boost = to;
		return r.GetReward(cur, s, false);
	};

	// A full grab from empty is the unit of this term.
	CHECK(gain(0.f, 100.f) == doctest::Approx(1.f).epsilon(1e-4));

	// Small pad (+12) from empty gives base floor (0.25) + sqrt increment.
	CHECK(gain(0.f, 12.f) > 0.5f);

	// Small pad (+12) even when mostly full (75 -> 87) stays above the base floor (0.25).
	CHECK(gain(75.f, 87.f) > 0.25f);

	// Spending boost is not punished; only gaining is paid.
	CHECK(gain(100.f, 40.f) == 0.f);
	CHECK(gain(50.f, 50.f) == 0.f);
}

// A flip is worth +500 uu/s (FLIP_INITIAL_VEL_SCALE) whatever direction it is
// aimed, and that is a TRAVEL primitive, not a shooting one. The first version
// of this term measured the change in CLOSING SPEED ON THE BALL at the flip's
// rising edge, which pays a plain forward dodge and pays a side dodge exactly
// zero -- the side impulse only becomes forward speed after the air-roll
// correction, part-way through FLIP_TORQUE_TIME. It therefore could only ever
// reinforce flips already aimed at the ball, which is the behaviour that needs
// no help.
//
// v2 measures the car's own horizontal speed across the whole dodge and gates
// on the ball being far away, so the only thing it can pay for is using a flip
// to cross the field.
TEST_CASE("a forward dodge far from the ball pays for the speed it built") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 6000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};

	// Rising edge: nothing is paid yet, the dodge has not finished.
	prev.vel = {0, 1400, 0};
	prev.isOnGround = false;
	cur.vel = {0, 1900, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	cur.isOnGround = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// Falling edge of isFlipping: the dodge is over, pay for the gain.
	prev = cur;
	cur.isFlipping = false;
	cur.pos = {0, 400, 17};
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("a side dodge converted into speed pays, where the v1 term paid zero") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 6000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};
	prev.isOnGround = cur.isOnGround = false;

	// Rising edge at 1400 forward.
	prev.vel = {0, 1400, 0};
	cur.vel = {500, 1400, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// Mid-dodge the impulse is still mostly sideways: |v| has barely moved,
	// which is exactly the moment v1 sampled and scored zero.
	prev = cur;
	cur.vel = {450, 1500, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// The air-roll correction lands it forward before the dodge ends.
	prev = cur;
	cur.vel = {0, 1900, 0};
	cur.isFlipping = false;
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("the peak speed during the dodge counts, not the speed after landing scrub") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 6000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};

	prev.vel = {0, 1400, 0};
	prev.isOnGround = false;
	cur.vel = {0, 1900, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	cur.isOnGround = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// isFlipping also clears when the car lands mid-dodge (Car.cpp:114), and
	// the wheels scrub speed on contact. Paying the landed value would price
	// the flip below what it actually produced.
	prev = cur;
	cur.vel = {0, 1750, 0};
	cur.isFlipping = false;
	cur.isOnGround = true;
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("a flip taken at the ball pays nothing, because contact flips already pay") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	// Inside FlipSpeedReward::MIN_BALL_DIST.
	s.ball.pos = {0, 800, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};
	prev.isOnGround = cur.isOnGround = false;

	prev.vel = {0, 1400, 0};
	cur.vel = {0, 1900, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	prev = cur;
	cur.isFlipping = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);
}

TEST_CASE("a flip that builds speed away from play pays nothing") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 6000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};
	prev.isOnGround = cur.isOnGround = false;

	prev.vel = {0, 1400, 0};
	cur.vel = {500, 1400, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// Ends the dodge faster, but travelling away from the ball down -Y.
	prev = cur;
	cur.vel = {0, -1900, 0};
	cur.isFlipping = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);
}

TEST_CASE("a flip taken at supersonic pays nothing, there is no headroom") {
	FlipSpeedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 6000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.index = prev.index = 0;
	prev.pos = cur.pos = {0, 0, 17};
	prev.isOnGround = cur.isOnGround = false;

	prev.vel = {0, RLGC::CommonValues::SUPERSONIC_THRESHOLD, 0};
	cur.vel = {0, 2300, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	prev = cur;
	cur.isFlipping = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);
}

// The accuracy term. TouchGoalAccel projects onto the direction to the goal
// CENTRE, so aim enters only through a cosine and a shot from 4000 uu that
// misses the goal entirely still earns 83% of a perfect one -- less than the
// 19% it loses from a 10% drop in power. That is a term with a power gradient
// and no accuracy gradient, and this is the missing half.
//
// It is a PLATEAU across the whole mouth on purpose. Corner shots are usually
// the right shot, so paying for centrality would teach the wrong thing; what
// is being paid for is the difference between on and off target, and `Goal` is
// left to decide where within the mouth is best.
TEST_CASE("a shot anywhere inside the mouth pays the same") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// 2.56 s to the orange goal plane.
	s.ball.pos = {0, 0, 93};

	// The ball was stationary before the touch, so the whole of its velocity
	// is the change this touch caused.
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};

	// Dead centre. Strength is (v . toGoal) / 3611, and toGoal has no x
	// component from here, so every case below shares the same strength and
	// the test still isolates PLACEMENT.
	s.ball.vel = {0, 2000, 0};
	const float centre = r.GetReward(cur, s, false);
	CHECK(centre > 0.f);
	CHECK(centre < 1.f);

	// Just inside the right post: 850 of a 892.755 half-width.
	s.ball.vel = {850.f / 2.56f, 2000, 0};
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(centre).epsilon(1e-4));

	// Just inside the left post.
	s.ball.vel = {-850.f / 2.56f, 2000, 0};
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(centre).epsilon(1e-4));
}

// THE DRIBBLE FARM, and it was a hole in this term's first design. Placement
// was a plateau on purpose -- corner shots are usually the right shot -- but
// nothing depended on how hard the ball was hit, so a ball rolling goalward on
// the car's hood paid EXACTLY what a 2000 uu/s strike paid, on every contact
// rising edge. Measured at 1025M: 4.16 touch-units per contact sequence, one
// sequence every ~50 steps, against a goal worth 25 once. Walking the ball to
// the line was strictly optimal and slower was strictly better.
//
// The discriminator is the CHANGE the touch made, not the ball's speed: a
// dribbled ball travels at the car's speed (Player/Grounded Speed 1309), so an
// absolute-speed factor would still have paid a dribble ~40%. A carried ball
// moves WITH the car, so its delta-v is ~0; a struck ball leaves it.
TEST_CASE("a ball carried on the hood is not a shot, however well aimed") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	s.ball.pos = {0, 3000, 93};
	sPrev.ball.pos = {0, 2900, 93};

	// Dead on target and arriving in under a second -- but the touch changed
	// the ball's velocity by nothing at all, because it is being carried.
	sPrev.ball.vel = {0, 1500, 0};
	s.ball.vel = {0, 1500, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// A nudge of the kind a dribble actually imparts: 40 uu/s.
	s.ball.vel = {0, 1540, 0};
	const float nudge = r.GetReward(cur, s, false);

	// The same contact, but struck.
	s.ball.vel = {0, 3300, 0};
	const float strike = r.GetReward(cur, s, false);

	CHECK(nudge > 0.f);
	CHECK(strike > 40.f * nudge);
}

TEST_CASE("shot strength is LINEAR, because TouchGoalAccel already pays for power") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// Close enough that even the slow case clears MAX_TIME.
	s.ball.pos = {0, 3000, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};

	// Doubling the change doubles the payment. Making this convex too would
	// double-count power against TouchGoalAccel and re-create the blast-it
	// failure that convexity already invites once.
	s.ball.vel = {0, 900, 0};
	const float half = r.GetReward(cur, s, false);
	s.ball.vel = {0, 1800, 0};
	const float full = r.GetReward(cur, s, false);
	CHECK(full == doctest::Approx(2.f * half).epsilon(0.02));

	// Saturates at the same 130 kph TouchGoalAccel uses, so the currency is
	// unchanged: past it, more speed buys nothing.
	s.ball.vel = {0, RLGC::Math::KPHToVel(260), 0};
	const float sat = r.GetReward(cur, s, false);
	s.ball.vel = {0, RLGC::Math::KPHToVel(520), 0};
	CHECK(r.GetReward(cur, s, false) == doctest::Approx(sat).epsilon(1e-4));
	CHECK(sat > full);
}

TEST_CASE("a touch that knocks the ball backwards is not a shot") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// Ball still ends up travelling at the net, but this touch SLOWED it.
	s.ball.pos = {0, 0, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 2500, 0};
	s.ball.vel = {0, 2000, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);
}

TEST_CASE("a shot wide of the post falls off smoothly, so wide shots have a gradient") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;
	s.ball.pos = {0, 0, 93};

	const float halfWidth = RLGC::CommonValues::GOAL_WIDTH_FROM_CENTER;
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};

	s.ball.vel = {0, 2000, 0};
	const float onTarget = r.GetReward(cur, s, false);

	// One half-width outside the post: exp(-1) of the on-target value.
	s.ball.vel = {(halfWidth * 2.f) / 2.56f, 2000, 0};
	CHECK(r.GetReward(cur, s, false) ==
		  doctest::Approx(onTarget * std::exp(-1.f)).epsilon(1e-2));

	// Two half-widths outside: exp(-2). Monotone, never zero, so there is
	// always a gradient pulling a wide shot back in.
	s.ball.vel = {(halfWidth * 3.f) / 2.56f, 2000, 0};
	CHECK(r.GetReward(cur, s, false) ==
		  doctest::Approx(onTarget * std::exp(-2.f)).epsilon(1e-2));
}

TEST_CASE("a shot over the crossbar falls off like a wide one") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// 0.56 s to the plane, arriving one half-width above the crossbar.
	s.ball.pos = {0, 4000, 93};
	const float t = 0.56f;
	const float wantZ = RLGC::CommonValues::GOAL_HEIGHT +
						RLGC::CommonValues::GOAL_WIDTH_FROM_CENTER;
	const float vz = (wantZ - 93.f + 0.5f * 650.f * t * t) / t;
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 2000, vz};

	// Same outgoing ball, but the touch changed nothing about it, so it is not
	// a shot at all.
	CHECK(r.GetReward(cur, s, false) == 0.f);

	sPrev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, 2000, vz};
	const float over = r.GetReward(cur, s, false);
	s.ball.vel = {0, 2000, 0};
	CHECK(over == doctest::Approx(r.GetReward(cur, s, false) * std::exp(-1.f)).epsilon(0.05));
}

TEST_CASE("a ground shot that would fall below the floor still counts") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// Ballistically this arrives 2000 uu underground; a real ball rolls in.
	s.ball.pos = {0, 0, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, 2000, 0};
	CHECK(r.GetReward(cur, s, false) > 0.f);
}

TEST_CASE("a shot at the wrong net pays nothing") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	s.ball.pos = {0, 0, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, -2000, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// The same shot is a real shot for the orange car.
	cur.team = Team::ORANGE;
	CHECK(r.GetReward(cur, s, false) > 0.f);
}

TEST_CASE("a poke too slow to reach the goal is not a shot") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;

	// 5120 uu at 1000 uu/s is 5.12 s, past MAX_TIME.
	s.ball.pos = {0, 0, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, 1000, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);
}

TEST_CASE("a carry pays the shot term once, not once per step") {
	ShotOnTargetReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;
	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	s.ball.pos = {0, 0, 93};
	sPrev.ball.pos = s.ball.pos;
	sPrev.ball.vel = {0, 0, 0};
	s.ball.vel = {0, 2000, 0};

	cur.ballTouchedStep = true;
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(cur, s, false) > 0.f);

	prev.ballTouchedStep = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);

	cur.ballTouchedStep = false;
	CHECK(r.GetReward(cur, s, false) == 0.f);
}


// Blue defends -BACK_WALL_Y, so a ball travelling in -y threatens blue's net.

namespace {

void SetupThreatOnBlue(RLGC::GameState& sPrev, RLGC::GameState& s) {
	s.prev = &sPrev;
	sPrev.ball.pos = {0, 0, 93};
	sPrev.ball.vel = {0, -2000, 0};
	s.ball.pos = sPrev.ball.pos;
}

RLGC::Player Toucher(Team team) {
	static RLGC::Player prev = {};
	prev.ballTouchedStep = false;
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = team;
	cur.ballTouchedStep = true;
	return cur;
}

}  // namespace

TEST_CASE("clearing a ball off your own goal line pays, and the botch charges") {
	SaveReward r;
	RLGC::GameState sPrev = {}, s = {};
	SetupThreatOnBlue(sPrev, s);
	RLGC::Player cur = Toucher(Team::BLUE);

	CHECK(SaveReward::ThreatAtOwnNet(sPrev, Team::BLUE) == doctest::Approx(1.f));

	s.ball.vel = {0, 2000, 0};
	const float cleared = r.GetReward(cur, s, false);
	CHECK(cleared == doctest::Approx(1.f));

	// A harmless ball turned into a shot on your own net.
	RLGC::GameState bPrev = {}, b = {};
	b.prev = &bPrev;
	bPrev.ball.pos = {0, 0, 93};
	bPrev.ball.vel = {0, 2000, 0};
	b.ball.pos = bPrev.ball.pos;
	b.ball.vel = {0, -2000, 0};
	const float botched = r.GetReward(cur, b, false);
	CHECK(botched == doctest::Approx(-1.f));

	CHECK(cleared == doctest::Approx(-botched));
}

TEST_CASE("a deflection that leaves the ball on target pays only the difference") {
	SaveReward r;
	RLGC::GameState sPrev = {}, s = {};
	SetupThreatOnBlue(sPrev, s);
	RLGC::Player cur = Toucher(Team::BLUE);

	s.ball.vel = {600, -2000, 0};
	const float partial = r.GetReward(cur, s, false);

	CHECK(partial > 0.f);
	CHECK(partial < 1.f);

	// Anti-farm: a repeat touch collects only the threat it newly removed.
	RLGC::GameState s2Prev = {}, s2 = {};
	s2.prev = &s2Prev;
	s2Prev.ball.pos = s.ball.pos;
	s2Prev.ball.vel = s.ball.vel;
	s2.ball.pos = s.ball.pos;
	s2.ball.vel = s.ball.vel;
	CHECK(r.GetReward(cur, s2, false) == doctest::Approx(0.f));
}

TEST_CASE("SaveReward is blind to a ball that threatens nobody") {
	SaveReward r;
	RLGC::GameState sPrev = {}, s = {};
	s.prev = &sPrev;
	sPrev.ball.pos = {0, 0, 93};
	sPrev.ball.vel = {2000, 0, 0};
	s.ball.pos = sPrev.ball.pos;
	s.ball.vel = {-2000, 0, 0};

	CHECK(r.GetReward(Toucher(Team::BLUE), s, false) == doctest::Approx(0.f));
}

TEST_CASE("SaveReward pays on the rising edge only") {
	SaveReward r;
	RLGC::GameState sPrev = {}, s = {};
	SetupThreatOnBlue(sPrev, s);
	s.ball.vel = {0, 2000, 0};

	RLGC::Player prev = {};
	prev.ballTouchedStep = true;  // already in contact last step
	RLGC::Player cur = {};
	cur.prev = &prev;
	cur.team = Team::BLUE;
	cur.ballTouchedStep = true;

	CHECK(r.GetReward(cur, s, false) == 0.f);
}

TEST_CASE("SaveReward is team-invariant") {
	SaveReward r;

	RLGC::GameState bluePrev = {}, blue = {};
	SetupThreatOnBlue(bluePrev, blue);
	blue.ball.vel = {0, 2000, 0};

	// Reflected: orange defends +BACK_WALL_Y.
	RLGC::GameState orangePrev = {}, orange = {};
	orange.prev = &orangePrev;
	orangePrev.ball.pos = {0, 0, 93};
	orangePrev.ball.vel = {0, 2000, 0};
	orange.ball.pos = orangePrev.ball.pos;
	orange.ball.vel = {0, -2000, 0};

	CHECK(r.GetReward(Toucher(Team::BLUE), blue, false) ==
	      doctest::Approx(r.GetReward(Toucher(Team::ORANGE), orange, false)));
}

TEST_CASE("the save term's zero-sum scale closes the farm rather than funding it") {
	const RewardBudget b = {};

	// At k < 1 the population nets S(1-k) per event: see docs/comments.md.
	CHECK(b.saveOpponentScale == 1.f);
	CHECK(b.shotOnTargetOpponentScale == 1.f);
}

TEST_CASE("a saved on-target shot is never net-negative for the shooter") {
	const RewardBudget b = {};

	// The exp(-miss/MISS_SCALE) is common to both terms and cancels.
	constexpr float P16_SHOT_STRENGTH = 0.5247f;  // p17cal probe, tail 10
	CHECK(b.save <= b.shotOnTarget * P16_SHOT_STRENGTH);
}
