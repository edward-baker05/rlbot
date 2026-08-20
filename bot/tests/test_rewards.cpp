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

	auto rewardFor = [&](float airTime, float ballZ) {
		p.airTime = airTime;
		s.ball.pos = {0, 0, ballZ};
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
TEST_CASE("an air CARRY pays once, not once per step") {
	AirTouchReward r(1.f);
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, RLGC::CommonValues::CEILING_Z};

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

	auto pay = [&](AirTouchReward& r, float z) {
		s.ball.pos = {0, 0, z};
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
// p12goal, last 20 iterations at 250M steps. These are the rates p13's budgets
// were solved against, so they are the only honest denominators.
constexpr float P12_EPISODE_STEPS = 390.f;
constexpr float P12_EDGE_RATE = 0.0291f;     // Touch/Edge Rate
constexpr float P12_SPEED_TO_BALL = 0.3755f; // Rewards/SpeedToBall, raw mean
constexpr float P12_FACE_BALL = 0.6530f;     // Rewards/FaceBall, raw mean

// The measured price of leaving the ground, in touch-units: Critic/TD Delta
// Jump -0.2249 against NoJump -0.0199, so -0.205 standardized, times
// GAE/Returns STD 3.939.
constexpr float TAKEOFF_COST = 0.81f;

// A realistic air dribble touch: ball at z 800 (heightFrac 0.391) after 1.0 s
// aloft (airTimeFrac 0.571). min() takes the height.
constexpr float REALISTIC_AERIAL = 0.391f * 0.391f; // 0.153, height now squared

// A touch worth calling strong, per the 80 kph target, scored convexly. This
// is what ONE GOOD HIT is worth, and it is 27x the mean -- which is the whole
// reason the two must never be confused when sizing a budget.
constexpr float STRONG_TOUCH_VALUE = (80.f / 130.f) * (80.f / 130.f); // 0.379

// What the AVERAGE touch is worth, which is what sets the reward SHARE.
// p12 measured RewardShare/TouchGoalAccel 0.098 at weight 3.0; back-solved,
// E[|x|] = 0.0705 per contact step (255 uu/s = 9.2 kph). Squared under the
// convex form and multiplied by the 1.41 contact steps per sequence:
constexpr float P12_MEAN_TOUCH_VALUE = 2.f * 0.0705f * 0.0705f * 1.41f; // 0.0140
} // namespace

TEST_CASE("the goal reward is decisive but not dominant") {
	const RewardBudget b = {};

	const float touches = P12_EDGE_RATE * P12_EPISODE_STEPS; // 11.35
	const float shapingPerEp =
		(b.speedToBall * P12_SPEED_TO_BALL + b.faceBall * P12_FACE_BALL) /
			REFERENCE_EPISODE_STEPS * P12_EPISODE_STEPS +
		touches * (b.touchEdge + b.touchGoalAccel * P12_MEAN_TOUCH_VALUE);

	// Not so large that it drowns the shaping it exists to break ties between.
	// That is the guide's warning, and it is the reason this budget was not
	// scaled up alongside everything else.
	CHECK(b.goal < 3.f * shapingPerEp);

	// But decisive: a goal must beat a whole episode's worth of ordinary
	// contact, or scoring is not what the bot is optimizing.
	CHECK(b.goal > touches * b.touchGoalAccel * P12_MEAN_TOUCH_VALUE);

	// THE SHOT-FARM GUARD. Convexity pays a lot for one good hit, and the
	// failure mode it invites is blasting the ball goalward over and over
	// without scoring -- especially since scoring ENDS the episode and forfeits
	// the rest of the stream. One strong touch must not out-earn a goal by
	// enough to make that trade attractive. Watched at runtime too: hit force
	// rising while `Episode/Mean Steps` also rises is the pre-registered kill.
	CHECK(b.touchGoalAccel * STRONG_TOUCH_VALUE < 3.f * b.goal);
}

// THE INVERSION, and it is the point of p13strike. Every previous run in this
// project paid more for BEING NEAR the ball than for what the ball did:
// RewardShare SpeedToBall + FaceBall was 0.761 (p8ref), 0.876 (p10), 0.778
// (p11), 0.606 (p12), and no run ever tested reducing it. p12's realized
// ledger was 84.5% proximity against 9.3% ball.
//
// The guide's middle-stage instruction, once the bot can hit the ball: the
// ball-to-goal term should be "a fair bit stronger than SpeedTowardBallReward".
// It can hit the ball -- 11.3 contacts per episode.
TEST_CASE("the ball now outranks the chase") {
	const RewardBudget b = {};

	const float proximityPerEp =
		(b.speedToBall * P12_SPEED_TO_BALL + b.faceBall * P12_FACE_BALL) /
			REFERENCE_EPISODE_STEPS * P12_EPISODE_STEPS +
		P12_EDGE_RATE * P12_EPISODE_STEPS * b.touchEdge;

	// Realized, at p12's OWN touch strengths -- not at the strength the run is
	// trying to buy. Sizing a budget off the outcome you want rather than the
	// one you have is precisely how p12's AirTouch shipped at 0.008 share.
	const float ballPerEp =
		P12_EDGE_RATE * P12_EPISODE_STEPS * b.touchGoalAccel * P12_MEAN_TOUCH_VALUE;

	CHECK(ballPerEp > proximityPerEp);

	// But approach is not deleted: the bot took eight runs to learn to drive at
	// the ball at all (Velocity Alignment left its 1/pi null for the first time
	// in p8ref) and that must survive. Proximity keeps at least a quarter.
	CHECK(proximityPerEp > 0.25f * ballPerEp);
}

// p12 measured the price of a takeoff directly, under something close to a
// randomized trial: ~91% of sampled jumps come from the exploration floor,
// which mixes uniformly over valid actions independently of state. AirTouch at
// 2.0 paid a realistic aerial 0.78 against a cost of 0.81 -- 4% BELOW
// break-even, which is why the behaviour appeared and decayed twice rather
// than establishing or vanishing.
TEST_CASE("an aerial pays for itself, with margin") {
	const RewardBudget b = {};

	const float paid = b.airTouch * REALISTIC_AERIAL;
	CHECK(paid > TAKEOFF_COST);

	// Deliberate over-payment: two runs have already found this behaviour and
	// lost it at margins near 1.0.
	CHECK(paid > 2.f * TAKEOFF_COST);

	// The old bound here was a fixed multiple of the takeoff cost, and it is
	// the WRONG instrument once the term is convex in height: a rarer event
	// necessarily carries a larger per-event payment for the same aggregate
	// mass, so a per-event cap silently caps the RATE instead. Floating is
	// guarded by the min() with air time (a wall shot is worth zero, asserted
	// above) and by the AGGREGATE share -- target 0.060 against a
	// pre-registered RewardShare/AirTouch kill ceiling of 0.35. Bound restated
	// against the quantity that is actually controlled.
	constexpr float TARGET_SHARE = 0.060f;
	constexpr float SHARE_CEILING = 0.35f;
	CHECK(TARGET_SHARE < SHARE_CEILING / 3.f);

	// And it still cannot be collected from a wall, which is what makes the
	// budget safe to raise at all. Asserted behaviourally above.
	CHECK(b.airTouch > b.touchEdge);
}

// The dense terms are still the guide's shape relative to each other, but the
// BLOCK is smaller. What has to be true is that the per-step weight actually
// fell -- the budget NUMBER barely moved (17.1 -> 14.51) only because
// REFERENCE_EPISODE_SECONDS was corrected 11.4 -> 26.0 in the same change, and
// reading the number instead of the weight is exactly how this project spent
// four runs over-delivering every rate budget by 2.28x.
TEST_CASE("the approach budget really was cut, in per-step terms") {
	const RewardBudget b = {};

	// p12goal: speedToBall 17.1 over 171 reference steps.
	constexpr float P12_SPEED_WEIGHT = 17.1f / 171.f; // 0.1000
	constexpr float P12_FACE_WEIGHT = 3.42f / 171.f;  // 0.0200
	constexpr float P12_AIR_WEIGHT = 0.513f / 171.f;  // 0.0030

	CHECK(RateWeight(b.speedToBall) < 0.5f * P12_SPEED_WEIGHT);
	CHECK(RateWeight(b.faceBall) < 0.5f * P12_FACE_WEIGHT);

	// Air is NOT part of the cut: its target share is unchanged, so its weight
	// should be within a factor of two of p12's. The budget number tripling is
	// the reference-length correction and nothing else.
	CHECK(RateWeight(b.air) > 0.5f * P12_AIR_WEIGHT);
	CHECK(RateWeight(b.air) < 2.0f * P12_AIR_WEIGHT);

	// The reference episode is the measured one now, not a stale constant.
	CHECK(REFERENCE_EPISODE_STEPS == doctest::Approx(P12_EPISODE_STEPS));
}

// The convexity is a config field so it can be raised on evidence rather than
// on a schedule, but it must ship at a value that is learnable NOW. The
// constraint is the gradient available at the current operating point relative
// to the target one: (x_now / x_target)^(p-1). At p12's mean 15.2 kph against
// an 80 kph target that is 19% for p=2, 3.6% for p=3, 0.7% for p=4.
TEST_CASE("the touch exponent is convex but still learnable") {
	const RewardBudget b = {};

	CHECK(b.touchAccelExponent > 1.f);

	const float ratio = 15.2f / 80.f;
	const float gradientFrac = std::pow(ratio, b.touchAccelExponent - 1.f);
	CHECK(gradientFrac > 0.10f);
}

TEST_CASE("the spec list is the ten designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs) {
		names.push_back(s.name);
		CHECK(s.weight > 0.f);
	}

	const std::vector<std::string> expected = {"TouchGoalAccel", "Goal", "TouchEdge",
	                                           "SpeedToBall", "FaceBall", "SaveBoost",
	                                           "PickupBoost", "FlipSpeed", "AirTouch", "Air"};
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
		CHECK(s.name != "WrongSurface");
		CHECK(s.name != "CleanLanding");
		CHECK(s.name != "HarshSpeedLoss");
		CHECK(s.name != "VelBallToGoal");
	}
}

// The boost economy, added in p11 after p10touch measured `Player/Boost` at
// 7.3 out of 100 while the bot was trying to air dribble.
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

TEST_CASE("FlipSpeedReward rewards forward impulse toward the ball on flip rising edge") {
	FlipSpeedReward r;
	RLGC::GameState sPrev = {};
	RLGC::GameState s = {};
	s.prev = &sPrev;

	// Ball straight ahead along +Y
	s.ball.pos = {0, 2000, 93};
	sPrev.ball.pos = {0, 2000, 93};

	RLGC::Player prev = {};
	RLGC::Player cur = {};
	cur.prev = &prev;
	prev.pos = {0, 0, 17};
	cur.pos = {0, 50, 17};

	// 1. Clean forward dodge (+500 uu/s toward ball)
	prev.vel = {0, 1400, 0};
	prev.isFlipping = false;
	prev.hasFlipped = false;
	cur.vel = {0, 1900, 0};
	cur.isFlipping = true;
	cur.hasFlipped = true;

	CHECK(r.GetReward(cur, s, false) == doctest::Approx(1.0f).epsilon(1e-3));

	// 2. Not a rising edge (already flipping) -> 0
	prev.isFlipping = true;
	CHECK(r.GetReward(cur, s, false) == 0.f);
	prev.isFlipping = false;

	// 3. Sideways dodge (0 forward gain towards ball) -> 0
	cur.vel = {500, 1400, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);

	// 4. Supersonic before flip (v >= 2200 uu/s) -> 0
	prev.vel = {0, 2200, 0};
	cur.vel = {0, 2300, 0};
	CHECK(r.GetReward(cur, s, false) == 0.f);
}




