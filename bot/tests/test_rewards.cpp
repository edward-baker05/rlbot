#include "doctest/doctest.h"

#include <Config.h>
#include <env/Rewards.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

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

// The other half of the fix, and the guide's actual middle-stage prescription:
// "a slight push that barely changes the velocity of the ball will give almost
// no reward, but a strong shot will give lots of reward."
TEST_CASE("StrongTouch pays for hit force and pays a dribble nothing") {
	RLGC::StrongTouchReward r;

	RLGC::GameState prev = {};
	RLGC::GameState s = {};
	s.prev = &prev;

	RLGC::Player p = {};
	p.ballTouchedStep = true;

	auto rewardFor = [&](float deltaVel) {
		prev.ball.vel = {0, 0, 0};
		s.ball.vel = {deltaVel, 0, 0};
		return r.GetReward(p, s, false);
	};

	// The 20 kph floor is 555.6 uu/s -- 1 kph is 27.78 uu/s, not 9.17.
	CHECK(RLGC::Math::KPHToVel(20) == doctest::Approx(555.6f).epsilon(0.01));
	CHECK(RLGC::Math::KPHToVel(130) == doctest::Approx(3611.f).epsilon(0.01));

	// A dribble carry nudges the ball by tens of uu/s and is nowhere near the
	// floor. It must pay EXACTLY zero, or the farm survives the rising edge.
	CHECK(rewardFor(5.f) == 0.f);
	CHECK(rewardFor(50.f) == 0.f);
	CHECK(rewardFor(200.f) == 0.f);
	CHECK(rewardFor(555.f) == 0.f);

	// A real strike scales linearly and saturates at 3611 uu/s of delta-v.
	// p1pay measured typical hit force around 1400 uu/s, which lands at ~0.39
	// -- so realized values are well under 1.0 and `Touch/Strong Value` is what
	// turns the provisional budget into a measured one.
	CHECK(rewardFor(600.f) > 0.f);
	CHECK(rewardFor(1400.f) == doctest::Approx(1400.f / 3611.f).epsilon(0.02));
	CHECK(rewardFor(3611.f) == doctest::Approx(1.f).epsilon(0.02));
	CHECK(rewardFor(6000.f) == doctest::Approx(1.f)); // clamped

	// No touch, no reward, however fast the ball is moving.
	p.ballTouchedStep = false;
	CHECK(rewardFor(3611.f) == 0.f);
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

	// Event budgets are per occurrence, so they are the weight unchanged.
	CHECK(weightOf("StrongTouch") == doctest::Approx(cfg.rewards.strongTouch));
	CHECK(weightOf("TouchEdge") == doctest::Approx(cfg.rewards.touchEdge));
}

// The unit is one FULL-POWER ball touch, 1.0 by definition. Anything else and
// the ledger stops reading in touches, which is the whole reason the
// denominator moved off goals: a goal arrives 0.116 times per episode and
// cannot be checked against telemetry.
TEST_CASE("a full-power touch is the unit") {
	CHECK(TrainConfig{}.rewards.strongTouch == doctest::Approx(1.f));

	// Arriving is worth strictly less than connecting.
	CHECK(TrainConfig{}.rewards.touchEdge < TrainConfig{}.rewards.strongTouch);
}

// The single biggest numerical departure p7approach made from the reference,
// and it went unnoticed because the two stacks were denominated in different
// currencies. Restated in touch-units, p7approach paid 1.83 for a whole
// episode of PERFECT approach where the guide pays 20.5.
TEST_CASE("dense approach dominates a single touch by an order of magnitude") {
	const RewardBudget b = {};
	const float dense = b.speedToBall + b.faceBall;

	CHECK(dense > 10.f * b.strongTouch);
	CHECK(dense == doctest::Approx(20.52f).epsilon(1e-3));
}

// The guide's proportions: touch 50, speed 5, face 1, air 0.15. Divided
// through by touch and expressed as episode integrals, the ratios between the
// dense terms must survive.
TEST_CASE("budgets keep the guide's proportions") {
	const RewardBudget b = {};

	// speed:face is 5:1.
	CHECK(b.speedToBall / b.faceBall == doctest::Approx(5.f).epsilon(1e-3));

	// face:air is 1:0.15.
	CHECK(b.faceBall / b.air == doctest::Approx(1.f / 0.15f).epsilon(1e-3));

	// Air is a nudge, not a policy: under 3% of the dense budget. Ours is
	// already 93% airborne, so if this ever grows it stops being a port.
	CHECK(b.air < 0.03f * (b.speedToBall + b.faceBall));
}

TEST_CASE("the spec list is the five designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs) {
		names.push_back(s.name);
		CHECK(s.weight > 0.f);
	}

	const std::vector<std::string> expected = {"StrongTouch", "TouchEdge", "SpeedToBall",
	                                           "FaceBall", "Air"};
	CHECK(names == expected);
}

// Every one of these was in the stack at some point and each is a decision to
// leave out, not an oversight. Goal is noise at a touch ratio of 0.001;
// WrongSurface held 30% of p7approach's reward mass as a tuning penalty; the
// rest are tuning rewards the guide's troubleshooting section says to remove.
TEST_CASE("no goal, boost, ball-to-goal or tuning terms are in the stack") {
	auto specs = GeneralRewardSpecs(TrainConfig{});
	for (auto& s : specs) {
		CHECK(s.name != "Goal");
		CHECK(s.name != "WrongSurface");
		CHECK(s.name != "CleanLanding");
		CHECK(s.name != "HarshSpeedLoss");
		CHECK(s.name != "VelBallToGoal");
		CHECK(s.name != "PickupBoost");
		CHECK(s.name != "SaveBoost");
	}
}
