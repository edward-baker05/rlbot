#include "doctest/doctest.h"
#include "TestCommon.h"

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
	REQUIRE(specs.size() >= 4);
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
TEST_CASE("WrongSurfaceReward charges only non-wheel contact") {
	WrongSurfaceReward r;
	RLGC::Player p = {};
	RLGC::GameState s = {};

	// Driving normally: wheels down, chassis clear.
	p.isOnGround = true;
	p.worldContact.hasContact = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Airborne with nothing touching: free. Leaving the ground is not the
	// offence; landing wrong is.
	p.isOnGround = false;
	p.worldContact.hasContact = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// THE GATE. Chassis scraping while the wheels are still doing their job --
	// a wall-curve transition, a bottomed-out suspension -- is not a loss of
	// control and must not be charged.
	p.isOnGround = true;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// On the roof, the side, the nose: all the same, all fully charged.
	p.isOnGround = false;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == -1.f);
}

TEST_CASE("CleanLandingReward pays the squared impact it absorbed") {
	CleanLandingReward r(LANDING_REF_IMPACT);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// No prev: the episode just reset and the velocity discontinuity is a
	// state-setter teleport, not a landing.
	RLGC::Player orphan = {};
	orphan.isOnGround = true;
	CHECK(r.GetReward(orphan, s, false) == 0.f);

	// Still airborne: no edge.
	p.isOnGround = false;
	prev.isOnGround = false;
	prev.vel = {0, 0, -1200};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Already grounded last step: no edge either.
	p.isOnGround = true;
	prev.isOnGround = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A real landing edge, but the chassis is touching -- this is a crash, and
	// WrongSurface is already charging for it. Paying here too would let the
	// bot buy its way out of that penalty by crashing fast.
	p.isOnGround = true;
	prev.isOnGround = false;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Clean landing from a held single jump (~453 uu/s). The SQUARE is the
	// anti-farm mechanism: linear would make bunny-hopping competitive with a
	// real aerial on a per-second basis.
	p.worldContact.hasContact = false;
	prev.vel = {0, 0, -450};
	CHECK(r.GetReward(p, s, false)
	      == doctest::Approx((450.f / 1100.f) * (450.f / 1100.f)).epsilon(1e-4));

	// Saturates: falling faster than the reference pays 1, not more.
	prev.vel = {0, 0, -1100};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));
	prev.vel = {0, 0, -2300};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));

	// Rising into a landing is not a fall. Only downward speed counts, so a
	// wall landing scores zero -- a known and accepted limitation.
	prev.vel = {0, 0, 900};
	CHECK(r.GetReward(p, s, false) == 0.f);
	prev.vel = {2300, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("a bunny hop is worth far less per second than an aerial") {
	// The farm argument from the design doc, as an executable check. Hop:
	// ~450 uu/s off a held jump, ~1.4 s round trip. Aerial from ~1000 uu:
	// sqrt(2*650*1000) = 1140 uu/s, ~3.5 s round trip.
	CleanLandingReward r(LANDING_REF_IMPACT);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;
	p.isOnGround = true;
	p.worldContact.hasContact = false;
	prev.isOnGround = false;

	prev.vel = {0, 0, -450};
	const float hopRate = r.GetReward(p, s, false) / 1.4f;

	prev.vel = {0, 0, -1140};
	const float aerialRate = r.GetReward(p, s, false) / 3.5f;

	// Under LINEAR scaling these are 0.24/s vs 0.27/s and hopping is a viable
	// farm. Squared, the aerial must dominate by a wide margin.
	CHECK(aerialRate > hopRate * 2.f);
}

TEST_CASE("TouchEdgeReward pays once per contact, not once per step") {
	TouchEdgeReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// Not touching.
	p.ballTouchedStep = false;
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// First contact of a sequence: paid.
	p.ballTouchedStep = true;
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 1.f);

	// THE DRIBBLE GUARD. Still touching from last step pays nothing. A
	// per-step touch reward IS a dribble reward -- carrying the ball on the
	// nose would collect it ~180x an episode, which is the flick-bot local
	// optimum arriving through the back door.
	p.ballTouchedStep = true;
	prev.ballTouchedStep = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A touch on the first step after a reset is a genuine new contact.
	RLGC::Player fresh = {};
	fresh.ballTouchedStep = true;
	CHECK(r.GetReward(fresh, s, false) == 1.f);
}

TEST_CASE("FaceBallAxisReward is the unsigned lobe of facing") {
	FaceBallAxisReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	// Same z as the ball, so toBall is exactly horizontal and lines up exactly
	// with a horizontal forward vector -- otherwise "nose at the ball" is only
	// approximately aligned and the tight epsilon below is spurious.
	p.pos = {0, 0, 93};

	// Nose at the ball.
	p.rotMat.forward = {0, 1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Nose directly AWAY pays exactly the same. This is the term's defining
	// property and the reason it is split out rather than hidden inside
	// rectified FaceBall weights.
	p.rotMat.forward = {0, -1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Perpendicular pays nothing.
	p.rotMat.forward = {1, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Sitting on the ball: no direction to face, and no divide by zero.
	p.pos = s.ball.pos;
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("FaceBall plus FaceBallAxis reconstruct the asymmetric form") {
	// The decomposition from the design doc, as an executable assertion:
	//
	//   w+ * max(0,c) + w- * min(0,c) == ws * c + wa * |c|
	//   with ws = (w+ + w-)/2 and wa = (w+ - w-)/2
	//
	// Facing away is sometimes correct, so the negative side is weaker. That
	// asymmetry is REAL but it is not free: the |c| half pays identically for
	// nose-at-ball and nose-directly-away, and it is an annuity. Shipping it as
	// two specs is what makes it visible in RewardShare.
	RLGC::FaceBallReward signedTerm;
	FaceBallAxisReward axisTerm;

	const float ws = 0.20f;          // FaceBall budget
	const float wa = 0.20f / 3.f;    // FaceBallAxis budget: exactly ws/3
	const float wPlus = ws + wa;
	const float wMinus = ws - wa;

	// The 2:1 ratio the design fixes.
	CHECK(wPlus / wMinus == doctest::Approx(2.f).epsilon(1e-5));

	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	// Same z as the ball; see the note in the FaceBallAxisReward test above.
	p.pos = {0, 0, 93};

	// Facing the ball: the pair must equal the positive rectified weight.
	p.rotMat.forward = {0, 1, 0};
	float combined = ws * signedTerm.GetReward(p, s, false)
	               + wa * axisTerm.GetReward(p, s, false);
	CHECK(combined == doctest::Approx(wPlus * 1.f).epsilon(1e-4));

	// Facing away: the pair must equal the WEAKER negative rectified weight.
	p.rotMat.forward = {0, -1, 0};
	combined = ws * signedTerm.GetReward(p, s, false)
	         + wa * axisTerm.GetReward(p, s, false);
	CHECK(combined == doctest::Approx(wMinus * -1.f).epsilon(1e-4));
	// And it must still be negative -- weaker, not inverted.
	CHECK(combined < 0.f);
}

TEST_CASE("SpeedSquaredReward discounts the free coasting floor") {
	SpeedSquaredReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.vel = {0, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f));

	// Throttle-only top speed: holdable forever with no boost and no skill.
	// Linear scaling would hand this 0.613 of the term's maximum for free;
	// squaring cuts it to 0.375.
	p.vel = {THROTTLE_TOP_SPEED, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.375f).epsilon(0.01));

	// Supersonic.
	p.vel = {RLGC::CommonValues::CAR_MAX_SPEED, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Boost/flip speed must beat coasting by 2.67x, not 1.63x. That ratio is
	// the reason for the square.
	p.vel = {THROTTLE_TOP_SPEED, 0, 0};
	const float coast = r.GetReward(p, s, false);
	p.vel = {RLGC::CommonValues::CAR_MAX_SPEED, 0, 0};
	const float fast = r.GetReward(p, s, false);
	CHECK(fast / coast == doctest::Approx(2.67f).epsilon(0.02));

	// Clamped, not extrapolated -- a bump or a ramp can exceed CAR_MAX_SPEED.
	p.vel = {5000, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));
}

TEST_CASE("HarshSpeedLossReward charges collisions, not braking or hits") {
	HarshSpeedLossReward r(HARSH_LOSS_THRESHOLD);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// No prev: a reset, not a crash.
	RLGC::Player orphan = {};
	orphan.vel = {0, 0, 0};
	CHECK(r.GetReward(orphan, s, false) == 0.f);

	// Gaining speed.
	prev.vel = {500, 0, 0};
	p.vel = {900, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Hard braking: ~3500 uu/s^2 over a 1/15 s step is 233 uu/s, comfortably
	// under the 400 threshold. Deliberate deceleration must be free.
	prev.vel = {1400, 0, 0};
	p.vel = {1167, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Exactly at the threshold is still free.
	prev.vel = {1400, 0, 0};
	p.vel = {1000, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A wall. Charged, and negative.
	prev.vel = {1400, 0, 0};
	p.vel = {0, 0, 0};
	const float crash = r.GetReward(p, s, false);
	CHECK(crash < 0.f);
	CHECK(crash == doctest::Approx(-(1400.f - 400.f) / (2300.f - 400.f)).epsilon(1e-4));

	// THE HIT EXEMPTION. A hard strike costs speed, and that is a good
	// outcome. Charging for it would penalise striking the ball -- the exact
	// mistake p4pbrs made from the other direction.
	p.ballTouchedStep = true;
	CHECK(r.GetReward(p, s, false) == 0.f);
	p.ballTouchedStep = false;

	// Saturates at a full-speed stop.
	prev.vel = {2300, 0, 0};
	p.vel = {0, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(-1.f).epsilon(0.01));
}

TEST_CASE("budget conversion is the only route to a per-step weight") {
	// A rate budget is what holding the behaviour perfectly for one reference
	// episode earns. 150 steps = 10 s at 15 Hz.
	CHECK(REFERENCE_EPISODE_STEPS == doctest::Approx(150.f));
	CHECK(RateWeight(0.30f) == doctest::Approx(0.30f / 150.f));
	CHECK(RateWeight(0.30f) * REFERENCE_EPISODE_STEPS == doctest::Approx(0.30f));

	// A per-second budget is the cost of one second of the condition.
	CHECK(PerSecondWeight(0.10f) == doctest::Approx(0.10f / 15.f));
	CHECK(PerSecondWeight(0.10f) * STEPS_PER_SECOND == doctest::Approx(0.10f));
}

TEST_CASE("no shaping term can outearn a goal by accident") {
	// The p1air failure, as a regression test. `grounded = 0.05` integrated to
	// 9.0 goal-units per episode -- nine goals per episode for holding still on
	// the wheels -- and nobody noticed because nobody wrote down the integral.
	//
	// Every RATE term's whole-episode earnings must stay well under one goal.
	const RewardBudget b = {};
	CHECK(b.speed < RewardBudget::GOAL);
	CHECK(b.faceBall < RewardBudget::GOAL);
	CHECK(b.faceBallAxis < RewardBudget::GOAL);

	// And all of them together must not outweigh a goal either.
	CHECK(b.speed + b.faceBall + b.faceBallAxis < RewardBudget::GOAL);
}

TEST_CASE("FaceBallAxis is exactly one third of FaceBall") {
	// This ratio IS the 2:1 asymmetry: w+ = ws + wa, w- = ws - wa, and
	// wa = ws/3 gives w+/w- = 2 exactly. If the two budgets drift apart the
	// asymmetry silently becomes something else.
	const RewardBudget b = {};
	CHECK(b.faceBallAxis == doctest::Approx(b.faceBall / 3.f).epsilon(1e-5));

	const float wPlus = b.faceBall + b.faceBallAxis;
	const float wMinus = b.faceBall - b.faceBallAxis;
	CHECK(wPlus / wMinus == doctest::Approx(2.f).epsilon(1e-4));
}

TEST_CASE("the spec list is the eight designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs)
		names.push_back(s.name);

	const std::vector<std::string> expected = {
		"Goal", "Touch", "CleanLanding", "WrongSurface",
		"HarshSpeedLoss", "Speed", "FaceBall", "FaceBallAxis",
	};
	CHECK(names == expected);

	// THE SIGN CONVENTION. Penalty classes return negative values, so every
	// weight must be positive -- a negative weight on a negative class value
	// double-negates a penalty into a reward, and nothing else in the stack
	// would reveal it.
	for (auto& s : specs)
		CHECK(s.weight > 0.f);

	// Goal is the unit.
	CHECK(specs[0].weight == doctest::Approx(1.f));
}

TEST_CASE("no zero-weight placeholder specs remain") {
	// The old stack kept zero-weight specs so RewardShare indices stayed
	// aligned across reward phases. There are no phases now, so a zero-weight
	// spec would just be a term that silently does nothing.
	for (auto& s : GeneralRewardSpecs(TrainConfig{}))
		CHECK(s.weight != 0.f);
}
