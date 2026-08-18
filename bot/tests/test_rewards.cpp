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
	// speed, unlike the SpeedSquared term it replaces -- there is no free
	// coasting floor to discount here, because standing still and driving away
	// both pay zero already.
	p.vel = {0, V / 2.f, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.5f).epsilon(1e-4));

	// Perpendicular: closing speed is zero.
	p.vel = {V, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// THE RECTIFICATION. Driving away pays nothing -- it is not punished.
	// Upstream's VelocityPlayerToBallReward would return -1 here. Plenty of
	// correct play moves away from the ball, and the signed form also lets a
	// circling bot generate large +/- values that cancel to nothing, which is
	// what p1air's RewardShare 0.482 at ~zero net was.
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

TEST_CASE("FaceBallRectifiedReward pays for pointing at the ball only") {
	FaceBallRectifiedReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	p.pos = {0, 0, 93};

	// Nose at the ball.
	p.rotMat.forward = {0, 1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Perpendicular pays nothing.
	p.rotMat.forward = {1, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Nose directly AWAY pays nothing AND costs nothing. Both halves matter:
	// the p6budget stack did not pay for facing away, it punished it at half
	// rate, and shadow defence and retreating for a bounce both need the nose
	// off the ball.
	p.rotMat.forward = {0, -1, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Sitting on the ball: no direction to face, and no divide by zero.
	p.pos = s.ball.pos;
	p.rotMat.forward = {0, 1, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("the rectified term is the asymmetric form at w- = 0") {
	// The decomposition the p6budget design was built on, kept as an
	// executable assertion because it is what justifies collapsing two specs
	// into one:
	//
	//   w+ * max(0,c) + w- * min(0,c) == ws * c + wa * |c|
	//   with ws = (w+ + w-)/2 and wa = (w+ - w-)/2
	//
	// At w- = 0 that gives ws == wa == w+/2: the signed lobe and the |c| lobe
	// carry EQUAL weight. Which is the point worth remembering -- "stop paying
	// for facing away" means RAISING the |c| lobe to match the signed one, not
	// deleting it. Shipping one clamped term is the same reward with one
	// budget instead of two.
	RLGC::FaceBallReward signedTerm;
	FaceBallRectifiedReward rectified;

	const float wPlus = 0.05f;       // the FaceBall budget
	const float ws = wPlus / 2.f;    // signed lobe
	const float wa = wPlus / 2.f;    // |c| lobe

	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	p.pos = {0, 0, 93};

	// Three orientations, spanning both lobes' behaviour.
	const Vec dirs[] = {{0, 1, 0}, {1, 0, 0}, {0, -1, 0}};
	for (const Vec& dir : dirs) {
		p.rotMat.forward = dir;
		const float c = signedTerm.GetReward(p, s, false);
		const float decomposed = ws * c + wa * std::fabs(c);
		CHECK(decomposed
		      == doctest::Approx(wPlus * rectified.GetReward(p, s, false)).epsilon(1e-4));
	}
}

TEST_CASE("budget conversion is the only route to a per-step weight") {
	// A rate budget is what holding the behaviour perfectly for one reference
	// episode earns. 171 steps = 11.4 s at 15 Hz, which is p6budget's MEASURED
	// Episode/Mean Steps -- the previous 150 was a working figure and every
	// rate term was over-delivering by 14%.
	CHECK(REFERENCE_EPISODE_STEPS == doctest::Approx(171.f));
	CHECK(RateWeight(0.50f) == doctest::Approx(0.50f / 171.f));
	CHECK(RateWeight(0.50f) * REFERENCE_EPISODE_STEPS == doctest::Approx(0.50f));

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
	CHECK(b.speedToBall < RewardBudget::GOAL);
	CHECK(b.faceBall < RewardBudget::GOAL);

	// And all of them together must not outweigh a goal either.
	CHECK(b.speedToBall + b.faceBall < RewardBudget::GOAL);
}

TEST_CASE("approach dominates facing by an order of magnitude") {
	// p6budget measured its facing terms taking 62% of net earnings and 66% of
	// the entire run's ledger improvement while its velocity-to-ball alignment
	// never moved. Facing is a tiebreaker against driving backwards at the
	// ball; if these budgets ever drift back together, that is the failure
	// returning.
	const RewardBudget b = {};
	CHECK(b.speedToBall >= 10.f * b.faceBall);

	// And a touch has to be worth more than a long stretch of approaching.
	// One touch against a full reference episode of PERFECT closing speed --
	// which no bot achieves -- keeps finishing an approach worth more than
	// repeating one, which is the counterweight to SpeedToBall being farmable
	// around a chase-hit-chase cycle.
	CHECK(b.touch > 0.5f * b.speedToBall);
}

TEST_CASE("the spec list is the five designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs)
		names.push_back(s.name);

	const std::vector<std::string> expected = {
		"Goal", "Touch", "SpeedToBall", "FaceBall", "WrongSurface",
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
