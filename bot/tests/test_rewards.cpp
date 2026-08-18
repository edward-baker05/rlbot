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

// Direction is the whole point. p11 showed a force-only term cannot stop the
// poke farm: Touch/Hit Force fell 878 -> 551, below StrongTouch's own 555.6
// floor, so the average touch earned nothing from it while TouchEdge doubled.
TEST_CASE("TouchGoalAccel pays for moving the ball toward the right net") {
	TouchGoalAccelReward r;

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

// The min() is the anti-farm device and the reason this term is safe to add to
// a bot that already reaches high balls by driving up the wall.
TEST_CASE("AirTouch pays nothing for a wall shot, however high") {
	AirTouchReward r;
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
TEST_CASE("the goal reward is decisive but not dominant") {
	const RewardBudget b = {};

	// Worth clearly more than any single touch.
	CHECK(b.goal > 3.f * b.touchGoalAccel);

	// But small against a whole episode of shaping. At p11 rates the realized
	// ledger is ~123 touch-units per episode, so one goal is under a tenth.
	constexpr float REALIZED_LEDGER_PER_EPISODE = 123.f;
	CHECK(b.goal < 0.15f * REALIZED_LEDGER_PER_EPISODE);
}

// Budgets are stated per REFERENCE episode (171 steps) but episodes now run
// ~1734 steps, so the raw numbers no longer describe the realized ledger. The
// property that must survive is the one p7approach got wrong: dense approach
// has to dominate contact for a bot still learning to reach the ball.
//
// Checked in realized per-episode terms at p10touch's measured rates, which is
// the only comparison that means anything once REFERENCE_EPISODE_STEPS is
// 10x stale.
TEST_CASE("dense approach still dominates contact in realized terms") {
	const RewardBudget b = {};

	// p10touch, last 60 iterations.
	constexpr float EPISODE_STEPS = 1734.f;
	constexpr float ALIGNMENT = 0.7465f;
	constexpr float EDGE_RATE = 0.021f;
	constexpr float STRONG_VALUE = 0.104f;

	const float densePerEp =
		(b.speedToBall + b.faceBall) / REFERENCE_EPISODE_STEPS * ALIGNMENT * EPISODE_STEPS;
	const float touches = EDGE_RATE * EPISODE_STEPS;
	const float touchPerEp = touches * (b.touchEdge + b.touchGoalAccel * STRONG_VALUE);

	CHECK(densePerEp > touchPerEp);

	// But not by so much that contact is noise: p7approach's stack paid 1.83
	// touch-units for a whole episode of perfect approach and never learned to
	// approach at all, while p9rel's inverted the ratio and learned to dribble.
	CHECK(densePerEp < 10.f * touchPerEp);
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

TEST_CASE("the spec list is the nine designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs) {
		names.push_back(s.name);
		CHECK(s.weight > 0.f);
	}

	const std::vector<std::string> expected = {"TouchGoalAccel", "Goal", "TouchEdge",
	                                           "SpeedToBall", "FaceBall", "SaveBoost",
	                                           "PickupBoost", "AirTouch", "Air"};
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

TEST_CASE("PickupBoost is the increment of SaveBoost's potential") {
	RLGC::PickupBoostReward r;
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

	// Topping up when low pays far more than the same 12 boost when nearly
	// full -- which is what makes small pads worth taking.
	CHECK(gain(0.f, 12.f) > 3.f * gain(88.f, 100.f));

	// Spending boost is not punished; only gaining is paid.
	CHECK(gain(100.f, 40.f) == 0.f);
	CHECK(gain(50.f, 50.f) == 0.f);
}

