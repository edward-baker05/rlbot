#include "doctest/doctest.h"
#include "TestCommon.h"

#include <Config.h>
#include <env/Rewards.h>

#include <RLGymCPP/CommonValues.h>

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

TEST_CASE("TouchHeightReward pays zero for no touch and for ground touches") {
	TouchHeightReward r(1500.f);
	RLGC::Player p = {};
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 1000};

	p.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	p.ballTouchedStep = true;
	s.ball.pos = {0, 0, RLGC::CommonValues::BALL_RADIUS};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("TouchHeightReward scales with height and saturates at 1") {
	TouchHeightReward r(1500.f);
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	RLGC::GameState s = {};

	s.ball.pos = {0, 0, RLGC::CommonValues::BALL_RADIUS + 750.f};
	const float mid = r.GetReward(p, s, false);
	CHECK(mid == doctest::Approx(0.5f));

	s.ball.pos = {0, 0, 3000.f};
	CHECK(r.GetReward(p, s, false) == 1.f);
}
