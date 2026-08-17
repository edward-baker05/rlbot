#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/PlayPhase.h>

#include <string>

using namespace Hive;

static RLGC::Player MakePlayer(Vec pos, bool onGround, Team team = Team::BLUE) {
	RLGC::Player p = {};
	p.pos = pos;
	p.isOnGround = onGround;
	p.team = team;
	return p;
}

static RLGC::GameState MakeState(Vec ballPos) {
	RLGC::GameState s = {};
	s.ball.pos = ballPos;
	return s;
}

TEST_CASE("airborne car near a high ball classifies as AirDribble") {
	auto p = MakePlayer({0, 0, 600}, false);
	auto s = MakeState({0, 100, 700});
	CHECK(ClassifyPhase(p, s) == PlayPhase::AirDribble);
}

TEST_CASE("airborne car far from a high ball classifies as Aerial") {
	auto p = MakePlayer({0, 0, 600}, false);
	auto s = MakeState({2000, 2000, 900});
	CHECK(ClassifyPhase(p, s) == PlayPhase::Aerial);
}

TEST_CASE("grounded car with ball on roof classifies as GroundDribble") {
	auto p = MakePlayer({0, 0, 17}, true);
	auto s = MakeState({0, 50, 140});
	CHECK(ClassifyPhase(p, s) == PlayPhase::GroundDribble);
}

TEST_CASE("tumbling car far from ball classifies as Recover") {
	auto p = MakePlayer({0, 0, 300}, false);
	auto s = MakeState({3000, 3000, 100});
	// Ball below aerialBallZ, so not Aerial; airborne and far, so Recover.
	CHECK(ClassifyPhase(p, s) == PlayPhase::Recover);
}

TEST_CASE("ball deep in own half classifies as Defend, mirrored by team") {
	auto blue = MakePlayer({0, 0, 17}, true, Team::BLUE);
	auto orange = MakePlayer({0, 0, 17}, true, Team::ORANGE);
	auto ballBlueSide = MakeState({0, -4000, 100});
	auto ballOrangeSide = MakeState({0, 4000, 100});
	CHECK(ClassifyPhase(blue, ballBlueSide) == PlayPhase::Defend);
	CHECK(ClassifyPhase(orange, ballOrangeSide) == PlayPhase::Defend);
	CHECK(ClassifyPhase(blue, ballOrangeSide) == PlayPhase::Neutral);
}

TEST_CASE("grounded car mid-field with distant ball classifies as Neutral") {
	auto p = MakePlayer({0, 0, 17}, true);
	auto s = MakeState({1000, 1000, 100});
	CHECK(ClassifyPhase(p, s) == PlayPhase::Neutral);
}

TEST_CASE("PlayPhaseName covers every phase") {
	for (int i = 0; i < PLAY_PHASE_COUNT; i++)
		CHECK(std::string(PlayPhaseName(static_cast<PlayPhase>(i))) != "Unknown");
}
