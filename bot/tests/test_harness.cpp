#include "doctest/doctest.h"
#include "TestCommon.h"

using namespace RLGC;

TEST_CASE("harness runs and RocketSim initializes") {
	CHECK(1 + 1 == 2);
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	CHECK(arena != nullptr);
	delete arena;
}
