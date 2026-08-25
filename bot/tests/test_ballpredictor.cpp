#include "doctest/doctest.h"
#include "TestCommon.h"

// TestCommon.h pulls in RLGymCPP/Framework.h, which is what brings RocketSim's
// Arena into scope. There is no RLGymCPP/Sim/... include path.

TEST_CASE("test harness can create a RocketSim arena") {
	Dash::Test::EnsureRocketSim();
	RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	REQUIRE(arena != nullptr);
	CHECK(arena->_cars.empty());
	delete arena;
}
