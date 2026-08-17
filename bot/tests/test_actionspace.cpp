#include "doctest/doctest.h"
#include "TestCommon.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

// The reward design depends on what a near-uniform policy does with this
// action table: if a large share of actions press jump, a fresh policy is
// airborne almost all the time, and any reward that pays for velocity
// (rather than for being on the ground and driving) is farmed by tumbling.
// Pinned here so an upstream change to the table shows up as a failing test
// rather than as a mysteriously unlearnable run.
TEST_CASE("DefaultAction table composition matches what reward design assumes") {
	RLGC::DefaultAction parser;
	const auto& actions = parser.actions;

	REQUIRE(actions.size() == 90);

	int jumps = 0;
	for (const auto& a : actions)
		if (a.jump)
			jumps++;

	CHECK(jumps == 18); // 20% of actions press jump
}
