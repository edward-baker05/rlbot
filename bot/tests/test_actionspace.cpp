#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/Actions.h>

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <utility>

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

// THE PRIOR THAT ACTUALLY MATTERS.
//
// The test above pins the raw table at 18/90 = 20% jump actions. That is the
// number Python RLGym's LookupAction exposes, because it applies no mask. It is
// NOT the number this project's policy sees: `DefaultAction::GetActionMask`
// offers a grounded car only the 24 ground actions plus the 18 jump actions, so
// its jump prior is 18/42 = 42.9% -- more than double.
//
// Ground dwell is 1/p_jump decision steps and an air stint is ~15, so a UNIFORM
// policy is airborne ~87% masked against ~75% unmasked. p1advnorm measured
// `Player/In Air Ratio` 0.886 at a jump rate of 0.43, matching the masked
// figure. Eight runs were spent fighting air time that the mask doubles the
// prior of, and the mask was never in the comparison because only the raw table
// was pinned.
//
// These are the null values for `Action/Jump When Grounded *` in
// docs/metrics.md. If they move, that document is wrong.
TEST_CASE("grounded jump prior: masked 42.9%, unmasked 20%") {
	RLGC::GameState state = {};
	RLGC::Player grounded = {};
	grounded.isOnGround = true;
	grounded.boost = 100;

	auto jumpShare = [&](RLGC::DefaultAction& parser) {
		const auto mask = parser.GetActionMask(grounded, state);
		int available = 0, jumps = 0;
		for (size_t i = 0; i < mask.size(); i++) {
			if (!mask[i])
				continue;
			available++;
			if (parser.actions[i].jump)
				jumps++;
		}
		REQUIRE(available > 0);
		return std::pair<int, float>{available, float(jumps) / float(available)};
	};

	RLGC::DefaultAction masked;
	auto [maskedCount, maskedShare] = jumpShare(masked);
	CHECK(maskedCount == 42);
	CHECK(maskedShare == doctest::Approx(18.f / 42.f).epsilon(1e-4)); // 0.4286

	Hive::UnmaskedAction unmasked;
	auto [unmaskedCount, unmaskedShare] = jumpShare(unmasked);
	CHECK(unmaskedCount == 90);
	CHECK(unmaskedShare == doctest::Approx(18.f / 90.f).epsilon(1e-4)); // 0.20

	CHECK(maskedShare > 2.f * unmaskedShare);
}

// A DRY CAR HAS A 50% JUMP PRIOR, and the reason is an ordering quirk upstream.
//
// `GetActionMask` removes the boost actions when `player.boost == 0`, and THEN
// unconditionally ORs the jump mask back in. Nine of the eighteen jump actions
// press boost, so a car with an empty tank gets them re-enabled: 24 ground - 6
// boosted = 18, plus all 18 jump = 36 available, of which 18 jump.
//
// Two consequences. Those nine actions are dead inputs -- boost does nothing at
// zero -- so the policy spends probability mass on no-ops the mask exists to
// remove. And the jump prior rises from 42.9% to 50% in exactly the state this
// bot lives in: p7approach ran at `Player/Boost` 12-15 out of 100, falling over
// the run.
//
// Not fixed here. It is upstream behaviour, it is part of what the port is
// measuring, and changing it would be a second variable. Recorded so the null
// for `Action/Jump When Grounded *` is known to be a RANGE, not a point.
TEST_CASE("a dry car's masked jump prior is 50%, not 42.9%") {
	RLGC::GameState state = {};
	RLGC::Player dry = {};
	dry.isOnGround = true;
	dry.boost = 0;

	RLGC::DefaultAction masked;
	const auto mask = masked.GetActionMask(dry, state);
	int available = 0, jumps = 0, deadBoost = 0;
	for (size_t i = 0; i < mask.size(); i++) {
		if (!mask[i])
			continue;
		available++;
		if (masked.actions[i].jump)
			jumps++;
		// Offered despite an empty tank, purely because jumpMask is applied
		// after the boost filter.
		if (masked.actions[i].boost)
			deadBoost++;
	}

	CHECK(available == 36);
	CHECK(jumps == 18);
	CHECK(float(jumps) / float(available) == doctest::Approx(0.5f).epsilon(1e-4));

	// Every boost action still on offer is a jump action, which is the
	// signature of the ordering: the ground ones were correctly removed.
	CHECK(deadBoost == 9);
}

// Nulls for `Action/Steer Nonzero`, which is sampled only on grounded upright
// steps (Train.cpp). Every jump action carries steer == 0 -- the aerial block
// sets `steer = yaw` and forces `yaw == 0` whenever `jump == 1` -- so adding
// the jump mask to a grounded car dilutes the steering prior as well as
// inflating the jump one.
//
// This is why "the bot stopped steering" is harder to read than it looks:
// p7approach's 0.087 has to be judged against 0.381, not against 0.5.
TEST_CASE("grounded steer-nonzero prior: masked 16/42, unmasked 48/90") {
	RLGC::GameState state = {};
	RLGC::Player grounded = {};
	grounded.isOnGround = true;
	grounded.boost = 100;

	auto steerShare = [&](RLGC::DefaultAction& parser) {
		const auto mask = parser.GetActionMask(grounded, state);
		int available = 0, steering = 0;
		for (size_t i = 0; i < mask.size(); i++) {
			if (!mask[i])
				continue;
			available++;
			if (parser.actions[i].steer != 0.f)
				steering++;
		}
		return std::pair<int, int>{steering, available};
	};

	RLGC::DefaultAction masked;
	auto [mSteer, mAvail] = steerShare(masked);
	CHECK(mAvail == 42);
	CHECK(mSteer == 16);
	CHECK(float(mSteer) / float(mAvail) == doctest::Approx(0.3810f).epsilon(1e-3));

	Hive::UnmaskedAction unmasked;
	auto [uSteer, uAvail] = steerShare(unmasked);
	CHECK(uAvail == 90);
	CHECK(uSteer == 48);
	CHECK(float(uSteer) / float(uAvail) == doctest::Approx(0.5333f).epsilon(1e-3));

	// No jump action can steer, which is the whole mechanism.
	int jumpSteering = 0;
	for (const auto& a : masked.actions)
		if (a.jump && a.steer != 0.f)
			jumpSteering++;
	CHECK(jumpSteering == 0);
}
