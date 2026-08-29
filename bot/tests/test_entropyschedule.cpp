#include "doctest/doctest.h"

#include <Config.h>

using namespace Dash;

// The schedule is a pure function of the step count on purpose -- that is what
// makes it survive a resume -- so it is worth pinning exactly.

TEST_CASE("EntropyTargetAt holds the target until the anchor") {
	CHECK(EntropyTargetAt(0, 0.49f, 0.25f, 0.02f, 4'000'000'000LL) == doctest::Approx(0.49f));
	CHECK(EntropyTargetAt(3'000'000'000LL, 0.49f, 0.25f, 0.02f, 4'000'000'000LL) ==
		  doctest::Approx(0.49f));
	CHECK(EntropyTargetAt(4'000'000'000LL, 0.49f, 0.25f, 0.02f, 4'000'000'000LL) ==
		  doctest::Approx(0.49f));
}

TEST_CASE("EntropyTargetAt decays linearly per billion steps") {
	const int64_t from = 4'000'000'000LL;
	CHECK(EntropyTargetAt(from + 1'000'000'000LL, 0.49f, 0.25f, 0.02f, from) ==
		  doctest::Approx(0.47f));
	CHECK(EntropyTargetAt(from + 5'000'000'000LL, 0.49f, 0.25f, 0.02f, from) ==
		  doctest::Approx(0.39f));
	CHECK(EntropyTargetAt(from + 10'000'000'000LL, 0.49f, 0.25f, 0.02f, from) ==
		  doctest::Approx(0.29f));
}

TEST_CASE("EntropyTargetAt never goes below the floor") {
	const int64_t from = 1;
	CHECK(EntropyTargetAt(12'000'000'000LL, 0.49f, 0.25f, 0.02f, from) ==
		  doctest::Approx(0.25f));
	CHECK(EntropyTargetAt(500'000'000'000LL, 0.49f, 0.25f, 0.02f, from) ==
		  doctest::Approx(0.25f));
}

TEST_CASE("EntropyTargetAt is disabled by a floor at or above the target") {
	CHECK(EntropyTargetAt(9'000'000'000LL, 0.49f, 0.49f, 0.02f, 1) == doctest::Approx(0.49f));
	CHECK(EntropyTargetAt(9'000'000'000LL, 0.49f, 0.60f, 0.02f, 1) == doctest::Approx(0.49f));
	CHECK(EntropyTargetAt(9'000'000'000LL, 0.49f, 0.25f, 0.0f, 1) == doctest::Approx(0.49f));
}

TEST_CASE("EntropyTargetAt treats a missing anchor as disabled, not as step zero") {
	// Forgetting --entropy-decay-from must be a no-op, not an 0.08 drop.
	CHECK(EntropyTargetAt(4'152'675'048LL, 0.49f, 0.25f, 0.02f, 0) ==
		  doctest::Approx(0.49f));
	CHECK(EntropyTargetAt(50'000'000'000LL, 0.49f, 0.25f, 0.02f, 0) ==
		  doctest::Approx(0.49f));
}

TEST_CASE("EntropyTargetAt is resume-invariant") {
	// The whole point: the value at a step does not depend on how many times
	// the run was stopped and restarted on the way there.
	const int64_t from = 4'152'675'048LL;
	for (int64_t step = from; step < from + 8'000'000'000LL; step += 137'000'000LL) {
		const float a = EntropyTargetAt(step, 0.49f, 0.25f, 0.02f, from);
		const float b = EntropyTargetAt(step, 0.49f, 0.25f, 0.02f, from);
		CHECK(a == doctest::Approx(b));
		CHECK(a <= 0.49f);
		CHECK(a >= 0.25f);
	}
}
