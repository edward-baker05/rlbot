#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/Obs.h>
#include <env/PadGeometryObs.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <iterator>

using namespace Dash;

namespace {

// A pattern with no mirror symmetry of its own, so a test that passes under
// mirroring is testing the obs rather than the fixture.
bool PadUp(int i) { return (i * 7 + 3) % 5 != 0; }

RLGC::GameState MakeState() {
	RLGC::GameState s = {};
	s.ball.pos = {900, 1500, 300};
	s.ball.vel = {-200, 400, 50};
	s.lastTickCount = 0;

	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {-500, -800, 17};
	blue.vel = {300, 900, 0};
	blue.boost = 40;
	blue.isOnGround = true;
	blue.rotMat = RotMat(Vec(1, 0, 0), Vec(0, -1, 0), Vec(0, 0, 1));

	RLGC::Player orange = blue;
	orange.carId = 2;
	orange.team = Team::ORANGE;
	orange.pos = {1200, 2000, 17};
	orange.vel = {-100, -600, 0};
	orange.rotMat = RotMat(Vec(0, -1, 0), Vec(-1, 0, 0), Vec(0, 0, 1));

	s.players = {blue, orange};
	return s;
}

} // namespace

TEST_CASE("PadGeometryObs is the predictive obs plus exactly one pad block") {
	Dash::Test::EnsureRocketSim();
	auto padGeometry = MakeObsBuilder(3, ObsMode::PadGeometry);
	auto predictive = MakeObsBuilder(3, ObsMode::Predictive);

	RLGC::GameState s = MakeState();
	const RLGC::FList p = padGeometry->BuildObs(s.players[0], s);
	const RLGC::FList q = predictive->BuildObs(s.players[0], s);

	CHECK(p.size() == q.size() + PadGeometryObs::PAD_BLOCK);

	// A pure append is what lets migrate-obs widen an existing run rather than
	// forcing a restart, so the prefix has to be untouched.
	for (size_t i = 0; i < q.size(); i++)
		CHECK(p[i] == doctest::Approx(q[i]));
}

TEST_CASE("PadGeometryObs reports the big pads in the car's frame") {
	Dash::Test::EnsureRocketSim();
	PadGeometryObs obs(3);

	RLGC::GameState s = MakeState();
	const RLGC::FList result = obs.BuildObs(s.players[0], s);
	const RLGC::Player &car = s.players[0];

	const size_t base = result.size() - PadGeometryObs::PAD_BLOCK;
	for (size_t n = 0; n < std::size(PadGeometryObs::BIG_PADS); n++) {
		const Vec expected =
			car.rotMat.Dot(
				RLGC::CommonValues::BOOST_LOCATIONS[PadGeometryObs::BIG_PADS[n]] -
				car.pos) *
			RLGC::AdvancedObs::POS_COEF;

		CHECK(result[base + n * 4 + 0] == doctest::Approx(expected.x).epsilon(0.001));
		CHECK(result[base + n * 4 + 1] == doctest::Approx(expected.y).epsilon(0.001));
		CHECK(result[base + n * 4 + 2] == doctest::Approx(expected.z).epsilon(0.001));
	}
}

TEST_CASE("PadGeometryObs reports the small pads nearest the car, nearest first") {
	Dash::Test::EnsureRocketSim();
	PadGeometryObs obs(3);

	RLGC::GameState s = MakeState();
	const RLGC::FList result = obs.BuildObs(s.players[0], s);
	const RLGC::Player &car = s.players[0];

	const size_t base = result.size() - PadGeometryObs::PAD_BLOCK +
						std::size(PadGeometryObs::BIG_PADS) * 4;

	float prevDist = -1.f;
	for (int n = 0; n < PadGeometryObs::NEAREST_SMALL; n++) {
		// Undo the frame change: the car-local vector's length is the distance.
		const Vec local = {result[base + n * 4 + 0], result[base + n * 4 + 1],
						   result[base + n * 4 + 2]};
		const float dist = local.Length() / RLGC::AdvancedObs::POS_COEF;

		CHECK(dist >= prevDist);
		prevDist = dist;
	}

	// Nothing nearer than the sixth reported pad may have been skipped.
	int nearer = 0;
	for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		bool big = false;
		for (int b : PadGeometryObs::BIG_PADS)
			big |= (b == i);
		if (big)
			continue;
		if (RLGC::CommonValues::BOOST_LOCATIONS[i].Dist(car.pos) <= prevDist + 1.f)
			nearer++;
	}
	CHECK(nearer == PadGeometryObs::NEAREST_SMALL);
}

TEST_CASE("PadGeometryObs mirrors the pad block for orange") {
	Dash::Test::EnsureRocketSim();
	PadGeometryObs obs(3);

	// A state and its exact mirror. Pad slot i has to mean the same pad, at the
	// same place, to whichever team is reading it -- BOOST_LOCATIONS is
	// symmetric under the inversion, which is what makes that true.
	RLGC::GameState s = {};
	s.ball.pos = {300, 1200, 400};
	s.ball.vel = {150, -600, 100};
	s.lastTickCount = 0;
	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {-700, -900, 17};
	blue.vel = {200, 500, 0};
	blue.rotMat = RotMat(Vec(0, 1, 0), Vec(1, 0, 0), Vec(0, 0, 1));
	s.players = {blue};

	RLGC::GameState m = s;
	m.ball.pos = {-300, -1200, 400};
	m.ball.vel = {-150, 600, 100};
	RLGC::Player orange = blue;
	orange.team = Team::ORANGE;
	orange.pos = {700, 900, 17};
	orange.vel = {-200, -500, 0};
	orange.rotMat = RotMat(Vec(0, -1, 0), Vec(-1, 0, 0), Vec(0, 0, 1));
	m.players = {orange};

	for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		s.boostPads[i] = PadUp(i);
		s.boostPadTimers[i] = PadUp(i) ? 0.f : 1.5f;
		m.boostPadsInv[i] = PadUp(i);
		m.boostPadTimersInv[i] = PadUp(i) ? 0.f : 1.5f;
	}

	const RLGC::FList bObs = obs.BuildObs(s.players[0], s);
	const RLGC::FList oObs = obs.BuildObs(m.players[0], m);

	const size_t base = bObs.size() - PadGeometryObs::PAD_BLOCK;
	for (size_t i = base; i < bObs.size(); i++)
		CHECK(bObs[i] == doctest::Approx(oObs[i]).epsilon(0.001));
}

TEST_CASE("PadGeometryObs does not depend on the order players are listed in") {
	Dash::Test::EnsureRocketSim();
	PadGeometryObs obs(3);

	RLGC::GameState s = MakeState();
	RLGC::Player extra = s.players[1];
	extra.carId = 3;
	extra.pos = {-200, 2400, 17};
	s.players.push_back(extra);

	RLGC::GameState swapped = s;
	std::swap(swapped.players[1], swapped.players[2]);

	const RLGC::FList a = obs.BuildObs(s.players[0], s);
	const RLGC::FList b = obs.BuildObs(swapped.players[0], swapped);

	REQUIRE(a.size() == b.size());
	for (size_t i = 0; i < a.size(); i++)
		CHECK(a[i] == doctest::Approx(b[i]));
}
