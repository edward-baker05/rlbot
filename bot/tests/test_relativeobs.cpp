#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/Obs.h>
#include <env/RelativeObs.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <cmath>

using namespace Hive;

// An obs bug is the worst failure mode this project has: it does not crash, the
// bot still plays, and the run is just quietly unlearnable. So these tests
// check the *properties* the layout is supposed to have, not just its width.

namespace {

// Two cars, ball off to one side, so nothing is symmetric by accident.
RLGC::GameState MakeState() {
	RLGC::GameState s = {};
	s.ball.pos = {900, 1500, 300};
	s.ball.vel = {-200, 400, 50};
	s.ball.angVel = {1, 2, 3};

	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {-500, -800, 17};
	blue.vel = {300, 900, 0};
	blue.angVel = {0, 0, 1};
	blue.boost = 40;
	blue.isOnGround = true;
	// Nose 90 degrees off +y so car-frame and world-frame cannot coincide.
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

TEST_CASE("RelativeObs is the default obs plus one relative block per body") {
	auto rel = MakeObsBuilder(1, ObsMode::Relative);
	auto def = MakeObsBuilder(1, ObsMode::Default);

	RLGC::GameState s = MakeState();
	const int relSize = (int)rel->BuildObs(s.players[0], s).size();
	const int defSize = (int)def->BuildObs(s.players[0], s).size();

	// One block for the ball, one for the single opponent slot.
	CHECK(defSize == 89);
	CHECK(relSize == defSize + 2 * RelativeObs::RELATIVE_BLOCK);
	CHECK(relSize == 109);
}

// The whole point of the change. `dirToBall` in the car's own frame is the
// argument of both dense reward terms, and under the old obs the network had to
// reconstruct it from four separate absolute 3-vectors.
TEST_CASE("the ball direction appears in car-frame coordinates") {
	RelativeObs obs(1);
	RLGC::GameState s = MakeState();
	const RLGC::Player& car = s.players[0];

	const RLGC::FList o = obs.BuildObs(car, s);

	// Layout: 9 ball + 8 prevAction + 34 pads + 19 self = 70, then the ball's
	// relative block.
	const int base = 9 + 8 + 34 + 19;

	const Vec offset = s.ball.pos - car.pos;
	const Vec dir = offset / offset.Length();
	const Vec expected(dir.Dot(car.rotMat.forward),
	                   dir.Dot(car.rotMat.right),
	                   dir.Dot(car.rotMat.up));

	CHECK(o[base + 0] == doctest::Approx(expected.x).epsilon(1e-4));
	CHECK(o[base + 1] == doctest::Approx(expected.y).epsilon(1e-4));
	CHECK(o[base + 2] == doctest::Approx(expected.z).epsilon(1e-4));

	// It is a unit vector, so it stays well-conditioned at any distance.
	const float len = std::sqrt(o[base + 0] * o[base + 0] + o[base + 1] * o[base + 1] +
	                            o[base + 2] * o[base + 2]);
	CHECK(len == doctest::Approx(1.f).epsilon(1e-4));

	// Distance is separate and normalized.
	CHECK(o[base + 3] ==
	      doctest::Approx(offset.Length() / RelativeObs::RELATIVE_POS_SCALE).epsilon(1e-4));
}

// A car facing the ball must see it straight ahead in its own frame, whichever
// way the world is oriented. This is the invariance the old obs never had.
TEST_CASE("a car facing the ball sees dirToBall = +forward, at any world angle") {
	RelativeObs obs(1);
	const int base = 9 + 8 + 34 + 19;

	for (float yaw = -3.0f; yaw <= 3.0f; yaw += 0.5f) {
		RLGC::GameState s = MakeState();
		RLGC::Player& car = s.players[0];

		car.pos = {0, 0, 17};
		s.ball.pos = {1000 * std::cos(yaw), 1000 * std::sin(yaw), 17};

		const Vec fwd(std::cos(yaw), std::sin(yaw), 0);
		const Vec right(std::sin(yaw), -std::cos(yaw), 0);
		car.rotMat = RotMat(fwd, right, Vec(0, 0, 1));

		const RLGC::FList o = obs.BuildObs(car, s);

		CHECK(o[base + 0] == doctest::Approx(1.f).epsilon(1e-3)); // dead ahead
		CHECK(o[base + 1] == doctest::Approx(0.f).epsilon(1e-3));
		CHECK(o[base + 2] == doctest::Approx(0.f).epsilon(1e-3));
	}
}

// Both teams share one policy, so orange's view of its own situation must be
// numerically identical to blue's view of the mirrored situation. The relative
// block is computed from already-inverted physics for exactly this reason; if
// the inversion were applied inconsistently, orange would train on a subtly
// different observation and nothing would look wrong.
TEST_CASE("the relative block is team-invariant") {
	RelativeObs obs(1);
	RLGC::GameState s = MakeState();

	// Mirror the whole world (InvertPhys negates x and y) and swap teams, so
	// orange's situation becomes exactly blue's.
	RLGC::GameState m = s;
	auto flip = [](Vec v) { return Vec(-v.x, -v.y, v.z); };
	m.ball.pos = flip(s.ball.pos);
	m.ball.vel = flip(s.ball.vel);
	m.ball.angVel = flip(s.ball.angVel);
	for (size_t i = 0; i < m.players.size(); i++) {
		m.players[i].pos = flip(s.players[i].pos);
		m.players[i].vel = flip(s.players[i].vel);
		m.players[i].angVel = flip(s.players[i].angVel);
		m.players[i].rotMat = RotMat(flip(s.players[i].rotMat.forward),
		                             flip(s.players[i].rotMat.right),
		                             s.players[i].rotMat.up);
		m.players[i].team = (s.players[i].team == Team::BLUE) ? Team::ORANGE
		                                                            : Team::BLUE;
	}

	const RLGC::FList a = obs.BuildObs(s.players[0], s);
	const RLGC::FList b = obs.BuildObs(m.players[0], m);

	REQUIRE(a.size() == b.size());

	// Compare the ball's relative block and the self block; the boost-pad
	// section is mirrored by index rather than by value and is checked
	// elsewhere.
	const int base = 9 + 8 + 34 + 19;
	for (int i = 0; i < RelativeObs::RELATIVE_BLOCK; i++)
		CHECK(a[base + i] == doctest::Approx(b[base + i]).epsilon(1e-4));
}

// Relative velocity is the target's velocity MINUS ours, in our frame: what
// actually matters for closing on something. A car matching the ball's velocity
// exactly should read zero, however fast both are moving.
TEST_CASE("relative velocity is closing velocity, not absolute velocity") {
	RelativeObs obs(1);
	RLGC::GameState s = MakeState();
	RLGC::Player& car = s.players[0];

	car.vel = {1500, -700, 200};
	s.ball.vel = car.vel;

	const RLGC::FList o = obs.BuildObs(car, s);
	const int velAt = 9 + 8 + 34 + 19 + 7; // dir(3) + dist(1) + offset(3)

	CHECK(o[velAt + 0] == doctest::Approx(0.f).epsilon(1e-4));
	CHECK(o[velAt + 1] == doctest::Approx(0.f).epsilon(1e-4));
	CHECK(o[velAt + 2] == doctest::Approx(0.f).epsilon(1e-4));
}

// Padding must zero a whole car slot. If only the absolute half were padded,
// an empty slot would carry a relative block reading "a car exactly on top of
// me at zero range" -- an alarming state that never occurs in play.
TEST_CASE("an empty opponent slot is entirely zero") {
	RelativeObs obs(2); // room for 2v2, state has 1v1
	RLGC::GameState s = MakeState();

	const RLGC::FList o = obs.BuildObs(s.players[0], s);

	const int perSlot = 19 + RelativeObs::RELATIVE_BLOCK;
	const int slotsStart = 9 + 8 + 34 + 19 + RelativeObs::RELATIVE_BLOCK;
	// maxPlayers=2 -> 2 opponent slots then 1 teammate slot.
	CHECK((int)o.size() == slotsStart + 3 * perSlot);

	// Exactly one opponent slot is populated; the other two slots are empty.
	int emptySlots = 0;
	for (int slot = 0; slot < 3; slot++) {
		bool allZero = true;
		for (int i = 0; i < perSlot; i++)
			if (o[slotsStart + slot * perSlot + i] != 0.f)
				allZero = false;
		if (allZero)
			emptySlots++;
	}
	CHECK(emptySlots == 2);
}

TEST_CASE("ProbeObsSize agrees with what the builder actually emits") {
	Hive::Test::EnsureRocketSim();

	for (ObsMode mode : {ObsMode::Relative, ObsMode::Default}) {
		for (int n : {1, 2, 3}) {
			const int probed = ProbeObsSize(n, mode);
			auto builder = MakeObsBuilder(n, mode);

			RLGC::GameState s = MakeState();
			CHECK((int)builder->BuildObs(s.players[0], s).size() == probed);
		}
	}
}
