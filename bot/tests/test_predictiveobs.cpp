#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/BallPredictor.h>
#include <env/Obs.h>
#include <env/PredictiveObs.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cmath>

using namespace Dash;

// An obs bug is the worst failure mode this project has: it does not crash, the
// bot still plays, and the run is just quietly unlearnable. These check the
// *properties* the layout is supposed to have, not just its width.

namespace {

RLGC::GameState MakeState() {
	RLGC::GameState s = {};
	s.ball.pos = {900, 1500, 300};
	s.ball.vel = {-200, 400, 50};
	s.ball.angVel = {1, 2, 3};
	s.lastTickCount = 0;

	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {-500, -800, 17};
	blue.vel = {300, 900, 0};
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

TEST_CASE("PredictiveObs is the advanced obs plus exactly one prediction block") {
	Dash::Test::EnsureRocketSim();
	auto predictive = MakeObsBuilder(3, ObsMode::Predictive);
	auto advanced = MakeObsBuilder(3, ObsMode::Advanced);

	RLGC::GameState s = MakeState();
	const int predSize = (int)predictive->BuildObs(s.players[0], s).size();
	const int advSize = (int)advanced->BuildObs(s.players[0], s).size();

	CHECK(advSize == 225);
	CHECK(predSize == advSize + PredictiveObs::PREDICT_BLOCK);
	CHECK(predSize == 249);
}

TEST_CASE("PredictiveObs leaves the existing dimensions untouched") {
	Dash::Test::EnsureRocketSim();
	auto predictive = MakeObsBuilder(3, ObsMode::Predictive);
	auto advanced = MakeObsBuilder(3, ObsMode::Advanced);

	RLGC::GameState s = MakeState();
	// One player only: the padded slots are shuffled, so a two-player state
	// would differ between builders by shuffle order alone.
	s.players.resize(1);

	const RLGC::FList p = predictive->BuildObs(s.players[0], s);
	const RLGC::FList a = advanced->BuildObs(s.players[0], s);

	REQUIRE(p.size() == a.size() + PredictiveObs::PREDICT_BLOCK);
	for (size_t i = 0; i < a.size(); i++)
		CHECK(p[i] == doctest::Approx(a[i]));
}

TEST_CASE("PredictiveObs samples are the predicted ball in the car's frame") {
	Dash::Test::EnsureRocketSim();
	PredictiveObs obs(3);
	BallPredictor reference;

	RLGC::GameState s = MakeState();
	s.players.resize(1);

	const RLGC::FList result = obs.BuildObs(s.players[0], s);
	const BallTrajectory& t = reference.Get(s);

	const size_t base = result.size() - PredictiveObs::PREDICT_BLOCK;
	const RLGC::Player& car = s.players[0];

	for (int k = 0; k < BallPredictor::NUM_SAMPLES; k++) {
		const Vec expected =
			car.rotMat.Dot(t.pos[BallPredictor::SAMPLE_TICKS[k]] - car.pos) *
			RLGC::AdvancedObs::POS_COEF;

		CHECK(result[base + k * 3 + 0] == doctest::Approx(expected.x).epsilon(0.001));
		CHECK(result[base + k * 3 + 1] == doctest::Approx(expected.y).epsilon(0.001));
		CHECK(result[base + k * 3 + 2] == doctest::Approx(expected.z).epsilon(0.001));
	}
}

TEST_CASE("PredictiveObs mirrors the prediction block for orange") {
	Dash::Test::EnsureRocketSim();
	PredictiveObs obs(3);

	// A state and its exact mirror. Blue's view of one must equal orange's
	// view of the other, or the two teams are learning different games.
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

	const RLGC::FList bObs = obs.BuildObs(s.players[0], s);
	const RLGC::FList oObs = obs.BuildObs(m.players[0], m);

	const size_t base = bObs.size() - PredictiveObs::PREDICT_BLOCK;
	for (size_t i = base; i < bObs.size(); i++)
		CHECK(bObs[i] == doctest::Approx(oObs[i]).epsilon(0.001));
}

TEST_CASE("PredictiveObs event features encode a goal-bound ball") {
	Dash::Test::EnsureRocketSim();
	PredictiveObs obs(3);

	RLGC::GameState s = {};
	s.ball.pos = {0, 3000, 93};
	s.ball.vel = {0, 3000, 0};
	s.lastTickCount = 0;
	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {0, 0, 17};
	blue.rotMat = RotMat(Vec(0, 1, 0), Vec(1, 0, 0), Vec(0, 0, 1));
	s.players = {blue};

	const RLGC::FList result = obs.BuildObs(s.players[0], s);
	const size_t goalFlag = result.size() - 2;
	const size_t goalTime = result.size() - 1;

	// Heading into orange's net, seen from blue: scoring, so +1.
	CHECK(result[goalFlag] == doctest::Approx(1.f));
	CHECK(result[goalTime] < 1.f);
	CHECK(result[goalTime] > 0.f);
}

TEST_CASE("PredictiveObs emits finite values for a resting ball") {
	Dash::Test::EnsureRocketSim();
	PredictiveObs obs(3);

	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 93};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;
	RLGC::Player blue = {};
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = {0, -1000, 17};
	blue.rotMat = RotMat(Vec(0, 1, 0), Vec(1, 0, 0), Vec(0, 0, 1));
	s.players = {blue};

	const RLGC::FList result = obs.BuildObs(s.players[0], s);
	for (float v : result)
		CHECK(std::isfinite(v));

	// No bounce and no goal: both times saturate, flag is neutral.
	CHECK(result[result.size() - 6] == doctest::Approx(1.f)); // bounce time
	CHECK(result[result.size() - 2] == doctest::Approx(0.f)); // goal flag
	CHECK(result[result.size() - 1] == doctest::Approx(1.f)); // goal time
}
