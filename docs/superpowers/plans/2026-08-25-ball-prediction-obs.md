# Ball-Prediction Observation Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 24-dimension ball-prediction block to the observation and graft the widened input onto t1's 890M-step checkpoint without changing its behaviour.

**Architecture:** A `BallPredictor` owns a car-less RocketSim `Arena` and produces a tick-indexed ball trajectory, re-simulating only when the live ball diverges from its own prediction. A new `PredictiveObs` obs builder extends `AdvancedObsPadded` with six geometrically-spaced predicted positions in car-local frame plus six bounce/goal event features. A `migrate-obs` subcommand zero-pads the shared head's first Linear so the wider input is exactly behaviour-preserving.

**Tech Stack:** C++20, CMake, RocketSim (via RLGymCPP), libtorch, doctest.

**Spec:** `docs/superpowers/specs/2026-08-25-ball-prediction-obs-design.md`

## Global Constraints

- **Never modify `external/`.** Six local patches live there and `scripts/apply_external_patches.py --check` runs on every build via the `check_external_patches` target. Changes go in `bot/`.
- **Never touch `bot/build/checkpoints/t1/` in place.** t1 is a live 890M-step run. All migration output goes to new directories.
- Build type defaults to `Release`. A Debug RocketSim build is ~10x slower and will make the Task 5 measurement meaningless.
- Existing obs modes `Default`, `Advanced`, `Relative` must keep byte-identical output. The `t2`, `main-p19pool`, `687282952` and `747642208` checkpoints must still load.
- Obs sizes, verified against the live checkpoint: old = **225**, new = **249**, block = **24**.
- Sample times `0.15, 0.30, 0.55, 0.95, 1.60, 2.60` s → tick offsets `18, 36, 66, 114, 192, 312` at 120 Hz.
- Simulation horizon = 6 s = **720 ticks**.
- Scale constants come from `RLGC::AdvancedObs`: `POS_COEF = 1/5000`, `VEL_COEF = 1/2300`.
- Build with `scripts/build.sh` (it reapplies the external patches; a bare `cmake --build` will fail the `check_external_patches` target). There is no `build.sh` at the repo root.

---

### Task 1: Restore the doctest test target

The project had a doctest suite (`HiveTests`) that was dropped in the Hive→Dash rename at
commit `6ba0bf6`. There is no test target today, so TDD for the rest of this plan has
nowhere to live. Recover the harness from git history and rename it.

**Files:**
- Create: `bot/tests/doctest/doctest.h` (recovered from git)
- Create: `bot/tests/test_main.cpp`
- Create: `bot/tests/TestCommon.h`
- Create: `bot/tests/test_ballpredictor.cpp`
- Modify: `bot/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `Dash::Test::EnsureRocketSim()` — idempotent RocketSim init for any test touching an `Arena`. A `DashTests` binary at `bot/build/DashTests`.

- [ ] **Step 1: Recover the vendored doctest header**

```bash
mkdir -p bot/tests/doctest
git show 6ba0bf6^:bot/tests/doctest/doctest.h > bot/tests/doctest/doctest.h
```

- [ ] **Step 2: Write the doctest main**

Create `bot/tests/test_main.cpp`:

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
```

- [ ] **Step 3: Write the shared test helper**

Create `bot/tests/TestCommon.h`. This is the old `Hive::Test` helper renamed to `Dash::Test`,
with the env var updated to the `DASH_` prefix this codebase now uses (falling back to the
old `HIVE_` name the way `Train.cpp` does):

```cpp
#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <cstdlib>

namespace Dash::Test {

// RocketSim asserts if initialized twice and Arena creation asserts if never
// initialized; every test that touches an Arena funnels through here.
inline void EnsureRocketSim() {
	static bool done = false;
	if (!done) {
		const char* env = std::getenv("DASH_COLLISION_MESHES");
		if (!env)
			env = std::getenv("HIVE_COLLISION_MESHES");
		RocketSim::Init(env ? env : "collision_meshes");
		done = true;
	}
}

} // namespace Dash::Test
```

- [ ] **Step 4: Write a failing placeholder test**

Create `bot/tests/test_ballpredictor.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <RLGymCPP/Sim/Arena/Arena.h>

TEST_CASE("test harness can create a RocketSim arena") {
	Dash::Test::EnsureRocketSim();
	RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	REQUIRE(arena != nullptr);
	CHECK(arena->_cars.empty());
	delete arena;
}
```

- [ ] **Step 5: Add the test target to CMake**

In `bot/CMakeLists.txt`, immediately after the `add_executable(DashBot src/main.cpp)` /
`target_link_libraries(DashBot PRIVATE DashCore)` pair, insert:

```cmake
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# doctest is vendored as a single header in tests/doctest. DashCore exists
# precisely so the bot binary and this share one compilation of the sources.
add_executable(DashTests
	tests/test_main.cpp
	tests/test_ballpredictor.cpp
)
target_link_libraries(DashTests PRIVATE DashCore)
target_include_directories(DashTests PRIVATE tests)
```

Then find the `set_target_properties(DashBot PROPERTIES ...)` block that sets
`BUILD_RPATH`/`INSTALL_RPATH`, and the later one that sets
`RUNTIME_OUTPUT_DIRECTORY`. Apply both to `DashTests` as well by changing each
`set_target_properties(DashBot PROPERTIES` to `set_target_properties(DashBot DashTests PROPERTIES`.
The tests load the same libtorch and need the same collision meshes next to the binary.

- [ ] **Step 6: Build and run to verify the harness works**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests
```
Expected: PASS, `1 test case, 2 assertions`. If the build fails on the patch check, that
is the `check_external_patches` target doing its job — `scripts/build.sh` should have reapplied
them, so investigate rather than bypassing.

- [ ] **Step 7: Commit**

```bash
git add bot/tests bot/CMakeLists.txt
git commit -m "test: restore doctest test target as DashTests"
```

---

### Task 2: BallPredictor trajectory simulation and caching

**Files:**
- Create: `bot/src/env/BallPredictor.h`
- Create: `bot/src/env/BallPredictor.cpp`
- Modify: `bot/CMakeLists.txt` (add the two files to the `DashCore` source list)
- Test: `bot/tests/test_ballpredictor.cpp`

**Interfaces:**
- Consumes: `Dash::Test::EnsureRocketSim()` from Task 1.
- Produces:
  - `struct Dash::BallTrajectory` with members `std::vector<RocketSim::Vec> pos`, `std::vector<RocketSim::Vec> vel`, `uint64_t startTick`, `int bounceTick`, `RocketSim::Vec bouncePos`, `int goalTick`, `int goalTeam`. `bounceTick`/`goalTick` are tick offsets from `startTick`, or `-1` for "did not occur within the horizon". `goalTeam` is `0` (blue's net) / `1` (orange's net) / `-1` (none).
  - `class Dash::BallPredictor` with `const BallTrajectory& Get(const RLGC::GameState& state)`, `void Reset()`, `uint64_t SimulationCount() const`.
  - `Dash::BallPredictor::SIM_HORIZON_TICKS = 720`, `::TICK_RATE = 120`, `::SAMPLE_TICKS` (a `std::array<int, 6>`), `::NUM_SAMPLES = 6`.

Task 3 fills in `bounceTick`/`bouncePos`/`goalTick`/`goalTeam`; this task leaves them at
their "did not occur" defaults and covers only trajectory simulation and cache
invalidation.

- [ ] **Step 1: Write the failing tests**

Replace the placeholder test in `bot/tests/test_ballpredictor.cpp` with:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/BallPredictor.h>

#include <RLGymCPP/Gamestates/GameState.h>

using namespace Dash;

namespace {

// A ball in free flight, high enough that it will not hit anything for a while.
RLGC::GameState MakeFlyingState(uint64_t tick = 0) {
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 1200};
	s.ball.vel = {300, 500, 200};
	s.ball.angVel = {0, 0, 0};
	s.lastTickCount = tick;
	return s;
}

} // namespace

TEST_CASE("BallPredictor produces a full-horizon trajectory") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;
	RLGC::GameState s = MakeFlyingState();

	const BallTrajectory& t = pred.Get(s);

	CHECK(t.pos.size() == BallPredictor::SIM_HORIZON_TICKS + 1);
	CHECK(t.vel.size() == BallPredictor::SIM_HORIZON_TICKS + 1);
	CHECK(t.startTick == 0);

	// Slice 0 is the present, so it must equal the state we handed in.
	CHECK(t.pos[0].x == doctest::Approx(s.ball.pos.x));
	CHECK(t.pos[0].y == doctest::Approx(s.ball.pos.y));
	CHECK(t.pos[0].z == doctest::Approx(s.ball.pos.z));
}

TEST_CASE("BallPredictor prediction matches a real arena stepped forward") {
	Dash::Test::EnsureRocketSim();

	// Ground truth: an arena with no cars, stepped by hand.
	RocketSim::Arena* truth = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	RLGC::GameState s = MakeFlyingState();
	truth->ball->SetState([&]{
		RocketSim::BallState bs = truth->ball->GetState();
		bs.pos = s.ball.pos;
		bs.vel = s.ball.vel;
		bs.angVel = s.ball.angVel;
		return bs;
	}());

	BallPredictor pred;
	const BallTrajectory& t = pred.Get(s);

	// 240 ticks is long enough to include the first ground bounce.
	truth->Step(240);
	const RocketSim::Vec truthPos = truth->ball->GetState().pos;

	CHECK(t.pos[240].x == doctest::Approx(truthPos.x).epsilon(0.001));
	CHECK(t.pos[240].y == doctest::Approx(truthPos.y).epsilon(0.001));
	CHECK(t.pos[240].z == doctest::Approx(truthPos.z).epsilon(0.001));

	delete truth;
}

TEST_CASE("BallPredictor reuses the cached trajectory when the ball is untouched") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	const uint64_t simsAfterFirst = pred.SimulationCount();
	REQUIRE(simsAfterFirst == 1);

	// Advance 8 ticks along the predicted path -- exactly what an untouched
	// ball does over one env step at tickSkip 8.
	RLGC::GameState next = s;
	next.lastTickCount = 1008;
	next.ball.pos = first.pos[8];
	next.ball.vel = first.vel[8];

	pred.Get(next);
	CHECK(pred.SimulationCount() == simsAfterFirst); // no re-simulation
}

TEST_CASE("BallPredictor re-simulates when the ball diverges from prediction") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// A touch: same position, very different velocity.
	RLGC::GameState touched = s;
	touched.lastTickCount = 1008;
	touched.ball.vel = {-1500, -900, 400};

	pred.Get(touched);
	CHECK(pred.SimulationCount() == 2);
}

TEST_CASE("BallPredictor re-simulates when the cached horizon is exhausted") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	// Jump far enough ahead that fewer than the deepest sample (312 ticks)
	// remain in the cache.
	const int offset = BallPredictor::SIM_HORIZON_TICKS - 100;
	RLGC::GameState late = s;
	late.lastTickCount = 1000 + offset;
	late.ball.pos = first.pos[offset];
	late.ball.vel = first.vel[offset];

	pred.Get(late);
	CHECK(pred.SimulationCount() == 2);
}

TEST_CASE("BallPredictor::Reset forces the next call to re-simulate") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = MakeFlyingState(1000);
	const BallTrajectory& first = pred.Get(s);
	REQUIRE(pred.SimulationCount() == 1);

	RLGC::GameState next = s;
	next.lastTickCount = 1008;
	next.ball.pos = first.pos[8];
	next.ball.vel = first.vel[8];

	pred.Reset();
	pred.Get(next);
	CHECK(pred.SimulationCount() == 2);
}
```

- [ ] **Step 2: Add the new test file and sources to CMake**

In `bot/CMakeLists.txt`, add to the `DashCore` source list, alphabetically among the
`src/env/` entries (after `src/env/AdvancedObsPadded.h`):

```cmake
	src/env/BallPredictor.cpp
	src/env/BallPredictor.h
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests
```
Expected: build FAILS with `env/BallPredictor.h: No such file or directory`.

- [ ] **Step 4: Write the header**

Create `bot/src/env/BallPredictor.h`:

```cpp
#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Dash {

// A tick-indexed ball trajectory, sampled at RocketSim's fixed 120Hz.
// Index i is the predicted state i ticks after `startTick`; index 0 is the
// present.
struct BallTrajectory {
	std::vector<RocketSim::Vec> pos;
	std::vector<RocketSim::Vec> vel;
	uint64_t startTick = 0;

	// Tick offsets from startTick, or -1 for "did not happen in the horizon".
	int bounceTick = -1;
	RocketSim::Vec bouncePos = {};
	int goalTick = -1;
	int goalTeam = -1;  // 0 = blue's net, 1 = orange's net, -1 = none
};

// Predicts where the ball goes if nobody touches it.
//
// The trajectory is only invalidated by a touch (or a reset), so a single
// simulation is reused across many env steps. Rather than plumbing touch
// events through, invalidation is detected by comparing the live ball against
// this predictor's own prediction for the current tick: any divergence beyond
// float noise means something interfered. That covers touches, goal resets and
// state setters uniformly, with no event wiring to get out of sync.
class BallPredictor {
public:
	static constexpr int TICK_RATE = 120;

	// 6 seconds. Deliberately more than the deepest sample (2.6s) so the
	// trajectory can be consumed for several seconds before needing a redo.
	static constexpr int SIM_HORIZON_TICKS = 720;

	static constexpr int NUM_SAMPLES = 6;

	// t = 0.15, 0.30, 0.55, 0.95, 1.60, 2.60 seconds, snapped to 120Hz ticks.
	// Geometric (ratio ~1.75): prediction accuracy decays with horizon, so the
	// dimensions are spent where the prediction is still true.
	static constexpr std::array<int, NUM_SAMPLES> SAMPLE_TICKS =
		{18, 36, 66, 114, 192, 312};

	// Divergence beyond this means the ball was interfered with. RocketSim is
	// deterministic, so an untouched ball matches to float precision; these are
	// far above that noise floor and far below any real touch.
	static constexpr float POS_TOLERANCE = 1.0f;   // uu
	static constexpr float VEL_TOLERANCE = 1.0f;   // uu/s

	BallPredictor();
	~BallPredictor();

	BallPredictor(const BallPredictor&) = delete;
	BallPredictor& operator=(const BallPredictor&) = delete;

	// Returns a trajectory valid for `state`, re-simulating only if needed.
	// The reference is invalidated by the next call.
	const BallTrajectory& Get(const RLGC::GameState& state);

	// Drops the cache. Call on episode reset.
	void Reset();

	// Number of full simulations run. For the performance gate and tests.
	uint64_t SimulationCount() const { return simCount; }

private:
	void Simulate(const RLGC::GameState& state);
	bool CacheValidFor(const RLGC::GameState& state) const;

	RocketSim::Arena* arena = nullptr;
	BallTrajectory traj = {};
	bool hasCache = false;
	uint64_t simCount = 0;
};

}  // namespace Dash
```

- [ ] **Step 5: Write the implementation**

Create `bot/src/env/BallPredictor.cpp`:

```cpp
#include "BallPredictor.h"

using namespace RLGC;

namespace Dash {

BallPredictor::BallPredictor() {
	// Car-less: a ball-only arena is far cheaper to step than a full one, and
	// cars are exactly what we are predicting the absence of.
	arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);

	traj.pos.resize(SIM_HORIZON_TICKS + 1);
	traj.vel.resize(SIM_HORIZON_TICKS + 1);
}

BallPredictor::~BallPredictor() {
	delete arena;
	arena = nullptr;
}

void BallPredictor::Reset() {
	hasCache = false;
}

bool BallPredictor::CacheValidFor(const RLGC::GameState& state) const {
	if (!hasCache)
		return false;

	// A rewound clock means a new episode.
	if (state.lastTickCount < traj.startTick)
		return false;

	const uint64_t offset = state.lastTickCount - traj.startTick;

	// Keep enough runway for the deepest sample.
	if (offset + SAMPLE_TICKS.back() > (uint64_t)SIM_HORIZON_TICKS)
		return false;

	// Did the ball actually go where we said it would?
	const RocketSim::Vec posErr = state.ball.pos - traj.pos[offset];
	if (posErr.Length() > POS_TOLERANCE)
		return false;

	const RocketSim::Vec velErr = state.ball.vel - traj.vel[offset];
	if (velErr.Length() > VEL_TOLERANCE)
		return false;

	return true;
}

void BallPredictor::Simulate(const RLGC::GameState& state) {
	RocketSim::BallState bs = arena->ball->GetState();
	bs.pos = state.ball.pos;
	bs.vel = state.ball.vel;
	bs.angVel = state.ball.angVel;
	arena->ball->SetState(bs);

	traj.startTick = state.lastTickCount;
	traj.bounceTick = -1;
	traj.bouncePos = {};
	traj.goalTick = -1;
	traj.goalTeam = -1;

	traj.pos[0] = bs.pos;
	traj.vel[0] = bs.vel;

	for (int i = 1; i <= SIM_HORIZON_TICKS; i++) {
		arena->Step(1);
		const RocketSim::BallState cur = arena->ball->GetState();
		traj.pos[i] = cur.pos;
		traj.vel[i] = cur.vel;
	}

	hasCache = true;
	simCount++;
}

const BallTrajectory& BallPredictor::Get(const RLGC::GameState& state) {
	if (!CacheValidFor(state))
		Simulate(state);
	return traj;
}

}  // namespace Dash
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests -tc="BallPredictor*"
```
Expected: PASS, 6 test cases.

If "prediction matches a real arena stepped forward" fails, the two arenas have
different mutator configs — check that neither test nor predictor calls
`SetMutatorConfig`.

- [ ] **Step 7: Commit**

```bash
git add bot/src/env/BallPredictor.h bot/src/env/BallPredictor.cpp \
        bot/tests/test_ballpredictor.cpp bot/CMakeLists.txt
git commit -m "feat: add BallPredictor with touch-invalidated trajectory cache"
```

---

### Task 3: Bounce and goal event extraction

The six sampled positions straddle and miss the discontinuities that matter — the ball can
bounce between the 0.95s and 1.60s samples and nothing in the position block records it.
This task fills in the four event fields `BallTrajectory` already declares.

**Files:**
- Modify: `bot/src/env/BallPredictor.h` (add detection constants)
- Modify: `bot/src/env/BallPredictor.cpp:Simulate`
- Test: `bot/tests/test_ballpredictor.cpp`

**Interfaces:**
- Consumes: `Dash::BallTrajectory`, `Dash::BallPredictor` from Task 2.
- Produces: populated `traj.bounceTick`, `traj.bouncePos`, `traj.goalTick`, `traj.goalTeam`. No new public functions.

- [ ] **Step 1: Write the failing tests**

Append to `bot/tests/test_ballpredictor.cpp`:

```cpp
TEST_CASE("BallPredictor detects the first ground bounce") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	// Dropped from height with no horizontal motion: the first bounce is
	// straight down onto the floor, under the drop point.
	RLGC::GameState s = {};
	s.ball.pos = {500, -700, 1000};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);

	REQUIRE(t.bounceTick > 0);
	// Free fall from ~910uu of travel under 650uu/s^2 is a bit over a second.
	CHECK(t.bounceTick < 200);
	CHECK(t.bouncePos.x == doctest::Approx(500).epsilon(0.01));
	CHECK(t.bouncePos.y == doctest::Approx(-700).epsilon(0.01));
	// Contact happens at roughly one ball radius above the floor.
	CHECK(t.bouncePos.z < 150.f);

	// The bounce must reverse vertical velocity.
	CHECK(t.vel[t.bounceTick - 1].z < 0);
	CHECK(t.vel[t.bounceTick].z > 0);
}

TEST_CASE("BallPredictor reports no bounce for a ball that stays airborne") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	// Hovering near the ceiling with almost no velocity: within 6 seconds it
	// falls, so instead use a ball already at rest on the floor, which never
	// registers a fresh bounce.
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 93};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);
	CHECK(t.bounceTick == -1);
}

TEST_CASE("BallPredictor detects a goal and which net") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	// Rolling hard at the orange net (+y).
	RLGC::GameState s = {};
	s.ball.pos = {0, 3000, 93};
	s.ball.vel = {0, 3000, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);

	REQUIRE(t.goalTick > 0);
	CHECK(t.goalTeam == 1);  // orange's net
	CHECK(t.goalTick < BallPredictor::SIM_HORIZON_TICKS);
}

TEST_CASE("BallPredictor reports no goal for a ball going nowhere") {
	Dash::Test::EnsureRocketSim();
	BallPredictor pred;

	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 93};
	s.ball.vel = {0, 0, 0};
	s.lastTickCount = 0;

	const BallTrajectory& t = pred.Get(s);
	CHECK(t.goalTick == -1);
	CHECK(t.goalTeam == -1);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests -tc="BallPredictor detects*,BallPredictor reports*"
```
Expected: FAIL — `t.bounceTick > 0` fails because Task 2 leaves it at -1.

- [ ] **Step 3: Add the detection constants to the header**

In `bot/src/env/BallPredictor.h`, after the `VEL_TOLERANCE` constant, add:

```cpp
	// A bounce is a tick where velocity direction turns sharply or speed jumps.
	// Deliberately not "velocity minus gravity*dt": the ball has drag
	// (MutatorConfig::ballDrag), so free flight is not exactly ballistic and an
	// exact test would need to track that constant. Direction change needs no
	// physics constants and works for floor, wall, ceiling and corner alike.
	static constexpr float BOUNCE_COS_THRESHOLD = 0.966f;  // ~15 degrees
	static constexpr float BOUNCE_SPEED_JUMP = 100.f;      // uu/s in one tick

	// Below this speed, velocity direction is noise; a resting ball must not
	// register bounces.
	static constexpr float BOUNCE_MIN_SPEED = 50.f;        // uu/s
```

- [ ] **Step 4: Populate the event fields during simulation**

In `bot/src/env/BallPredictor.cpp`, replace the simulation loop inside `Simulate` with:

```cpp
	for (int i = 1; i <= SIM_HORIZON_TICKS; i++) {
		arena->Step(1);
		const RocketSim::BallState cur = arena->ball->GetState();
		traj.pos[i] = cur.pos;
		traj.vel[i] = cur.vel;

		if (traj.bounceTick < 0) {
			const RocketSim::Vec& prevVel = traj.vel[i - 1];
			const float prevSpeed = prevVel.Length();
			const float curSpeed = cur.vel.Length();

			if (prevSpeed > BOUNCE_MIN_SPEED && curSpeed > BOUNCE_MIN_SPEED) {
				const float cosAngle =
					prevVel.Dot(cur.vel) / (prevSpeed * curSpeed);
				const bool turned = cosAngle < BOUNCE_COS_THRESHOLD;
				const bool jumped =
					std::abs(curSpeed - prevSpeed) > BOUNCE_SPEED_JUMP;

				if (turned || jumped) {
					traj.bounceTick = i;
					traj.bouncePos = cur.pos;
				}
			}
		}

		if (traj.goalTick < 0 && arena->IsBallScored()) {
			traj.goalTick = i;
			// RS_TEAM_FROM_Y's convention: the net the ball crossed into.
			traj.goalTeam = cur.pos.y > 0 ? 1 : 0;

			// Stop here. Past the goal line the simulation is meaningless --
			// the real game would have reset -- so freeze the remainder rather
			// than feeding the network an imaginary continuation.
			for (int j = i + 1; j <= SIM_HORIZON_TICKS; j++) {
				traj.pos[j] = cur.pos;
				traj.vel[j] = {0, 0, 0};
			}
			break;
		}
	}
```

Add `#include <cmath>` to the top of the file, after the existing include.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests -tc="BallPredictor*"
```
Expected: PASS, 10 test cases.

If "detects the first ground bounce" reports a `bounceTick` of 1, `BOUNCE_MIN_SPEED` is
letting a near-stationary start register — check the ball actually starts at rest in that
test and that the guard uses both `prevSpeed` and `curSpeed`.

- [ ] **Step 6: Commit**

```bash
git add bot/src/env/BallPredictor.h bot/src/env/BallPredictor.cpp bot/tests/test_ballpredictor.cpp
git commit -m "feat: detect first bounce and goal within predicted trajectory"
```

---

### Task 4: PredictiveObs builder and ObsMode wiring

**Files:**
- Create: `bot/src/env/PredictiveObs.h`
- Create: `bot/src/env/PredictiveObs.cpp`
- Modify: `bot/src/env/Obs.h` (add `ObsMode::Predictive`, declare `ObsModeName`)
- Modify: `bot/src/env/Obs.cpp` (dispatch, `ObsModeName`)
- Modify: `bot/src/train/Train.cpp:426` (replace the two-way ternary)
- Modify: `bot/src/main.cpp` (accept `--obs predictive`, `DASH_OBS=predictive`)
- Modify: `bot/CMakeLists.txt`
- Test: `bot/tests/test_predictiveobs.cpp`

**Interfaces:**
- Consumes: `Dash::BallPredictor`, `Dash::BallTrajectory`, `BallPredictor::SAMPLE_TICKS`, `BallPredictor::NUM_SAMPLES` from Tasks 2-3.
- Produces:
  - `class Dash::PredictiveObs : public Dash::AdvancedObsPadded`, with `static constexpr int PREDICT_BLOCK = 24`.
  - `Dash::ObsMode::Predictive` enum value.
  - `const char* Dash::ObsModeName(ObsMode mode)` returning `"Default"`, `"Advanced"`, `"Relative"`, `"Predictive"`.

- [ ] **Step 1: Write the failing tests**

Create `bot/tests/test_predictiveobs.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/BallPredictor.h>
#include <env/Obs.h>
#include <env/PredictiveObs.h>

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/GameState.h>

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
```

- [ ] **Step 2: Add the new files to CMake**

In `bot/CMakeLists.txt`, add to the `DashCore` source list after the `src/env/Obs.h` entry:

```cmake
	src/env/PredictiveObs.cpp
	src/env/PredictiveObs.h
```

and add to the `DashTests` source list:

```cmake
	tests/test_predictiveobs.cpp
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests
```
Expected: build FAILS with `env/PredictiveObs.h: No such file or directory`.

- [ ] **Step 4: Write the PredictiveObs header**

Create `bot/src/env/PredictiveObs.h`:

```cpp
#pragma once

#include "AdvancedObsPadded.h"
#include "BallPredictor.h"

namespace Dash {

// AdvancedObsPadded plus where the ball is going.
//
// The policy is a memoryless MLP seeing one ball snapshot per step. Future ball
// position is computable from that snapshot plus fixed geometry, so this adds
// no information -- but the function is piecewise (bounces are
// discontinuities), which is exactly what an MLP approximates badly. Supplying
// it turns a hard approximation problem into a lookup.
class PredictiveObs : public AdvancedObsPadded {
public:
	// 6 samples x 3 (car-local position) + 6 event features.
	static constexpr int PREDICT_BLOCK = BallPredictor::NUM_SAMPLES * 3 + 6;

	// Event times are normalized against the deepest sample, not the 6s
	// simulation horizon -- the extra simulation is a caching buffer, not part
	// of the feature space.
	static constexpr float TIME_NORM_TICKS =
		(float)BallPredictor::SAMPLE_TICKS.back();

	explicit PredictiveObs(int maxPlayers = 3) : AdvancedObsPadded(maxPlayers) {}

	void Reset(const RLGC::GameState& initialState) override;

	RLGC::FList BuildObs(const RLGC::Player& player,
	                     const RLGC::GameState& state) override;

private:
	BallPredictor predictor;
};

}  // namespace Dash
```

- [ ] **Step 5: Write the PredictiveObs implementation**

Create `bot/src/env/PredictiveObs.cpp`:

```cpp
#include "PredictiveObs.h"

#include <RLGymCPP/Gamestates/StateUtil.h>

#include <cmath>

using namespace RLGC;

namespace Dash {

void PredictiveObs::Reset(const GameState& initialState) {
	AdvancedObsPadded::Reset(initialState);
	predictor.Reset();
}

FList PredictiveObs::BuildObs(const Player& player, const GameState& state) {
	FList result = AdvancedObsPadded::BuildObs(player, state);

	const bool inv = player.team == Team::ORANGE;

	// Predict in world space once per arena tick -- both cars in a 1v1 share
	// this call, and the cache makes the second one free -- then invert per
	// player, the same way every other block in this obs is built.
	const BallTrajectory& traj = predictor.Get(state);

	const PhysState self = InvertPhys(player, inv);

	for (int k = 0; k < BallPredictor::NUM_SAMPLES; k++) {
		PhysState slice = {};
		slice.pos = traj.pos[BallPredictor::SAMPLE_TICKS[k]];
		slice.vel = traj.vel[BallPredictor::SAMPLE_TICKS[k]];
		const PhysState predicted = InvertPhys(slice, inv);

		// Car-local, matching AdvancedObs::AddPlayerToObs's convention for the
		// current ball position.
		result += self.rotMat.Dot(predicted.pos - self.pos) * POS_COEF;
	}

	// --- Event features ---

	if (traj.bounceTick >= 0) {
		result += (float)traj.bounceTick / TIME_NORM_TICKS > 1.f
			? 1.f
			: (float)traj.bounceTick / TIME_NORM_TICKS;

		PhysState bounce = {};
		bounce.pos = traj.bouncePos;
		result += InvertPhys(bounce, inv).pos * POS_COEF;
	} else {
		result += 1.f;                  // saturated: no bounce in horizon
		result += Vec(0, 0, 0);         // no position to report
	}

	// Only report goals inside the sample horizon; past that the prediction is
	// counterfactual enough that a confident goal flag would be a lie.
	const bool goalInHorizon =
		traj.goalTick >= 0 &&
		(float)traj.goalTick <= TIME_NORM_TICKS;

	if (goalInHorizon) {
		// traj.goalTeam is 1 for orange's net. From blue's perspective that is
		// scoring (+1); inverting flips it, so orange sees its own +1 the same
		// way.
		const float scoring = (traj.goalTeam == 1) ? 1.f : -1.f;
		result += inv ? -scoring : scoring;
		result += (float)traj.goalTick / TIME_NORM_TICKS;
	} else {
		result += 0.f;
		result += 1.f;
	}

	// A non-finite value silently poisons the whole batch; the existing
	// RelativeObs guards the same way.
	for (float& v : result) {
		if (!std::isfinite(v))
			v = 0.f;
	}

	return result;
}

}  // namespace Dash
```

- [ ] **Step 6: Add the enum value and the name helper**

In `bot/src/env/Obs.h`, add `Predictive,` to the `ObsMode` enum after `Relative,`, and
declare below `ProbeObsSize`:

```cpp
// Name used in CONFIG.json and console output. A two-way ternary used to do
// this job and would silently mislabel any third mode.
const char* ObsModeName(ObsMode mode);
```

In `bot/src/env/Obs.cpp`, add `#include "PredictiveObs.h"` alongside the other obs
includes, add the dispatch case inside `MakeObsBuilder`'s switch:

```cpp
	case ObsMode::Predictive:
		return std::make_unique<PredictiveObs>(maxPlayersPerTeam);
```

and add at the end of the `Dash` namespace:

```cpp
const char* ObsModeName(ObsMode mode) {
	switch (mode) {
	case ObsMode::Default:    return "Default";
	case ObsMode::Advanced:   return "Advanced";
	case ObsMode::Relative:   return "Relative";
	case ObsMode::Predictive: return "Predictive";
	}
	return "Unknown";
}
```

- [ ] **Step 7: Fix the mislabelling ternary and the CLI**

In `bot/src/train/Train.cpp`, find the line reading:

```cpp
			  << (cfg.obs == ObsMode::Advanced ? "Advanced" : "Default")
```

and replace it with:

```cpp
			  << ObsModeName(cfg.obs)
```

In `bot/src/main.cpp`, inside `RunTrain`'s `--obs` handling, add before the error branch:

```cpp
			} else if (modeStr == "predictive" || modeStr == "Predictive") {
				cfg.obs = Dash::ObsMode::Predictive;
```

and update that branch's error message to
`"Unknown obs mode: %s (expected 'default', 'advanced' or 'predictive')\n"`.

In `PrintUsage`, update the two obs lines to read
`"  --obs MODE           Observation builder: default, advanced or predictive "` and
`"  DASH_OBS             Observation builder: default, advanced or predictive "`.

Then check `bot/src/rlbot/DashBot.cpp`'s `BotSettings::FromEnvironment` for its own
`DASH_OBS` string parsing and add the same `predictive` case there, so the deploy path can
select the new mode.

- [ ] **Step 8: Run the tests to verify they pass**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests
```
Expected: PASS, all cases including the 7 new `PredictiveObs` ones.

If "leaves the existing dimensions untouched" fails, `PredictiveObs::BuildObs` is not
calling the base implementation first, or the base is being called with a mutated state.

If "mirrors the prediction block for orange" fails, an `InvertPhys` call is missing — most
likely on `bouncePos`, which is in world space and must be inverted like any other
position.

- [ ] **Step 9: Verify the existing modes are unchanged**

Run:
```bash
./bot/build/DashBot match --m1 t2 --m2 main-p19pool --games 4 --arenas 4 --cpu
```
Expected: the match runs and reports results. This confirms the `ObsMode` enum change did
not disturb loading of existing checkpoints. It must not be run while t1 is training on the
GPU — `--cpu` keeps it off the card.

- [ ] **Step 10: Commit**

```bash
git add bot/src/env/PredictiveObs.h bot/src/env/PredictiveObs.cpp \
        bot/src/env/Obs.h bot/src/env/Obs.cpp bot/src/train/Train.cpp \
        bot/src/main.cpp bot/src/rlbot/DashBot.cpp \
        bot/tests/test_predictiveobs.cpp bot/CMakeLists.txt
git commit -m "feat: add Predictive obs mode with ball-prediction block"
```

---

### Task 5: Performance gate

**This task can veto the feature.** The spec's stated gate is ~15% SPS loss. Measure before
building anything on top of this.

The measurement must not run full training: t1 is live on the GPU, and a concurrent
training probe would both disturb it and produce a meaningless number. This benchmarks the
CPU env-stepping path in isolation, which is where the prediction cost actually lands.

**Files:**
- Create: `bot/src/eval/PredictBench.h`
- Create: `bot/src/eval/PredictBench.cpp`
- Modify: `bot/src/main.cpp` (add the `predict-bench` subcommand)
- Modify: `bot/CMakeLists.txt`

**Interfaces:**
- Consumes: `Dash::MakeObsBuilder`, `Dash::ObsMode`, `Dash::ObsModeName` from Task 4; `Dash::CreateEnv` from `env/Env.h`; `Dash::BallPredictor::SimulationCount()` from Task 2.
- Produces: `int Dash::RunPredictBench(int numArenas, int steps)`, returning an exit code.

- [ ] **Step 1: Write the benchmark header**

Create `bot/src/eval/PredictBench.h`:

```cpp
#pragma once

namespace Dash {

// Times the CPU env-stepping path with and without the prediction block.
//
// Deliberately not a training run: t1 occupies the GPU, and the cost this gate
// is about is CPU ball simulation, which a policy-free loop measures cleanly.
int RunPredictBench(int numArenas, int steps);

}  // namespace Dash
```

- [ ] **Step 2: Write the benchmark implementation**

Create `bot/src/eval/PredictBench.cpp`:

```cpp
#include "PredictBench.h"

#include "../Config.h"
#include "../env/Env.h"
#include "../env/Obs.h"

#include <RLGymCPP/Gamestates/GameState.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

using namespace RLGC;

namespace Dash {

namespace {

struct BenchArena {
	Arena* arena = nullptr;
	std::unique_ptr<ObsBuilder> obs;
	GameState state;
};

// One pass: step every arena by tickSkip, then build every player's obs.
// Mirrors what EnvSet does per env step, minus the policy.
double TimeMode(ObsMode mode, int numArenas, int steps, int tickSkip) {
	std::vector<BenchArena> arenas(numArenas);
	for (int i = 0; i < numArenas; i++) {
		arenas[i].arena = Arena::Create(GameMode::SOCCAR);
		arenas[i].arena->AddCar(Team::BLUE);
		arenas[i].arena->AddCar(Team::ORANGE);
		arenas[i].arena->ResetToRandomKickoff(i);
		arenas[i].obs = MakeObsBuilder(3, mode);
		arenas[i].state = GameState(arenas[i].arena);
		arenas[i].obs->Reset(arenas[i].state);
	}

	const auto start = std::chrono::steady_clock::now();

	for (int s = 0; s < steps; s++) {
		for (auto& a : arenas) {
			a.arena->Step(tickSkip);
			a.state.UpdateFromArena(a.arena,
			                        std::vector<Action>(a.arena->_cars.size()),
			                        nullptr);
			for (const auto& p : a.state.players)
				a.obs->BuildObs(p, a.state);
		}
	}

	const auto end = std::chrono::steady_clock::now();

	for (auto& a : arenas)
		delete a.arena;

	const double seconds =
		std::chrono::duration<double>(end - start).count();
	return (double)numArenas * steps / seconds;  // env steps per second
}

}  // namespace

int RunPredictBench(int numArenas, int steps) {
	const int tickSkip = TrainConfig{}.tickSkip;

	std::printf("Warming up...\n");
	TimeMode(ObsMode::Advanced, numArenas, steps / 4, tickSkip);

	const double advanced = TimeMode(ObsMode::Advanced, numArenas, steps, tickSkip);
	const double predictive = TimeMode(ObsMode::Predictive, numArenas, steps, tickSkip);

	const double lossPct = 100.0 * (1.0 - predictive / advanced);

	std::printf("\n%d arenas, %d steps, tickSkip %d\n", numArenas, steps, tickSkip);
	std::printf("  %-12s %10.0f env-steps/sec\n", "Advanced", advanced);
	std::printf("  %-12s %10.0f env-steps/sec\n", "Predictive", predictive);
	std::printf("  throughput loss: %.1f%%\n", lossPct);
	std::printf("\n  GATE: %s (spec allows ~15%%)\n",
	            lossPct <= 15.0 ? "PASS" : "FAIL");

	return lossPct <= 15.0 ? 0 : 1;
}

}  // namespace Dash
```

- [ ] **Step 3: Wire the subcommand**

In `bot/CMakeLists.txt`, add to the `DashCore` source list after `src/eval/NectoBench.h`:

```cmake
	src/eval/PredictBench.cpp
	src/eval/PredictBench.h
```

In `bot/src/main.cpp`, add `#include "eval/PredictBench.h"` with the other eval includes,
add to `PrintUsage`'s command list:

```
		"  predict-bench    Measure the throughput cost of the prediction obs block\n"
```

and add to `main`'s dispatch chain, before the `necto-selftest` case:

```cpp
	if (command == "predict-bench") {
		int arenas = 64;
		int steps = 400;
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--arenas" && i + 1 < argc)
				arenas = std::atoi(argv[++i]);
			else if (arg == "--steps" && i + 1 < argc)
				steps = std::atoi(argv[++i]);
			else {
				std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
				return EXIT_FAILURE;
			}
		}
		RocketSim::Init(FindCollisionMeshes());
		return Dash::RunPredictBench(arenas, steps);
	}
```

- [ ] **Step 4: Build and run the gate**

Run:
```bash
scripts/build.sh && ./bot/build/DashBot predict-bench --arenas 64 --steps 400
```
Expected: a table and a PASS/FAIL verdict.

Use 64 arenas rather than the training config's 256 so the probe finishes quickly and
leaves CPU headroom for t1, which is still running. The ratio is what matters, not the
absolute number.

- [ ] **Step 5: Record the result and decide**

**If PASS:** record the measured numbers in the spec's "Known risks" section and continue
to Task 6.

**If FAIL:** stop and report to the user before continuing. The tuning levers, in order of
preference:
1. Reduce `SIM_HORIZON_TICKS` from 720 — less wasted simulation past the last sample, at
   the cost of re-simulating more often.
2. Drop the two deepest samples (192, 312), which the spec already flags as the most
   counterfactual, and reduce `SAMPLE_TICKS` to 4 entries. This changes `PREDICT_BLOCK`
   to 18 and the new obs size to 243 — Task 6's constants must follow.
3. Simulate at reduced resolution by stepping 2 ticks at a time and interpolating.

Do not silently pick one; the spec's success criteria depend on the block's contents.

- [ ] **Step 6: Commit**

```bash
git add bot/src/eval/PredictBench.h bot/src/eval/PredictBench.cpp \
        bot/src/main.cpp bot/CMakeLists.txt
git commit -m "feat: add predict-bench to gate the prediction block's cost"
```

---

### Task 6: The migrate-obs checkpoint surgery

**Files:**
- Create: `bot/src/eval/MigrateObs.h`
- Create: `bot/src/eval/MigrateObs.cpp`
- Modify: `bot/src/main.cpp` (add the `migrate-obs` subcommand)
- Modify: `bot/CMakeLists.txt` (sources, plus the GigaLearn private include dir)
- Test: `bot/tests/test_migrateobs.cpp`

**Interfaces:**
- Consumes: `GGL::Model`, `GGL::ModelConfig`, `GGL::PartialModelConfig` from
  `private/GigaLearnCPP/Util/Models.h`.
- Produces:
  - `Dash::MigrateResult { bool ok; std::string message; }`
  - `Dash::MigrateResult Dash::MigrateSharedHead(const std::filesystem::path& srcFolder, const std::filesystem::path& dstFolder, int oldObsSize, int newObsSize, const ModelShape& shape)`
  - `int Dash::RunMigrateObs(const std::filesystem::path& srcRun, const std::filesystem::path& dstRun, int oldObsSize, int newObsSize)` — migrates a whole run folder including `policy_versions/`.

**Why zero-padding is exactly equivalent:** `Models.cpp:17-22` applies LayerNorm *after*
each hidden Linear, never to the input, so the raw obs feeds straight into
`Linear(obsSize, 1024)`. Widening with zero columns changes no activation. This depends on
`standardizeObs` being false — it is (`LearnerConfig.h:48`, never set by `Train.cpp`), which
is why `RUNNING_STATS.json` has no `obs_stat` block. If that ever changes, this whole
approach breaks, because a zero input standardizes to `-mean/std`, not zero.

**Only the shared head takes obsSize.** `PPOLearner.cpp:62-63` sets the policy's and
critic's `numInputs` to `sharedHeadLayers.back()` (512), which the archive confirms.
`POLICY.lt` and `CRITIC.lt` are copied byte-for-byte.

- [ ] **Step 1: Write the failing test**

Create `bot/tests/test_migrateobs.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <eval/MigrateObs.h>
#include <policy/Policy.h>

#include <private/GigaLearnCPP/Util/Models.h>

#include <filesystem>
#include <torch/torch.h>

using namespace Dash;

namespace {

GGL::ModelConfig MakeHeadConfig(int numInputs, const ModelShape& shape) {
	GGL::PartialModelConfig partial = {};
	partial.layerSizes = shape.sharedHeadLayers;
	partial.activationType = shape.activation;
	partial.addLayerNorm = shape.addLayerNorm;
	partial.addOutputLayer = false;

	GGL::ModelConfig config = partial;
	config.numInputs = numInputs;
	config.numOutputs = 0;
	return config;
}

} // namespace

TEST_CASE("MigrateSharedHead preserves the network's output exactly") {
	RG_NO_GRAD;

	// A small shape keeps the test fast; the mechanism is size-independent.
	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const int oldObs = 12;
	const int newObs = 20;

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-test";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	// Build and save a randomly initialized old-width head.
	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObs, shape),
	                   torch::kCPU);
	oldHead.Save(tmp / "src", /*saveOptim=*/false);

	const MigrateResult result = MigrateSharedHead(
		tmp / "src", tmp / "dst", oldObs, newObs, shape);
	REQUIRE_MESSAGE(result.ok, result.message);

	GGL::Model newHead("shared_head", MakeHeadConfig(newObs, shape),
	                   torch::kCPU);
	newHead.Load(tmp / "dst", /*allowNotExist=*/false, /*loadOptim=*/false);

	// The same input, zero-padded, must produce the same output.
	torch::Tensor input = torch::randn({4, oldObs});
	torch::Tensor padded = torch::cat(
		{input, torch::zeros({4, newObs - oldObs})}, 1);

	torch::Tensor before = oldHead.Forward(input, false);
	torch::Tensor after = newHead.Forward(padded, false);

	CHECK(torch::allclose(before, after, 1e-6, 1e-6));

	std::filesystem::remove_all(tmp);
}

TEST_CASE("MigrateSharedHead zeroes only the new input columns") {
	RG_NO_GRAD;

	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const int oldObs = 12;
	const int newObs = 20;

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-cols";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObs, shape),
	                   torch::kCPU);
	oldHead.Save(tmp / "src", /*saveOptim=*/false);

	REQUIRE(MigrateSharedHead(tmp / "src", tmp / "dst", oldObs, newObs,
	                          shape).ok);

	GGL::Model newHead("shared_head", MakeHeadConfig(newObs, shape),
	                   torch::kCPU);
	newHead.Load(tmp / "dst", /*allowNotExist=*/false, /*loadOptim=*/false);

	torch::Tensor oldW = oldHead.seq->parameters()[0];
	torch::Tensor newW = newHead.seq->parameters()[0];

	REQUIRE(newW.size(1) == newObs);
	CHECK(torch::equal(newW.slice(1, 0, oldObs), oldW));
	CHECK(newW.slice(1, oldObs, newObs).abs().sum().item<float>() == 0.f);

	std::filesystem::remove_all(tmp);
}

TEST_CASE("MigrateSharedHead refuses to shrink the input") {
	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-shrink";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	GGL::Model head("shared_head", MakeHeadConfig(20, shape), torch::kCPU);
	head.Save(tmp / "src", /*saveOptim=*/false);

	const MigrateResult result =
		MigrateSharedHead(tmp / "src", tmp / "dst", 20, 12, shape);
	CHECK_FALSE(result.ok);

	std::filesystem::remove_all(tmp);
}
```

- [ ] **Step 2: Expose GigaLearn's private headers to DashCore and add the files**

`GigaLearnCPP` declares `src/private` as a PRIVATE include dir, so `GGL::Model` is not
reachable from `bot/` by default. Add the include path to our own CMakeLists — this is a
local change, not an edit to `external/`.

In `bot/CMakeLists.txt`, in the existing `target_include_directories(DashCore PRIVATE ...)`
block that adds the json path, add:

```cmake
	"${DASH_EXTERNAL}/GigaLearnCPP/GigaLearnCPP/src/private"
```

Change that block from `PRIVATE` to `PUBLIC` so `DashTests` can include
`private/GigaLearnCPP/Util/Models.h` too.

Add to the `DashCore` source list after `src/eval/NectoBench.h`:

```cmake
	src/eval/MigrateObs.cpp
	src/eval/MigrateObs.h
```

and to `DashTests`:

```cmake
	tests/test_migrateobs.cpp
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests
```
Expected: build FAILS with `eval/MigrateObs.h: No such file or directory`.

- [ ] **Step 4: Write the header**

Create `bot/src/eval/MigrateObs.h`:

```cpp
#pragma once

#include "../policy/Policy.h"

#include <filesystem>
#include <string>

namespace Dash {

struct MigrateResult {
	bool ok = false;
	std::string message;
};

// Widens a saved shared head's input layer, zero-filling the new columns.
//
// Exactly behaviour-preserving: LayerNorm sits after each hidden Linear, never
// on the input (Models.cpp:17-22), so the raw obs feeds straight into
// Linear(obsSize, N) and zero columns contribute nothing. This would NOT hold
// if standardizeObs were enabled, since a zero input standardizes to
// -mean/std.
MigrateResult MigrateSharedHead(const std::filesystem::path& srcFolder,
                                const std::filesystem::path& dstFolder,
                                int oldObsSize,
                                int newObsSize,
                                const ModelShape& shape);

// Migrates a whole run folder: the newest checkpoint plus every snapshot under
// policy_versions/, which are the self-play opponent ladder and would fail
// Model::Load's size check otherwise.
int RunMigrateObs(const std::filesystem::path& srcRun,
                  const std::filesystem::path& dstRun,
                  int oldObsSize,
                  int newObsSize);

}  // namespace Dash
```

- [ ] **Step 5: Write the implementation**

Create `bot/src/eval/MigrateObs.cpp`:

```cpp
#include "MigrateObs.h"
#include "Checkpoints.h"

#include <private/GigaLearnCPP/Util/Models.h>

#include <torch/torch.h>

#include <cstdio>

namespace Dash {

namespace {

GGL::ModelConfig MakeHeadConfig(int numInputs, const ModelShape& shape) {
	// Must match PPOLearner::MakeModels exactly, or the loaded parameter
	// shapes will not line up with the saved ones.
	GGL::PartialModelConfig partial = {};
	partial.layerSizes = shape.sharedHeadLayers;
	partial.activationType = shape.activation;
	partial.addLayerNorm = shape.addLayerNorm;
	partial.addOutputLayer = false;

	GGL::ModelConfig config = partial;
	config.numInputs = numInputs;
	config.numOutputs = 0;
	return config;
}

}  // namespace

MigrateResult MigrateSharedHead(const std::filesystem::path& srcFolder,
                                const std::filesystem::path& dstFolder,
                                int oldObsSize,
                                int newObsSize,
                                const ModelShape& shape) {
	RG_NO_GRAD;

	if (newObsSize < oldObsSize)
		return {false, "new obs size is smaller than the old one; this tool "
		               "only widens"};

	if (!std::filesystem::exists(srcFolder / "SHARED_HEAD.lt"))
		return {false, "no SHARED_HEAD.lt in " + srcFolder.string()};

	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObsSize, shape),
	                   torch::kCPU);
	try {
		oldHead.Load(srcFolder, /*allowNotExist=*/false, /*loadOptim=*/false);
	} catch (const std::exception& e) {
		return {false, std::string("failed to load old head: ") + e.what()};
	}

	GGL::Model newHead("shared_head", MakeHeadConfig(newObsSize, shape),
	                   torch::kCPU);

	auto from = oldHead.seq->parameters();
	auto to = newHead.seq->parameters();

	if (from.size() != to.size())
		return {false, "parameter count differs between old and new head"};

	for (size_t i = 0; i < from.size(); i++) {
		if (from[i].sizes() == to[i].sizes()) {
			to[i].copy_(from[i]);
			continue;
		}

		// The only legitimate mismatch is the first Linear's weight, whose
		// input dimension we are widening.
		const bool isFirstWeight =
			i == 0 && from[i].dim() == 2 &&
			from[i].size(0) == to[i].size(0) &&
			from[i].size(1) == oldObsSize && to[i].size(1) == newObsSize;

		if (!isFirstWeight)
			return {false, "unexpected parameter shape mismatch at index " +
			               std::to_string(i)};

		to[i].zero_();
		to[i].slice(1, 0, oldObsSize).copy_(from[i]);
	}

	std::filesystem::create_directories(dstFolder);
	newHead.Save(dstFolder, /*saveOptim=*/false);

	return {true, "migrated " + std::to_string(oldObsSize) + " -> " +
	              std::to_string(newObsSize)};
}

int RunMigrateObs(const std::filesystem::path& srcRun,
                  const std::filesystem::path& dstRun,
                  int oldObsSize,
                  int newObsSize) {
	const ModelShape shape = {};  // t1's shape is the ModelShape default

	const std::filesystem::path latest = FindLatestCheckpoint(srcRun);
	if (latest.empty()) {
		std::fprintf(stderr, "No complete checkpoint under %s\n",
		             srcRun.string().c_str());
		return EXIT_FAILURE;
	}

	// Migrate the head, then carry across everything the head does not own.
	// POLICY and CRITIC take sharedHeadLayers.back() as input, not obsSize
	// (PPOLearner.cpp:62-63), so they need no surgery.
	auto MigrateOne = [&](const std::filesystem::path& src,
	                      const std::filesystem::path& dst) -> bool {
		const MigrateResult r =
			MigrateSharedHead(src, dst, oldObsSize, newObsSize, shape);
		if (!r.ok) {
			std::fprintf(stderr, "  %s: %s\n", src.string().c_str(),
			             r.message.c_str());
			return false;
		}

		for (const auto& entry : std::filesystem::directory_iterator(src)) {
			if (entry.path().filename() == "SHARED_HEAD.lt")
				continue;
			// Optimizer state for the head is intentionally dropped; see the
			// note printed at the end.
			if (entry.path().filename() == "SHARED_HEAD_OPTIM.lt")
				continue;
			std::filesystem::copy(
				entry.path(), dst / entry.path().filename(),
				std::filesystem::copy_options::overwrite_existing);
		}
		return true;
	};

	std::filesystem::create_directories(dstRun);

	std::printf("Migrating newest checkpoint %s\n", latest.string().c_str());
	if (!MigrateOne(latest, dstRun / latest.filename()))
		return EXIT_FAILURE;

	// Run-level files sit beside the numbered checkpoint folders.
	for (const auto& entry : std::filesystem::directory_iterator(srcRun)) {
		if (entry.is_directory())
			continue;
		std::filesystem::copy(
			entry.path(), dstRun / entry.path().filename(),
			std::filesystem::copy_options::overwrite_existing);
	}

	// The self-play opponent ladder. Model::Load hard-checks parameter sizes,
	// so an unmigrated snapshot would abort the run at load time.
	const std::filesystem::path versions = srcRun / "policy_versions";
	if (std::filesystem::exists(versions)) {
		int migrated = 0;
		for (const auto& entry : std::filesystem::directory_iterator(versions)) {
			if (!entry.is_directory())
				continue;
			const auto dst =
				dstRun / "policy_versions" / entry.path().filename();
			if (!MigrateOne(entry.path(), dst))
				return EXIT_FAILURE;
			migrated++;
		}
		std::printf("Migrated %d policy version(s)\n", migrated);
	}

	std::printf(
		"\nDone: %s\n"
		"NOTE: shared-head optimizer state was NOT carried across. Adam's\n"
		"      moments re-estimate within a few hundred iterations; at lr 1e-4\n"
		"      the transient is small. Resume with a reduced --lr for the\n"
		"      first few iterations if you want to be careful.\n",
		dstRun.string().c_str());

	return EXIT_SUCCESS;
}

}  // namespace Dash
```

- [ ] **Step 6: Wire the subcommand**

In `bot/src/main.cpp`, add `#include "eval/MigrateObs.h"`, add to `PrintUsage`'s command
list:

```
		"  migrate-obs      Widen a checkpoint's input layer for a new obs mode\n"
```

and add to `main`'s dispatch chain:

```cpp
	if (command == "migrate-obs") {
		std::filesystem::path src, dst;
		int oldObs = 225, newObs = 249;
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--src" && i + 1 < argc)
				src = argv[++i];
			else if (arg == "--dst" && i + 1 < argc)
				dst = argv[++i];
			else if (arg == "--old-obs" && i + 1 < argc)
				oldObs = std::atoi(argv[++i]);
			else if (arg == "--new-obs" && i + 1 < argc)
				newObs = std::atoi(argv[++i]);
			else {
				std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
				return EXIT_FAILURE;
			}
		}
		if (src.empty() || dst.empty()) {
			std::fprintf(stderr,
			             "Usage: %s migrate-obs --src <run folder> "
			             "--dst <run folder>\n"
			             "       [--old-obs N] [--new-obs N]\n", argv[0]);
			return EXIT_FAILURE;
		}
		if (std::filesystem::exists(dst)) {
			std::fprintf(stderr, "Refusing to overwrite existing %s\n",
			             dst.string().c_str());
			return EXIT_FAILURE;
		}
		return Dash::RunMigrateObs(src, dst, oldObs, newObs);
	}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run:
```bash
scripts/build.sh && ./bot/build/DashTests -tc="MigrateSharedHead*"
```
Expected: PASS, 3 test cases.

If "preserves the network's output exactly" fails by a small margin, check that
`MakeHeadConfig` sets `addOutputLayer = false` — an extra output Linear would be randomly
initialized in the new model and copied in the old, producing a mismatch.

- [ ] **Step 8: Commit**

```bash
git add bot/src/eval/MigrateObs.h bot/src/eval/MigrateObs.cpp \
        bot/src/main.cpp bot/tests/test_migrateobs.cpp bot/CMakeLists.txt
git commit -m "feat: add migrate-obs to widen a checkpoint's input layer"
```

---

### Task 7: Migrate t1 and verify behavioural equivalence

This task runs the real migration and proves it. No new code — it is the acceptance gate
the spec requires before any training resumes.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-25-ball-prediction-obs-design.md` (record results)

**Interfaces:**
- Consumes: the `migrate-obs` and `match` subcommands.
- Produces: `bot/build/checkpoints/t3/`, a migrated copy of t1 ready to train.

- [ ] **Step 1: Confirm the obs sizes actually match the plan's constants**

Run:
```bash
./bot/build/DashBot predict-bench --arenas 1 --steps 1 2>&1 | head -5
python3 - <<'EOF'
import zipfile, pickletools, io
z = zipfile.ZipFile('bot/build/checkpoints/t1/890047828/SHARED_HEAD.lt')
raw = z.read('archive/data.pkl')
s = io.StringIO(); pickletools.dis(raw, s)
# The first BININT2/BININT1 pair after the first tensor is (out, in).
print([l for l in s.getvalue().splitlines() if 'BININT' in l][:6])
EOF
```
Expected: the dump shows `1024` then `225`. If it shows anything else, t1 has advanced or
its shape changed — stop and re-derive `--old-obs` before migrating.

- [ ] **Step 2: Snapshot t1 before touching anything**

t1 is a live run. Copy, never move.

```bash
cp -r bot/build/checkpoints/t1 bot/build/checkpoints/t1-frozen-890M
ls bot/build/checkpoints/t1-frozen-890M
```
Expected: the numbered checkpoint folder, `CONFIG.json`, `CONFIG_HISTORY.json`,
`necto_rating.json` and `policy_versions`.

- [ ] **Step 3: Run the migration**

```bash
./bot/build/DashBot migrate-obs \
  --src bot/build/checkpoints/t1-frozen-890M \
  --dst bot/build/checkpoints/t3 \
  --old-obs 225 --new-obs 249
```
Expected: "Migrating newest checkpoint ...", "Migrated 16 policy version(s)", "Done".

- [ ] **Step 4: Update t3's recorded obs mode**

`CONFIG.json` was copied verbatim and still says `"obs": "Advanced"`. The training run
reads its config from the CLI, not this file, but leaving it wrong would make the run
folder self-contradicting.

```bash
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path('bot/build/checkpoints/t3/CONFIG.json')
cfg = json.loads(p.read_text())
cfg['env']['obs'] = 'Predictive'
p.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
print(cfg['env']['obs'])
EOF
```
Expected: `Predictive`.

- [ ] **Step 5: Run the acceptance test**

The migrated head with zero columns must play identically to the original. Deterministic
actions and a fixed game count make this a real comparison.

```bash
./bot/build/DashBot match \
  --m1 bot/build/checkpoints/t1-frozen-890M/890047828 \
  --m2 bot/build/checkpoints/t3/890047828 \
  --label1 frozen --label2 migrated \
  --games 100 --arenas 32 --deterministic --cpu
```

Expected: **no statistically significant difference**, and goal totals that are equal or
within a goal or two. The two policies are the same function; any residual difference comes
from arena assignment and side ordering, not from the networks.

**If the verdict reports a significant difference, STOP.** That means the padding is not
behaviour-preserving and something in the assumptions is wrong — most likely
`standardizeObs`, the `ModelShape` used to reconstruct the head, or a policy version that
silently failed to migrate. Do not start training.

- [ ] **Step 6: Record the results in the spec**

Add the measured `predict-bench` numbers from Task 5 and this match's verdict to the spec's
"Success criteria" section, replacing the predictions with what actually happened.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-08-25-ball-prediction-obs-design.md
git commit -m "docs: record prediction-block benchmark and migration acceptance results"
```

---

### Task 8: Launch t3

Not code — the runbook for starting the new run, and what to watch.

- [ ] **Step 1: Stop t1**

t1 is at a self-play fixed point and the 2060 cannot usefully run both. Press `q` in its
terminal so it saves cleanly rather than killing it.

Verify the newest checkpoint timestamp stopped advancing:
```bash
ls -la bot/build/checkpoints/t1 | tail -3
```

- [ ] **Step 2: Start t3**

```bash
cd bot/build && ./DashBot train \
  --label t3 --obs predictive \
  --games 256 --self-play --necto
```

Expected in the header: `Observation size: 249 (mode=Predictive, maxPlayersPerTeam=3)` and
`NOTE: resuming from existing checkpoints in checkpoints/t3.`

**If it says `Observation size: 225`,** `--obs predictive` did not take effect — check the
`main.cpp` parsing from Task 4.

**If it starts from zero timesteps,** it did not find the migrated checkpoint — check that
`bot/build/checkpoints/t3/` contains the numbered folder and `RUNNING_STATS.json`.

- [ ] **Step 3: Confirm the graft is behaving over the first iterations**

Watch the first few reported iterations. Policy entropy and mean reward should continue
roughly where t1 left off, not collapse and re-climb. A collapse means the migration was
not behaviour-preserving after all, and Task 7's acceptance test missed it.

- [ ] **Step 4: Register the frozen benchmark opponent**

After ~20M steps, check whether the network is using the new columns at all, and how t3
compares to the control:

```bash
cd bot/build && ./DashBot match \
  --m1 ../../bot/build/checkpoints/t1-frozen-890M/890047828 \
  --m2 "$(ls -d checkpoints/t3/*/ | sort -n | tail -1)" \
  --label1 frozen-t1 --label2 t3 \
  --games 200 --arenas 32 --deterministic --cpu
```

This is the spec's success criterion 4. Run it periodically rather than reading Elo drift
alone.

- [ ] **Step 5: Watch for the passivity failure mode**

The spec's main behavioural risk is that seeing the landing spot teaches "drive to the
bounce point and wait". Elo will not show this early. Watch a few episodes directly:

```bash
cd bot/build && ./DashBot spectate --follow t3 --spawns training
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| 24-dim block, 18 samples + 6 events | 4 |
| Geometric schedule 0.15-2.60s | 2 (constants), 4 (encoding) |
| Car-local encoding matching AdvancedObs | 4 |
| No per-sample velocity | 4 (block is 6x3+6) |
| Rejected height feature | Not implemented — correct |
| Car-less arena, not RLBot's predictor | 2 |
| Touch-invalidated caching | 2 |
| Both players share one trajectory | 2 (cache), 4 (single `Get` per build) |
| `ObsBuilder::Reset` clears cache | 2, 4 |
| Cost measured before committing | 5 |
| `ObsMode::Predictive`, existing modes intact | 4 |
| `Train.cpp:426` ternary fix | 4 |
| Zero-pad first Linear | 6 |
| POLICY/CRITIC untouched | 6 |
| Optimizer state | 6 — **deviation**: dropped rather than padded, using the spec's stated fallback. Padding Adam state through `Optimizer::state()`'s `void*`-keyed map is materially fiddlier than the rest of the tool, and the spec explicitly sanctions the reset with reduced LR. Flagged to the user below. |
| `RUNNING_STATS.json` needs nothing | 6 (copied verbatim), 7 |
| `policy_versions/` migrated | 6, 7 |
| Necto unaffected | No task needed |
| Acceptance test | 7 |
| Fork to t3, freeze t1 | 7, 8 |
| Success criteria 1-4 | 5, 7, 8 |

**Placeholder scan:** No TBDs, no "add error handling", no "similar to Task N". Every code
step carries the actual code.

**Type consistency:** `BallTrajectory` fields (`pos`, `vel`, `startTick`, `bounceTick`,
`bouncePos`, `goalTick`, `goalTeam`) are declared in Task 2 and used with those exact names
in Tasks 3 and 4. `BallPredictor::SAMPLE_TICKS`/`NUM_SAMPLES`/`SIM_HORIZON_TICKS` declared
in Task 2, used in Tasks 3-5. `PredictiveObs::PREDICT_BLOCK` declared in Task 4, used in
Task 4's tests. `MigrateResult`/`MigrateSharedHead`/`RunMigrateObs` declared in Task 6's
header, used in Task 6's tests and `main.cpp`. `ObsModeName` declared in Task 4, used in
Task 4's `Train.cpp` edit.

## Deviation from the spec, for the user's decision

**Task 6 drops the shared-head optimizer state instead of padding it.** The spec's step 5
preferred padding Adam's `exp_avg`/`exp_avg_sq` and named the reset as fallback. Padding
requires reaching into `Optimizer::state()`, which is keyed by
`param.unsafeGetTensorImpl()` and holds `unique_ptr<OptimizerParamState>` — doable, but it
is the one part of the tool where a subtle mistake would corrupt training silently rather
than failing loudly, and the payoff is a few hundred iterations of Adam re-estimation at
`lr = 1e-4`.

If you would rather have the padding, say so and it becomes an extra task before Task 7.
