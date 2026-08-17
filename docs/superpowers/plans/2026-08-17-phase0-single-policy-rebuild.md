# Phase 0: Single-Policy 1v1 Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the two-policy multi-size bot into a lean single-policy 1v1 bot with trustworthy metrics, tests, and evaluation tooling, ready for long unattended training runs.

**Architecture:** One PPO policy trained by GigaLearn on RocketSim 1v1 games, kickoffs included via curriculum. A static `HiveCore` library shared by the bot binary and a doctest test binary. New instrumentation (per-term reward shares, per-scenario outcomes) reads GigaLearn's public `Learner::envSet`. Evaluation adds a `verify` parity subcommand, an `eval` head-to-head subcommand, and RLBot match scripting.

**Tech Stack:** C++20, CMake, GigaLearnCPP/RLGymCPP/RocketSim (in `external/GigaLearnCPP-Leak`), RLBot v5 cpp-interface, libtorch, doctest (vendored single header).

**Spec:** `docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`

## Global Constraints

- Indent with tabs, matching GigaLearn/RLGymCPP style. Our code lives in namespace `Hive`.
- Comments: only verified *why*-comments. When touching a file, delete or rewrite inherited commentary; do not add comments narrating what code does.
- Never modify `external/` (two documented local patches already exist; leave them).
- Build with `scripts/build.sh`. Run tests from the build dir: `cd bot/build && ./HiveTests` (collision meshes resolve relatively there).
- This phase fixes: `maxPlayersPerTeam = 1`, `tickSkip = 8`, `actionDelay = 7`, `DefaultAction` parser. The obs-width change invalidates old checkpoints — that is expected; none are worth keeping.
- Working tree contains uncommitted baseline changes (deleted docs/, trimmed Config.h/Rewards.cpp). Task 1 Step 1 commits them first so every later diff is clean.
- Commit after every task (and at marked steps). Message style matches repo history: imperative, no prefix tags.
- `GGL::Learner` public members used here: `envSet` (type `RLGC::EnvSet*`), `totalTimesteps`. `RLGC::EnvSet` public members used: `state.lastRewards[arenaIdx][rewardIdx]` (unweighted sampled per-term rewards; populated because `LearnerConfig.addRewardsToMetrics` defaults true), `state.terminals[arenaIdx]`, `stateSetters[arenaIdx]`, `rewards[arenaIdx]`. Learner loop order per step: reset-terminal-arenas → step → **stepCallback** — so inside the callback, a flagged terminal arena has NOT yet been reset and its state setter still describes the episode that just ended.

---

### Task 1: Commit baseline; split HiveCore; add doctest harness

**Files:**
- Modify: `bot/CMakeLists.txt` (target restructure)
- Modify: `scripts/build.sh` (build both targets)
- Create: `bot/tests/doctest/doctest.h` (vendored)
- Create: `bot/tests/TestCommon.h`
- Create: `bot/tests/test_main.cpp`
- Create: `bot/tests/test_harness.cpp`

**Interfaces:**
- Produces: static lib target `HiveCore` (all of `bot/src` except `main.cpp`; PUBLIC include dir `src`; PUBLIC links `GigaLearnCPP`, `RLBotCPP-static`, NCCL/NVSHMEM paths). Test binary `HiveTests` in `bot/build/`. Helper `Hive::Test::EnsureRocketSim()` — call at the top of any test that creates an `Arena` or builds an obs.
- Consumes: nothing.

- [ ] **Step 1: Commit the baseline working tree**

```bash
cd /home/edwardb/Documents/RLBotDev
git add -A && git commit -m "Strip AI-written docs and trim config to working baseline"
```

- [ ] **Step 2: Vendor doctest**

```bash
mkdir -p bot/tests/doctest
curl -fsSL https://raw.githubusercontent.com/doctest/doctest/v2.4.12/doctest/doctest.h -o bot/tests/doctest/doctest.h
```

- [ ] **Step 3: Restructure CMake targets**

In `bot/CMakeLists.txt`, replace the `add_executable(HivemindBot ...)` block (currently lists every source) with:

```cmake
add_library(HiveCore STATIC
	src/env/Env.cpp
	src/env/Env.h
	src/env/Obs.cpp
	src/env/Obs.h
	src/env/Rewards.cpp
	src/env/Rewards.h
	src/env/StateSetters.cpp
	src/env/StateSetters.h
	src/env/Terminal.h

	src/policy/PolicySet.cpp
	src/policy/PolicySet.h
	src/policy/Regime.cpp
	src/policy/Regime.h

	src/rlbot/HivemindBot.cpp
	src/rlbot/HivemindBot.h
	src/rlbot/PacketConvert.cpp
	src/rlbot/PacketConvert.h

	src/train/Train.cpp
	src/train/Train.h

	src/Config.h
)

target_include_directories(HiveCore PUBLIC src)
target_link_libraries(HiveCore PUBLIC
	GigaLearnCPP
	RLBotCPP-static
)

add_executable(HivemindBot src/main.cpp)
target_link_libraries(HivemindBot PRIVATE HiveCore)

add_executable(HiveTests
	tests/test_main.cpp
	tests/test_harness.cpp
)
target_link_libraries(HiveTests PRIVATE HiveCore)
target_include_directories(HiveTests PRIVATE tests)
```

Move the NCCL/NVSHMEM `foreach` loop so it links into `HiveCore` with `PUBLIC` instead of `HivemindBot` `PRIVATE`. Apply the existing RPATH + `RUNTIME_OUTPUT_DIRECTORY` `set_target_properties` calls to **both** `HivemindBot` and `HiveTests`. Keep `target_include_directories(HivemindBot PRIVATE src)` removed (it now comes from HiveCore). Keep the collision-mesh copy and metrics-receiver custom targets as they are, but change `add_dependencies(HivemindBot hive-metrics-receiver)` to keep HivemindBot only.

- [ ] **Step 4: Write the harness smoke test**

`bot/tests/test_main.cpp`:

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
```

`bot/tests/TestCommon.h`:

```cpp
#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <cstdlib>

namespace Hive::Test {

// RocketSim asserts if initialized twice and Arena creation asserts if never
// initialized; every test that touches an Arena funnels through here.
inline void EnsureRocketSim() {
	static bool done = false;
	if (!done) {
		const char* env = std::getenv("HIVE_COLLISION_MESHES");
		RocketSim::Init(env ? env : "collision_meshes");
		done = true;
	}
}

} // namespace Hive::Test
```

`bot/tests/test_harness.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

TEST_CASE("harness runs and RocketSim initializes") {
	CHECK(1 + 1 == 2);
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	CHECK(arena != nullptr);
	delete arena;
}
```

(If `Arena`/`GameMode` are not visible unqualified, qualify with the names `Obs.cpp` uses — it creates `Arena::Create(GameMode::SOCCAR)` with `using namespace RLGC;` — mirror that.)

- [ ] **Step 5: Update build.sh to build both targets**

In `scripts/build.sh`, change the build line to:

```bash
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target HivemindBot HiveTests
```

- [ ] **Step 6: Build and run the test, expect PASS**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

Expected: `1 test cases: 1 passed`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Split HiveCore library, add doctest test harness"
```

---

### Task 2: Single-policy restructure — config, env, CLI

**Files:**
- Modify: `bot/src/Config.h`
- Modify: `bot/src/env/Env.h`, `bot/src/env/Env.cpp`
- Modify: `bot/src/env/Rewards.cpp` (kickoff builder removal only)
- Modify: `bot/src/env/Rewards.h`
- Modify: `bot/src/env/Terminal.h`
- Modify: `bot/src/main.cpp`
- Modify: `bot/src/train/Train.cpp`
- Modify: `scripts/train.sh`, `scripts/watch.sh`

**Interfaces:**
- Produces: `Hive::TrainConfig` without `target`/`teamSizes`/`kickoffRewards`/`kickoffTimeoutSeconds`; `maxPlayersPerTeam = 1`; `tsPerItr = 100'000`; `RunName()` returns `"main"` or `"main-<label>"`. `CreateEnv(int index, const TrainConfig&)` unchanged signature, always builds 1v1. CLI commands: `train`, `play`.
- Consumes: Task 1's build layout.

- [ ] **Step 1: Trim Config.h**

In `bot/src/Config.h`:
- Delete `enum class TrainTarget`, `struct TeamSizeMix`, `struct KickoffRewardWeights`.
- In `CurriculumWeights`: delete the `groundDribble` member (spec D4: never a dribble curriculum entry). Keep `airDribble`, `flipReset`, `demo` at `0.f`.
- In `TrainConfig`: delete `target`, `teamSizes`, `kickoffRewards`, `kickoffTimeoutSeconds`; change `maxPlayersPerTeam` default from `3` to `1`; change `tsPerItr` from `50'000` to `100'000` (spec: 1v1 steps are ~3x cheaper, larger batches; `miniBatchSize` stays `25'000` — it is the VRAM knob).
- Replace `TargetName()`/`RunName()` with:

```cpp
	std::string RunName() const {
		return runLabel.empty() ? std::string("main") : "main-" + runLabel;
	}
```

- Trim the big banner comments per the comment policy; keep short why-notes on `maxPlayersPerTeam` (obs width invalidates checkpoints), `actionDelay` (must match deployment), `runLabel` (a second unlabeled run silently resumes the first).

- [ ] **Step 2: Simplify Env to 1v1**

In `bot/src/env/Env.h`: delete `struct TeamSizes` and `PickTeamSizes`. In `bot/src/env/Env.cpp`:
- Delete `PickTeamSizes` entirely.
- In `BuildGeneralCurriculum`: remove the `GroundDribbleState` line.
- Replace the body of `CreateEnv` so it always creates one car per team and always builds the general env:

```cpp
EnvCreateResult CreateEnv(int index, const TrainConfig& cfg) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.arena = arena;
	result.actionParser = new DefaultAction();
	result.obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam).release();
	result.stateSetter = BuildGeneralCurriculum(cfg.curriculum);
	result.rewards = BuildGeneralRewards(cfg);
	result.terminalConditions = {
		new NoTouchCondition(cfg.noTouchTimeoutSeconds),
		new GoalScoreCondition(),
	};
	return result;
}
```

Remove now-unused includes (`FuzzedKickoffState.h` stays — the curriculum still uses it).

- [ ] **Step 3: Remove the kickoff reward builder and dead terminal conditions**

- `bot/src/env/Rewards.h`: delete the `BuildKickoffRewards` declaration.
- `bot/src/env/Rewards.cpp`: delete the `BuildKickoffRewards` definition.
- `bot/src/env/Terminal.h`: delete `FirstTouchCondition` and `TimeoutCondition` (both were kickoff-target-only; resurrect from git if a hard episode cap is ever wanted).

- [ ] **Step 4: Collapse the CLI to `train` / `play`**

In `bot/src/main.cpp`:
- `RunTrain` loses its `TrainTarget` parameter and the `cfg.target = target;` line.
- In `main`, replace the two `train-general`/`train-kickoff` branches with one:

```cpp
	if (command == "train")
		return RunTrain(argc, argv);
```

- Update `PrintUsage`: commands are `train` and `play`; delete the `HIVE_KICKOFF_MODEL` line from the environment help (fully removed in Task 3).

In `bot/src/train/Train.cpp`: the config plumbing keeps working (fields it reads still exist); just delete the `Self-play:`/`Run:` startup print's use of anything removed — none is removed there — and leave the rest for Task 7.

- [ ] **Step 5: Update the wrapper scripts**

`scripts/train.sh` and `scripts/watch.sh`: remove the `general|kickoff` case blocks; the scripts no longer take a target argument. Replace with:

```bash
cd "$BUILD_DIR"
exec "$BIN" train "$@"          # train.sh
exec "$BIN" train --render "$@" # watch.sh
```

(Keep each script's existing preamble, build check, and comments-trimmed header. Usage lines become `scripts/train.sh [extra args...]`.)

- [ ] **Step 6: Build, run tests, smoke-train**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
./HivemindBot train --max-steps 200000 --games 16 --no-metrics --label smoke-task2
```

Expected: obs size prints a value **smaller than 165** (1v1 width), run name `main-smoke-task2`, reaches the step budget, saves, exits 0. Delete the smoke checkpoints afterwards: `rm -rf checkpoints/main-smoke-task2`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Collapse to single policy: 1v1 env, one train target"
```

---

### Task 3: Policy wrapper, PlayPhase extraction, deployment updates

**Files:**
- Create: `bot/src/policy/Policy.h`, `bot/src/policy/Policy.cpp`
- Create: `bot/src/env/PlayPhase.h`, `bot/src/env/PlayPhase.cpp`
- Delete: `bot/src/policy/PolicySet.h`, `bot/src/policy/PolicySet.cpp`, `bot/src/policy/Regime.h`, `bot/src/policy/Regime.cpp`
- Modify: `bot/src/rlbot/HivemindBot.h`, `bot/src/rlbot/HivemindBot.cpp`
- Modify: `bot/src/rlbot/PacketConvert.h`, `bot/src/rlbot/PacketConvert.cpp` (remove `IsKickoffPhase`)
- Modify: `bot/src/Config.h` (include swap), `bot/src/train/Train.cpp` (include swap), `bot/CMakeLists.txt` (source list)
- Modify: `bot/rlbot-config/run.sh`, `bot/rlbot-config/bot.toml`

**Interfaces:**
- Produces:
  - `Hive::ModelShape` now lives in `bot/src/policy/Policy.h` (same members: `sharedHeadLayers`, `policyLayers`, `activation`, `addLayerNorm`).
  - `class Hive::Policy` — `Policy(RLGC::ObsBuilder*, int obsSize, RLGC::ActionParser*, const ModelShape&, bool useGPU)`; `void Load(const std::filesystem::path&)`; `bool Loaded() const`; `std::vector<RLGC::Action> InferBatch(const std::vector<RLGC::Player>&, const std::vector<RLGC::GameState>&, bool deterministic, float temperature = 1.f)`.
  - `bot/src/env/PlayPhase.h` — `enum class PlayPhase`, `PLAY_PHASE_COUNT`, `PlayPhaseName(PlayPhase)`, `struct PhaseThresholds`, `ClassifyPhase(const RLGC::Player&, const RLGC::GameState&, const PhaseThresholds& = {})`, `struct PhaseCounts` — all moved verbatim from `Regime.h`/`Regime.cpp` (the `GroundDribble` phase label **stays**: it is a metric, not a curriculum entry).
- Consumes: Task 2's config.

- [ ] **Step 1: Extract PlayPhase**

Create `bot/src/env/PlayPhase.h` and `.cpp` by moving from `Regime.h` lines 110–153 and `Regime.cpp` lines 17–27 and 81–114: the `PlayPhase` enum, `PLAY_PHASE_COUNT`, `PlayPhaseName`, `PhaseThresholds`, `ClassifyPhase` (with its `OwnGoalRelY` helper), `PhaseCounts`. Keep the one comment explaining these are metrics-only labels. Update `bot/src/train/Train.cpp` to include `../env/PlayPhase.h` instead of `../policy/Regime.h`.

- [ ] **Step 2: Write Policy as a cut-down PolicySet**

`bot/src/policy/Policy.h`:

```cpp
#pragma once

#include <GigaLearnCPP/Util/InferUnit.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace Hive {

// Shape of the policy network. Single source of truth for training and
// deployment: both sides default-construct this struct. Changing it
// invalidates every existing checkpoint.
struct ModelShape {
	std::vector<int> sharedHeadLayers = {512, 512};
	std::vector<int> policyLayers = {512, 512, 512};
	GGL::ModelActivationType activation = GGL::ModelActivationType::RELU;
	bool addLayerNorm = true;
};

class Policy {
public:
	// obsBuilder and actionParser are borrowed; the caller keeps them alive.
	Policy(RLGC::ObsBuilder* obsBuilder,
	       int obsSize,
	       RLGC::ActionParser* actionParser,
	       const ModelShape& shape,
	       bool useGPU);

	Policy(const Policy&) = delete;
	Policy& operator=(const Policy&) = delete;

	// Throws std::runtime_error on a missing/invalid checkpoint folder.
	void Load(const std::filesystem::path& checkpointFolder);
	bool Loaded() const { return unit != nullptr; }

	std::vector<RLGC::Action> InferBatch(const std::vector<RLGC::Player>& players,
	                                     const std::vector<RLGC::GameState>& states,
	                                     bool deterministic,
	                                     float temperature = 1.f);

private:
	RLGC::ObsBuilder* obsBuilder;
	int obsSize;
	RLGC::ActionParser* actionParser;
	ModelShape shape;
	bool useGPU;
	std::unique_ptr<GGL::InferUnit> unit;
};

} // namespace Hive
```

`bot/src/policy/Policy.cpp`: port from `PolicySet.cpp` — `Load` keeps the checkpoint-folder existence check and the `.lt` file scan with its actionable error messages (update the message prefix to `"Policy: "` and the example path to `checkpoints/main/50000000`), then builds the `InferUnit` exactly as `PolicySet::Make` did. `InferBatch` becomes a straight pass-through:

```cpp
std::vector<Action> Policy::InferBatch(const std::vector<Player>& players,
                                       const std::vector<GameState>& states,
                                       bool deterministic,
                                       float temperature) {
	if (players.size() != states.size())
		throw std::runtime_error("Policy::InferBatch(): mismatched input sizes");
	if (!unit)
		throw std::runtime_error("Policy::InferBatch(): no model loaded");
	if (players.empty())
		return {};
	return unit->BatchInferActions(players, states, deterministic, temperature);
}
```

Delete `PolicySet.h/.cpp` and `Regime.h/.cpp`. Update `bot/src/Config.h` to include `policy/Policy.h` instead of `policy/PolicySet.h`. Update `bot/CMakeLists.txt`'s HiveCore source list (remove PolicySet/Regime, add Policy and PlayPhase files).

- [ ] **Step 3: Update the RLBot client**

`bot/src/rlbot/HivemindBot.h`:
- `BotSettings`: delete `kickoffModel`; rename `generalModel` to `model` (env var becomes `HIVE_MODEL`); change `maxPlayersPerTeam` default from `3` to `1`.
- `SharedContext`: replace `std::unique_ptr<PolicySet> policies` with `std::unique_ptr<Policy> policy`; include `../policy/Policy.h`.

`bot/src/rlbot/HivemindBot.cpp`:
- `FromEnvironment()`: read `HIVE_MODEL` (error message updated: `"HIVE_MODEL is not set..."`), drop the kickoff block, default `HIVE_MAX_PLAYERS_PER_TEAM` to `1`.
- `SharedContext::Initialize`: construct `Policy`, call `policy->Load(settings.model)`, drop the kickoff prints.
- `update()`: delete the `Regime` computation and the `regimes` vector; call `Context().policy->InferBatch(players, states, settings.deterministic, settings.temperature)`. Everything else (tick accounting, cadence, ControllerState output) is unchanged.

`bot/src/rlbot/PacketConvert.h/.cpp`: delete `IsKickoffPhase` (its only caller was the regime split).

`bot/src/main.cpp`: rename every `HIVE_GENERAL_MODEL` mention to `HIVE_MODEL` (the `PrintUsage` environment section and `RunPlay`'s "run by hand" hint).

- [ ] **Step 4: Update deployment config**

`bot/rlbot-config/run.sh`:
- Model discovery becomes:

```bash
: "${HIVE_MODEL:=$(ls -d "$BUILD_DIR"/checkpoints/main*/*/ 2>/dev/null | sort -V | tail -1)}"

if [[ -z "${HIVE_MODEL:-}" || ! -d "${HIVE_MODEL}" ]]; then
	echo "No model found. Train one first, or set HIVE_MODEL to a checkpoint folder." >&2
	exit 1
fi
export HIVE_MODEL
```

- Delete every `HIVE_KICKOFF_MODEL` line. Change `HIVE_MAX_PLAYERS_PER_TEAM` default from `3` to `1`.

`bot/rlbot-config/bot.toml`: description becomes `"Single-policy RL bot trained with GigaLearn on RocketSim."`; delete the kickoff fun_fact (replace with anything true, e.g. `fun_fact = "Trained entirely in RocketSim."`); tags `["1v1"]`.

- [ ] **Step 5: Build, test, smoke both paths**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
./HivemindBot train --max-steps 200000 --games 16 --no-metrics --label smoke-task3
HIVE_MODEL=$(ls -d checkpoints/main-smoke-task3/*/ | tail -1) RLBOT_AGENT_ID=hivemind/bot ./HivemindBot play
```

Expected: train exits 0; `play` loads the model, prints the settings line, then fails to connect to `127.0.0.1:23234` (no RLBot server) — **connection failure is the expected outcome; a model-load error is not.** Clean up `checkpoints/main-smoke-task3`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Replace PolicySet/Regime with single Policy, extract PlayPhase"
```

---

### Task 4: PlayPhase unit tests

**Files:**
- Create: `bot/tests/test_playphase.cpp`
- Modify: `bot/CMakeLists.txt` (add to HiveTests)

**Interfaces:**
- Consumes: `Hive::ClassifyPhase`, `Hive::PlayPhase`, `Hive::PhaseThresholds` from `env/PlayPhase.h` (Task 3).

- [ ] **Step 1: Write failing-by-absence tests**

`bot/tests/test_playphase.cpp` (default thresholds: airborneZ 200, aerialBallZ 500, airDribbleBallZ 400, ballNearDist 350, dribbleDist 200, dribbleBallZMax 300, defendThirdY -1700):

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/PlayPhase.h>

using namespace Hive;

static RLGC::Player MakePlayer(RLGC::Vec pos, bool onGround, RLGC::Team team = RLGC::Team::BLUE) {
	RLGC::Player p = {};
	p.pos = pos;
	p.isOnGround = onGround;
	p.team = team;
	return p;
}

static RLGC::GameState MakeState(RLGC::Vec ballPos) {
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
	auto blue = MakePlayer({0, 0, 17}, true, RLGC::Team::BLUE);
	auto orange = MakePlayer({0, 0, 17}, true, RLGC::Team::ORANGE);
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
```

- [ ] **Step 2: Add to CMake, build, run**

Add `tests/test_playphase.cpp` to the `HiveTests` source list.

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

Expected: all PASS. If a case fails, the classification logic and the test disagree — read `ClassifyPhase` and decide which is wrong; fix the code only if its behavior contradicts its intent (these thresholds are metric labels, not gameplay).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add PlayPhase classification tests"
```

---

### Task 5: Named reward specs + reward tests

**Files:**
- Modify: `bot/src/env/Rewards.h`, `bot/src/env/Rewards.cpp`
- Create: `bot/tests/test_rewards.cpp`
- Modify: `bot/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
struct RewardSpec {
	std::string name;   // metric label, e.g. "StrongTouch"
	float weight;
	std::function<RLGC::Reward*()> make;
};
std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg);
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg); // materializes the specs, same order
```

Order of specs == order of `envSet->rewards[arena]` == order of `envSet->state.lastRewards[arena]` — Task 7 depends on this.
- Consumes: `TrainConfig` (Task 2).

- [ ] **Step 1: Write failing tests**

`bot/tests/test_rewards.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <Config.h>
#include <env/Rewards.h>

#include <RLGymCPP/CommonValues.h>

using namespace Hive;

TEST_CASE("specs and built rewards agree in count and weight") {
	TrainConfig cfg = {};
	auto specs = GeneralRewardSpecs(cfg);
	auto built = BuildGeneralRewards(cfg);

	REQUIRE(specs.size() == built.size());
	REQUIRE(specs.size() >= 4);
	for (size_t i = 0; i < specs.size(); i++) {
		CHECK(specs[i].weight == built[i].weight);
		CHECK(!specs[i].name.empty());
		CHECK(built[i].reward != nullptr);
	}
	for (auto& wr : built)
		delete wr.reward;
}

TEST_CASE("spec names are unique") {
	auto specs = GeneralRewardSpecs(TrainConfig{});
	for (size_t i = 0; i < specs.size(); i++)
		for (size_t j = i + 1; j < specs.size(); j++)
			CHECK(specs[i].name != specs[j].name);
}

TEST_CASE("TouchHeightReward pays zero for no touch and for ground touches") {
	TouchHeightReward r(1500.f);
	RLGC::Player p = {};
	RLGC::GameState s = {};
	s.ball.pos = {0, 0, 1000};

	p.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	p.ballTouchedStep = true;
	s.ball.pos = {0, 0, RLGC::CommonValues::BALL_RADIUS};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("TouchHeightReward scales with height and saturates at 1") {
	TouchHeightReward r(1500.f);
	RLGC::Player p = {};
	p.ballTouchedStep = true;
	RLGC::GameState s = {};

	s.ball.pos = {0, 0, RLGC::CommonValues::BALL_RADIUS + 750.f};
	const float mid = r.GetReward(p, s, false);
	CHECK(mid == doctest::Approx(0.5f));

	s.ball.pos = {0, 0, 3000.f};
	CHECK(r.GetReward(p, s, false) == 1.f);
}
```

- [ ] **Step 2: Run to verify failure**

Add `tests/test_rewards.cpp` to CMake, build. Expected: **compile failure** — `GeneralRewardSpecs` and `RewardSpec` do not exist yet.

- [ ] **Step 3: Implement the spec refactor**

`bot/src/env/Rewards.h`: add `#include <functional>`, the `RewardSpec` struct, and the `GeneralRewardSpecs` declaration (see Interfaces). `bot/src/env/Rewards.cpp`:

```cpp
std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardWeights& w = cfg.rewards;

	if (cfg.rewardPhase != RewardPhase::Foundations) {
		throw std::runtime_error(
			"GeneralRewardSpecs(): only RewardPhase::Foundations is designed. "
			"Later phases are derived from the telemetry of the run before them.");
	}

	return {
		{"VelPlayerToBall", w.velPlayerToBall, [] { return new VelocityPlayerToBallReward(); }},
		{"StrongTouch", w.strongTouch,
		 [] { return (Reward*)new ZeroSumReward(new StrongTouchReward(TOUCH_MIN_KPH, TOUCH_MAX_KPH), 0.2f); }},
		{"VelBallToGoal", w.velBallToGoal,
		 [] { return (Reward*)new ZeroSumReward(new VelocityBallToGoalReward(), 0.3f); }},
		{"Goal", w.goal, [] { return new GoalReward(); }},
		{"PickupBoost", w.pickupBoost, [] { return new PickupBoostReward(); }},
		{"FaceBall", w.faceBall, [] { return new FaceBallReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}
```

(Adjust lambda return-type casts as the compiler requires; all lambdas must produce `RLGC::Reward*`.)

- [ ] **Step 4: Build and run tests, expect PASS**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Name reward terms via RewardSpec, test reward invariants"
```

---

### Task 6: State setter validity tests

**Files:**
- Create: `bot/tests/test_statesetters.cpp`
- Modify: `bot/CMakeLists.txt`

**Interfaces:**
- Consumes: the setter classes in `env/StateSetters.h` (their public min/max members are the claims under test) and `Hive::Test::EnsureRocketSim()`.

- [ ] **Step 1: Write the invariant tests**

`bot/tests/test_statesetters.cpp`. Field bounds from `RLGC::CommonValues`: side wall x ≈ 4096, back wall y ≈ 5120, ceiling z ≈ 2044 (use the named constants if present — check `CommonValues.h` — else these literals). Every test: create a 1v1 SOCCAR arena, run the setter's `ResetArena` 100 times, assert invariants **every** iteration.

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/StateSetters.h>

#include <RLGymCPP/CommonValues.h>

using namespace Hive;

namespace {

struct ArenaFixture {
	Arena* arena;
	ArenaFixture() {
		Hive::Test::EnsureRocketSim();
		arena = Arena::Create(GameMode::SOCCAR);
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}
	~ArenaFixture() { delete arena; }
};

bool InField(const Vec& p, float margin = 0.f) {
	return std::abs(p.x) < 4096.f + margin &&
	       std::abs(p.y) < 5120.f + 880.f + margin && // goals recess past the back wall
	       p.z > -1.f && p.z < 2044.f + margin;
}

template <typename Setter>
void CheckBasicInvariants(Setter& setter, int iterations = 100) {
	ArenaFixture f;
	for (int i = 0; i < iterations; i++) {
		setter.ResetArena(f.arena);
		CHECK(InField(f.arena->ball->GetState().pos));
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			CHECK(InField(cs.pos));
			CHECK(cs.boost >= 0.f);
			CHECK(cs.boost <= 100.f);
		}
	}
}

} // namespace

TEST_CASE("NeutralPlayState basic invariants") { NeutralPlayState s; CheckBasicInvariants(s); }
TEST_CASE("BallContactState basic invariants") { BallContactState s; CheckBasicInvariants(s); }
TEST_CASE("DefendState basic invariants")      { DefendState s; CheckBasicInvariants(s); }
TEST_CASE("RecoverState basic invariants")     { RecoverState s; CheckBasicInvariants(s); }
TEST_CASE("AerialState basic invariants")      { AerialState s; CheckBasicInvariants(s); }
TEST_CASE("AirDribbleState basic invariants")  { AirDribbleState s; CheckBasicInvariants(s); }
TEST_CASE("FlipResetState basic invariants")   { FlipResetState s; CheckBasicInvariants(s); }
TEST_CASE("DemoState basic invariants")        { DemoState s; CheckBasicInvariants(s); }

TEST_CASE("AerialState puts the ball in its configured height band with cars grounded") {
	AerialState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		CHECK(ball.pos.z >= s.minBallZ - 1.f);
		CHECK(ball.pos.z <= s.maxBallZ + 1.f);
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			CHECK(cs.pos.z < 100.f);
			CHECK(cs.boost >= s.minBoost - 1.f);
		}
	}
}

TEST_CASE("BallContactState spawns a car within its configured distance of the ball") {
	BallContactState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		float best = 1e9f;
		for (Car* car : f.arena->_cars)
			best = std::min(best, (car->GetState().pos - ball.pos).Length());
		CHECK(best <= s.maxDist + 100.f);
	}
}

TEST_CASE("FlipResetState puts the car airborne below the ball") {
	FlipResetState s;
	ArenaFixture f;
	for (int i = 0; i < 100; i++) {
		s.ResetArena(f.arena);
		auto ball = f.arena->ball->GetState();
		CHECK(ball.pos.z >= s.minBallZ - 1.f);
		bool anyAirborneBelow = false;
		for (Car* car : f.arena->_cars) {
			auto cs = car->GetState();
			if (cs.pos.z > 300.f && cs.pos.z < ball.pos.z)
				anyAirborneBelow = true;
		}
		CHECK(anyAirborneBelow);
	}
}
```

(API notes for the implementer: `Arena::ball->GetState()` returns `BallState` with `.pos`; `Car::GetState()` returns `CarState` with `.pos`, `.boost`. If member names differ, mirror the usage in `bot/src/env/StateSetters.cpp`, which sets these exact fields. RocketSim boost may be 0–100; if setters store 0–1, adapt the assertions to the convention `StateSetters.cpp` uses.)

- [ ] **Step 2: Build and run; triage failures as findings**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

Any failure here is a **real state-setter bug** (out-of-bounds spawn, claim not matching behavior). Fix the setter in `StateSetters.cpp` — not the test — unless the test misread the RocketSim API (e.g. boost scale), in which case fix the test and note the convention in `TestCommon.h`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add state setter validity tests"
```

---

### Task 7: Curriculum tracking, scenario metrics, reward-share metrics

**Files:**
- Create: `bot/src/env/Curriculum.h`, `bot/src/env/Curriculum.cpp`
- Create: `bot/src/train/Metrics.h`, `bot/src/train/Metrics.cpp`
- Create: `bot/tests/test_metrics.cpp`
- Modify: `bot/src/env/Env.cpp` (use CurriculumState), `bot/src/train/Train.cpp` (StepCallback additions), `bot/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
// env/Curriculum.h
struct CurriculumEntry { RLGC::StateSetter* setter; float weight; std::string name; };

// Drop-in replacement for RLGC::CombinedState that remembers which child it
// last picked. One instance per arena (CreateEnv makes one per env), so the
// last-picked name needs no locking.
class CurriculumState : public RLGC::StateSetter {
public:
	explicit CurriculumState(std::vector<CurriculumEntry> entries); // drops weight<=0 entries, deleting their setters; throws if none remain
	void ResetArena(Arena* arena) override;
	const std::string& LastPickedName() const; // empty until first reset
};

// train/Metrics.h
// Convert per-term Σ|weighted reward| into fractions of the total.
// Returns an empty vector when the total is zero.
std::vector<float> NormalizeShares(const std::vector<float>& weightedAbsTotals);
```

- Consumes: `GeneralRewardSpecs` (Task 5), `learner->envSet` internals per Global Constraints.

- [ ] **Step 1: Write failing tests**

`bot/tests/test_metrics.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/Curriculum.h>
#include <env/StateSetters.h>
#include <train/Metrics.h>

using namespace Hive;

TEST_CASE("NormalizeShares divides by the total") {
	auto shares = NormalizeShares({3.f, 1.f});
	REQUIRE(shares.size() == 2);
	CHECK(shares[0] == doctest::Approx(0.75f));
	CHECK(shares[1] == doctest::Approx(0.25f));
}

TEST_CASE("NormalizeShares of all zeros is empty") {
	CHECK(NormalizeShares({0.f, 0.f, 0.f}).empty());
	CHECK(NormalizeShares({}).empty());
}

TEST_CASE("CurriculumState picks children and records their names") {
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	CurriculumState cs({
		{new NeutralPlayState(), 1.f, "NeutralPlay"},
		{new AerialState(), 1.f, "Aerial"},
		{new BallContactState(), 0.f, "NeverPicked"}, // dropped: zero weight
	});

	CHECK(cs.LastPickedName().empty());

	bool sawNeutral = false, sawAerial = false;
	for (int i = 0; i < 200; i++) {
		cs.ResetArena(arena);
		const std::string& name = cs.LastPickedName();
		CHECK(name != "NeverPicked");
		if (name == "NeutralPlay") sawNeutral = true;
		if (name == "Aerial") sawAerial = true;
	}
	CHECK(sawNeutral);
	CHECK(sawAerial);
	delete arena;
}

TEST_CASE("CurriculumState with no positive weights throws") {
	CHECK_THROWS(CurriculumState({{new NeutralPlayState(), 0.f, "x"}}));
}
```

Add `tests/test_metrics.cpp` plus the four new source files to CMake. Build → expected **compile failure** (files don't exist).

- [ ] **Step 2: Implement CurriculumState**

`bot/src/env/Curriculum.cpp` (header per Interfaces; private members `std::vector<CurriculumEntry> entries; std::vector<float> cumulative; float total = 0.f; std::string lastPicked;`):

```cpp
CurriculumState::CurriculumState(std::vector<CurriculumEntry> in) {
	for (auto& e : in) {
		if (e.weight > 0.f) {
			total += e.weight;
			cumulative.push_back(total);
			entries.push_back(std::move(e));
		} else {
			delete e.setter;
		}
	}
	if (entries.empty())
		throw std::runtime_error("CurriculumState: every entry has zero weight");
}

void CurriculumState::ResetArena(Arena* arena) {
	const float f = RocketSim::Math::RandFloat(0, total);
	for (size_t i = 0; i < entries.size(); i++) {
		if (f <= cumulative[i]) {
			lastPicked = entries[i].name;
			entries[i].setter->ResetArena(arena);
			return;
		}
	}
	lastPicked = entries.back().name;
	entries.back().setter->ResetArena(arena);
}
```

(Match `RocketSim::Math::RandFloat`'s actual qualification to what `CombinedState.h` uses. Child setters are process-lifetime, matching upstream ownership.)

`bot/src/train/Metrics.cpp`:

```cpp
std::vector<float> NormalizeShares(const std::vector<float>& totals) {
	float sum = 0.f;
	for (float t : totals)
		sum += t;
	if (sum <= 0.f)
		return {};
	std::vector<float> out;
	out.reserve(totals.size());
	for (float t : totals)
		out.push_back(t / sum);
	return out;
}
```

In `bot/src/env/Env.cpp`, rewrite `BuildGeneralCurriculum` to return a `CurriculumState`:

```cpp
static StateSetter* BuildGeneralCurriculum(const CurriculumWeights& w) {
	return new CurriculumState({
		{new NeutralPlayState(), w.neutralPlay, "NeutralPlay"},
		{new BallContactState(), w.ballContact, "BallContact"},
		{new DefendState(), w.defend, "Defend"},
		{new RecoverState(), w.recover, "Recover"},
		{new AerialState(), w.aerial, "Aerial"},
		{new AirDribbleState(), w.airDribble, "AirDribble"},
		{new FlipResetState(), w.flipReset, "FlipReset"},
		{new DemoState(), w.demo, "Demo"},
		{new FuzzedKickoffState(), w.kickoff, "Kickoff"},
	});
}
```

- [ ] **Step 3: Build, run tests, expect PASS**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

- [ ] **Step 4: Wire the two new metric families into StepCallback**

In `bot/src/train/Train.cpp`:
- Add file-scope `static std::vector<std::pair<std::string, float>> g_RewardLabels;` (name, weight per term). In `RunTraining`, before constructing the learner: `for (auto& s : GeneralRewardSpecs(cfg)) g_RewardLabels.push_back({s.name, s.weight});` (the specs' `make` lambdas are never called here — no rewards are allocated).
- Inside `StepCallback`, after the existing per-player loop (still inside the `sample` gate), add:

```cpp
	// --- Reward shares ------------------------------------------------------
	// lastRewards holds each term's raw (unweighted, pre-zero-sum) reward for
	// one sampled player per arena; |r * w| across terms approximates where
	// the realized reward mass is going. This is the farming detector.
	auto& envSet = *learner->envSet;
	if (!g_RewardLabels.empty()) {
		std::vector<float> totals(g_RewardLabels.size(), 0.f);
		bool any = false;
		for (size_t a = 0; a < envSet.state.lastRewards.size(); a++) {
			const auto& last = envSet.state.lastRewards[a];
			if (last.size() != totals.size())
				continue;
			for (size_t j = 0; j < totals.size(); j++)
				totals[j] += std::fabs(last[j] * g_RewardLabels[j].second);
			any = true;
		}
		if (any) {
			auto shares = NormalizeShares(totals);
			for (size_t j = 0; j < shares.size(); j++)
				report.AddAvg("RewardShare/" + g_RewardLabels[j].first, shares[j]);
		}
	}

	// --- Scenario outcomes --------------------------------------------------
	// Terminal arenas have not been reset yet at callback time, so the
	// curriculum's last-picked name still labels the episode that just ended.
	for (size_t a = 0; a < envSet.stateSetters.size(); a++) {
		auto* cs = dynamic_cast<CurriculumState*>(envSet.stateSetters[a]);
		if (!cs || cs->LastPickedName().empty())
			continue;
		report.AddAvg("Scenario/" + cs->LastPickedName() + "/Share", 1.f);
		if (envSet.state.terminals[a]) {
			const bool goal = states[a].goalScored;
			report.AddAvg("Scenario/" + cs->LastPickedName() + "/EndedInGoal", goal ? 1.f : 0.f);
		}
	}
```

Add includes: `../env/Curriculum.h`, `../env/Rewards.h`, `Metrics.h`, `<RLGymCPP/EnvSet/EnvSet.h>`, `<cmath>`.

Note: `Scenario/<name>/Share` as written averages 1.0 for every scenario that appears (AddAvg only sees samples where it fired). To make it a true share, count names into a local `std::map<std::string, int>` across arenas and report `count / envSet.stateSetters.size()` per name — do it that way, mirroring how the existing `Phase/` block reports fractions.

- [ ] **Step 5: Build, smoke, verify metrics exist**

```bash
scripts/build.sh && cd bot/build && ./HiveTests
./HivemindBot train --max-steps 300000 --games 16 --label smoke-task7
```

With metrics on, check the run's CSV (written by `bot/metrics/metric_receiver.py` under `bot/build/`) contains columns starting `RewardShare/` and `Scenario/`; `RewardShare/*` columns for one row should sum to ≈1.0. Clean up the smoke checkpoint.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Track curriculum scenarios, log reward shares and scenario outcomes"
```

---

### Task 8: PacketConvert round-trip test

**Files:**
- Create: `bot/tests/test_packetconvert.cpp`
- Modify: `bot/CMakeLists.txt`

**Interfaces:**
- Consumes: `Hive::PacketConverter` (`Initialize(const rlbot::flat::FieldInfo*)`, `Convert(const rlbot::flat::GamePacket*)`), flatbuffers object API from the generated headers in `bot/build/rlbot-cpp/misc_generated.h` (types `GamePacketT`, `PlayerInfoT`, `BallInfoT`, `MatchInfoT`, structs `Physics(location, rotation, velocity, angular_velocity)`, `Vector3(x,y,z)`, `Rotator(pitch,yaw,roll)` — verify Rotator's ctor order in the header), `RLGC::CommonValues::BOOST_LOCATIONS`.

- [ ] **Step 1: Write the round-trip test**

`bot/tests/test_packetconvert.cpp`:

```cpp
#include "doctest/doctest.h"
#include "TestCommon.h"

#include <rlbot/PacketConvert.h>

#include <RLGymCPP/CommonValues.h>

using namespace Hive;

namespace {

// Build a serialized GamePacket via the flatbuffers object API, then view it
// through the same accessor types the live RLBot connection delivers.
struct PacketFixture {
	flatbuffers::FlatBufferBuilder fbb;
	const rlbot::flat::GamePacket* packet = nullptr;

	void Build(rlbot::flat::GamePacketT& pkt) {
		fbb.Clear();
		fbb.Finish(rlbot::flat::GamePacket::Pack(fbb, &pkt));
		packet = flatbuffers::GetRoot<rlbot::flat::GamePacket>(fbb.GetBufferPointer());
	}
};

std::unique_ptr<rlbot::flat::PlayerInfoT> MakePlayer(float x, float y, float z, uint32_t team, int32_t id) {
	auto p = std::make_unique<rlbot::flat::PlayerInfoT>();
	p->physics = std::make_unique<rlbot::flat::Physics>(
		rlbot::flat::Vector3(x, y, z),
		rlbot::flat::Rotator(0.f, 0.f, 0.f),
		rlbot::flat::Vector3(100.f, 0.f, 0.f),
		rlbot::flat::Vector3(0.f, 0.f, 0.f));
	p->team = team;
	p->player_id = id;
	p->boost = 33.f;
	p->air_state = rlbot::flat::AirState::OnGround;
	p->demolished_timeout = -1.f;
	p->dodge_timeout = -1.f;
	return p;
}

rlbot::flat::GamePacketT MakeBasePacket() {
	rlbot::flat::GamePacketT pkt;
	pkt.match_info = std::make_unique<rlbot::flat::MatchInfoT>();
	pkt.match_info->seconds_elapsed = 10.f;
	pkt.match_info->match_phase = rlbot::flat::MatchPhase::Active;

	auto ball = std::make_unique<rlbot::flat::BallInfoT>();
	ball->physics = std::make_unique<rlbot::flat::Physics>(
		rlbot::flat::Vector3(500.f, -1000.f, 93.f),
		rlbot::flat::Rotator(0.f, 0.f, 0.f),
		rlbot::flat::Vector3(-250.f, 400.f, 10.f),
		rlbot::flat::Vector3(0.f, 0.f, 0.f));
	pkt.balls.push_back(std::move(ball));

	pkt.players.push_back(MakePlayer(-2000.f, -3000.f, 17.f, 0, 7));
	pkt.players.push_back(MakePlayer(2000.f, 3000.f, 17.f, 1, 8));

	for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		pkt.boost_pads.push_back(rlbot::flat::BoostPadState(true, 0.f));
	return pkt;
}

} // namespace

TEST_CASE("Convert carries ball and player physics through") {
	PacketConverter conv;
	conv.Initialize(nullptr); // identity pad mapping

	PacketFixture fx;
	auto pkt = MakeBasePacket();
	fx.Build(pkt);

	RLGC::GameState gs = conv.Convert(fx.packet);

	CHECK(gs.ball.pos.x == doctest::Approx(500.f));
	CHECK(gs.ball.vel.y == doctest::Approx(400.f));
	REQUIRE(gs.players.size() == 2);
	CHECK(gs.players[0].pos.y == doctest::Approx(-3000.f));
	CHECK(gs.players[0].team == RLGC::Team::BLUE);
	CHECK(gs.players[1].team == RLGC::Team::ORANGE);
	CHECK(gs.players[0].boost == doctest::Approx(33.f));
	CHECK(gs.players[0].isOnGround);
	CHECK(!gs.players[0].isDemoed);
}

TEST_CASE("touch timestamps become per-step touch flags exactly once") {
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;

	auto pkt = MakeBasePacket();
	auto touch = std::make_unique<rlbot::flat::TouchT>();
	touch->game_seconds = 9.5f;
	pkt.players[0]->latest_touch = std::move(touch);
	fx.Build(pkt);
	RLGC::GameState first = conv.Convert(fx.packet);
	// First sighting of a touch time only seeds the baseline.
	CHECK(!first.players[0].ballTouchedStep);

	auto pkt2 = MakeBasePacket();
	auto touch2 = std::make_unique<rlbot::flat::TouchT>();
	touch2->game_seconds = 10.5f;
	pkt2.players[0]->latest_touch = std::move(touch2);
	fx.Build(pkt2);
	RLGC::GameState second = conv.Convert(fx.packet);
	CHECK(second.players[0].ballTouchedStep);

	fx.Build(pkt2);
	RLGC::GameState third = conv.Convert(fx.packet);
	// Same timestamp again: not a new touch.
	CHECK(!third.players[0].ballTouchedStep);
}

TEST_CASE("an inactive pad reads back inactive through the identity mapping") {
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;

	auto pkt = MakeBasePacket();
	pkt.boost_pads[5] = rlbot::flat::BoostPadState(false, 3.5f);
	fx.Build(pkt);
	RLGC::GameState gs = conv.Convert(fx.packet);

	CHECK(!gs.boostPads[5]);
	CHECK(gs.boostPadTimers[5] == doctest::Approx(3.5f));
	const int inv = RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT - 5 - 1;
	CHECK(!gs.boostPadsInv[inv]);
}
```

(Implementer notes: `BoostPadState` is a flatbuffers struct — check its ctor arg order (`is_active`, `timer`) in the generated header. `TouchT` field may be named `game_seconds`; confirm in `misc_generated.h`. Each `PacketConverter` keeps touch history, so tests construct a fresh converter.)

- [ ] **Step 2: Add generated-header include path if needed, build, run**

`HiveTests` already links `RLBotCPP-static` via HiveCore; if `rlbot::flat` object-API types fail to resolve, add `target_include_directories(HiveTests PRIVATE "${CMAKE_BINARY_DIR}/rlbot-cpp")` — but first check what include path `bot/src/rlbot/HivemindBot.cpp` gets them from (they come transitively from `<rlbot/Bot.h>`).

```bash
scripts/build.sh && cd bot/build && ./HiveTests
```

Expected: all PASS. A failure in the touch or pad tests is a real deployment-parity bug — fix `PacketConvert.cpp`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add PacketConvert round-trip tests"
```

---

### Task 9: `verify` subcommand — checkpoint/deployment parity check

**Files:**
- Create: `bot/src/Verify.h`, `bot/src/Verify.cpp`
- Modify: `bot/src/main.cpp`, `bot/CMakeLists.txt`

**Interfaces:**
- Produces: `int Hive::RunVerify(const std::filesystem::path& checkpointFolder);` — returns 0 on pass, 1 on failure. CLI: `HivemindBot verify <checkpoint-folder>`.
- Consumes: `Policy`, `MakeObsBuilder`, `ProbeObsSize`, `ModelShape` (Tasks 2–3).

- [ ] **Step 1: Implement RunVerify**

`bot/src/Verify.cpp` — the checks, in order, each printing PASS/FAIL:

```cpp
#include "Verify.h"

#include "Config.h"
#include "env/Obs.h"
#include "policy/Policy.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cstdio>
#include <cstring>

using namespace RLGC;

namespace Hive {

static bool SameAction(const Action& a, const Action& b) {
	return std::memcmp(&a, &b, sizeof(Action)) == 0;
}

int RunVerify(const std::filesystem::path& folder) {
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	TrainConfig cfg = {};
	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam);
	std::printf("Obs size: %d (maxPlayersPerTeam=%d)\n", obsSize, cfg.maxPlayersPerTeam);

	auto obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam);
	DefaultAction parser;

	// 1. The checkpoint loads under the compiled-in ModelShape. A shape
	//    mismatch throws here instead of silently misplaying in a match.
	Policy policy(obsBuilder.get(), obsSize, &parser, cfg.modelShape, /*useGPU=*/false);
	try {
		policy.Load(folder);
		std::printf("PASS  checkpoint loads with compiled ModelShape\n");
	} catch (const std::exception& e) {
		std::printf("FAIL  checkpoint load: %s\n", e.what());
		return 1;
	}

	// 2. Deterministic inference is actually deterministic, and the policy is
	//    not degenerate (always the same action regardless of state).
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);
	arena->ResetToRandomKickoff();

	int distinct = 0;
	bool deterministic = true;
	Action prev = {};
	for (int i = 0; i < 200; i++) {
		// Random controls step the arena into varied states.
		for (Car* car : arena->_cars) {
			CarControls c = {};
			c.throttle = Math::RandFloat(-1, 1);
			c.steer = Math::RandFloat(-1, 1);
			c.boost = Math::RandFloat(0, 1) > 0.7f;
			c.jump = Math::RandFloat(0, 1) > 0.9f;
			car->controls = c;
		}
		arena->Step(8);

		GameState gs(arena);
		auto a1 = policy.InferBatch({gs.players[0]}, {gs}, true);
		auto a2 = policy.InferBatch({gs.players[0]}, {gs}, true);
		if (!SameAction(a1[0], a2[0]))
			deterministic = false;
		if (i == 0 || !SameAction(a1[0], prev))
			distinct++;
		prev = a1[0];
	}
	delete arena;

	std::printf("%s  deterministic inference is repeatable\n", deterministic ? "PASS" : "FAIL");
	std::printf("%s  policy output varies with state (%d distinct actions over 200 states)\n",
	            distinct > 5 ? "PASS" : "FAIL", distinct);

	// 3. Deployment env vars, if set, agree with compiled training values.
	bool parity = true;
	struct { const char* env; int expected; } checks[] = {
		{"HIVE_TICK_SKIP", cfg.tickSkip},
		{"HIVE_ACTION_DELAY", cfg.actionDelay},
		{"HIVE_MAX_PLAYERS_PER_TEAM", cfg.maxPlayersPerTeam},
	};
	for (auto& c : checks) {
		const char* v = std::getenv(c.env);
		if (v && *v && std::atoi(v) != c.expected) {
			std::printf("FAIL  %s=%s but training used %d\n", c.env, v, c.expected);
			parity = false;
		}
	}
	if (parity)
		std::printf("PASS  HIVE_* env parity (unset vars use compiled defaults)\n");

	const bool ok = deterministic && distinct > 5 && parity;
	std::printf("%s\n", ok ? "VERIFY PASSED" : "VERIFY FAILED");
	return ok ? 0 : 1;
}

} // namespace Hive
```

(`GameState gs(arena)` mirrors `Obs.cpp`'s probe; `Math::RandFloat` qualification as in `CombinedState.h`. If `CarControls` field names differ, mirror `RLGC::Action`'s.)

`bot/src/Verify.h` declares `RunVerify`. In `main.cpp` add:

```cpp
	if (command == "verify") {
		if (argc < 3) {
			std::fprintf(stderr, "Usage: %s verify <checkpoint-folder>\n", argv[0]);
			return EXIT_FAILURE;
		}
		return Hive::RunVerify(argv[2]);
	}
```

and a usage line: `verify <folder>   Check a checkpoint loads and infers sanely before deploying it`.

- [ ] **Step 2: Build, run against a smoke checkpoint**

```bash
scripts/build.sh && cd bot/build
./HivemindBot train --max-steps 200000 --games 16 --no-metrics --label smoke-verify
./HivemindBot verify $(ls -d checkpoints/main-smoke-verify/*/ | tail -1)
```

Expected: `VERIFY PASSED`, exit 0. Also verify the failure path: `./HivemindBot verify /tmp/nonexistent` must print the load FAIL and exit 1. Clean up the smoke checkpoint.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add verify subcommand for checkpoint parity checks"
```

---

### Task 10: `eval` subcommand — head-to-head checkpoint matches

**Files:**
- Create: `bot/src/eval/Eval.h`, `bot/src/eval/Eval.cpp`
- Modify: `bot/src/main.cpp`, `bot/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
struct EvalConfig {
	std::filesystem::path blueModel;
	std::filesystem::path orangeModel;
	int games = 20;
	float maxSeconds = 300.f;   // per game, sim time
	bool useGPU = true;
	int64_t seed = -1;
};
struct EvalResult { int blueWins, orangeWins, draws, blueGoals, orangeGoals; };
EvalResult RunEval(const EvalConfig& cfg);
```

CLI: `HivemindBot eval --blue <ckpt> --orange <ckpt> [--games N] [--seconds S] [--cpu]`. This is the frozen-reference-pool tool: pit any two checkpoints (current vs a gate checkpoint, run A vs run B) headlessly.
- Consumes: `Policy`, `MakeObsBuilder`, `ProbeObsSize`, `TrainConfig` defaults for tickSkip/actionDelay.

- [ ] **Step 1: Implement RunEval**

`bot/src/eval/Eval.cpp` core loop (RocketSim only, no learner). Cadence replays training exactly: infer every `tickSkip` ticks, apply the fresh action `actionDelay` ticks later — the same queued/applied pattern as `HivemindBot::update`:

```cpp
EvalResult RunEval(const EvalConfig& ecfg) {
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	TrainConfig cfg = {};
	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam);
	auto obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam);
	DefaultAction parser;

	Policy blue(obsBuilder.get(), obsSize, &parser, cfg.modelShape, ecfg.useGPU);
	Policy orange(obsBuilder.get(), obsSize, &parser, cfg.modelShape, ecfg.useGPU);
	blue.Load(ecfg.blueModel);
	orange.Load(ecfg.orangeModel);

	EvalResult res = {};
	for (int game = 0; game < ecfg.games; game++) {
		Arena* arena = Arena::Create(GameMode::SOCCAR);
		Car* blueCar = arena->AddCar(Team::BLUE);
		Car* orangeCar = arena->AddCar(Team::ORANGE);

		int scoreBlue = 0, scoreOrange = 0;

		// RocketSim: typedef std::function<void(Arena*, Team scoringTeam, void*)>
		// GoalScoreEventFn; SetGoalScoreCallback(fn, void* userInfo = NULL).
		struct GoalFlag { bool scored = false; Team team = Team::BLUE; } goalFlag;
		arena->SetGoalScoreCallback(
			[](Arena*, Team team, void* user) {
				auto* g = static_cast<GoalFlag*>(user);
				g->scored = true;
				g->team = team;
			},
			&goalFlag);

		arena->ResetToRandomKickoff();

		struct Held { Action queued = {}, applied = {}; };
		Held held[2];

		const int totalTicks = static_cast<int>(ecfg.maxSeconds * 120.f);
		for (int tick = 0; tick < totalTicks; tick += cfg.tickSkip) {
			GameState gs(arena);

			auto actBlue = blue.InferBatch({gs.players[0]}, {gs}, true);
			auto actOrange = orange.InferBatch({gs.players[1]}, {gs}, true);
			held[0].queued = actBlue[0];
			held[1].queued = actOrange[0];

			// actionDelay: hold the old action for the first actionDelay ticks
			// of this window, then apply the fresh one.
			arena->Step(cfg.actionDelay);
			held[0].applied = held[0].queued;
			held[1].applied = held[1].queued;
			blueCar->controls = (CarControls)held[0].applied;
			orangeCar->controls = (CarControls)held[1].applied;
			arena->Step(cfg.tickSkip - cfg.actionDelay);

			if (goalFlag.scored) {
				if (goalFlag.team == Team::BLUE) scoreBlue++;
				else scoreOrange++;
				goalFlag.scored = false;
				arena->ResetToRandomKickoff();
			}
		}

		res.blueGoals += scoreBlue;
		res.orangeGoals += scoreOrange;
		if (scoreBlue > scoreOrange) res.blueWins++;
		else if (scoreOrange > scoreBlue) res.orangeWins++;
		else res.draws++;

		std::printf("Game %d/%d: blue %d - %d orange\n", game + 1, ecfg.games, scoreBlue, scoreOrange);
		delete arena;
	}
	return res;
}
```

Implementer notes (resolve against RocketSim headers, they are in `external/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/RocketSim/src`):
- `Arena::SetGoalScoreCallback(GoalScoreEventFn, void* userInfo)` — check the exact callback signature; the sketch's user-pointer pattern is the intent. If the callback provides the scoring team, credit that team.
- Players in `GameState(arena)` appear in car-add order: index 0 blue, 1 orange. Assert `gs.players[0].team == Team::BLUE` once per game.
- Controls setting must precede the `Step` that should feel them; the two-phase step above reproduces training's delay=7/skip=8 within one window. Keep the arithmetic in terms of `cfg.actionDelay`/`cfg.tickSkip`.
- Print a final summary: wins/draws/goals per side.

`main.cpp`: parse `--blue`, `--orange` (both required), `--games`, `--seconds`, `--cpu`; add a usage line: `eval --blue A --orange B   Play two checkpoints against each other in RocketSim`.

- [ ] **Step 2: Build, self-play sanity run**

```bash
scripts/build.sh && cd bot/build
./HivemindBot train --max-steps 200000 --games 16 --no-metrics --label smoke-eval
CKPT=$(ls -d checkpoints/main-smoke-eval/*/ | tail -1)
./HivemindBot eval --blue "$CKPT" --orange "$CKPT" --games 4 --seconds 60
```

Expected: 4 games complete with plausible output. A checkpoint against itself should be roughly balanced (small samples swing — the check is that games run and goals are attributed, not the exact score).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add eval subcommand: headless checkpoint-vs-checkpoint matches"
```

---

### Task 11: Ops tooling — throughput sweep, match script, run log

**Files:**
- Create: `scripts/throughput.sh`
- Create: `scripts/match.sh`
- Create: `runs/RUNLOG.md`
- Modify: `scripts/play.sh` (no change expected — verify only)

**Interfaces:**
- Consumes: `HivemindBot train --max-steps/--games/--label` (Task 2), `bot/rlbot-config/` (Task 3).
- Produces: `scripts/throughput.sh [game-counts...]`; `scripts/match.sh <opponent-bot-toml>`; `runs/RUNLOG.md` append-per-run convention.

- [ ] **Step 1: throughput.sh**

```bash
#!/usr/bin/env bash
# Measure training steps/sec across --games values. Wall-clock over a fixed
# step budget, so startup cost is included -- keep the budget large enough
# (5M) that it amortizes. Results are for RUNLOG.md; pick the numGames that
# wins and note it there.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/HivemindBot"
STEPS="${STEPS:-5000000}"

[[ -x "$BIN" ]] || { echo "Build first: scripts/build.sh" >&2; exit 1; }

GAMES=("${@:-64 128 192 256 320}")
[[ $# -gt 0 ]] && GAMES=("$@")

cd "$BUILD_DIR"
echo "games  steps  seconds  steps/sec"
for n in ${GAMES[@]}; do
	label="throughput-$n-$(date +%s)"
	start=$(date +%s)
	"$BIN" train --max-steps "$STEPS" --games "$n" --no-metrics --label "$label" > /dev/null
	end=$(date +%s)
	secs=$((end - start))
	rm -rf "checkpoints/main-$label"
	echo "$n  $STEPS  $secs  $((STEPS / secs))"
done
```

- [ ] **Step 2: match.sh**

```bash
#!/usr/bin/env bash
# Run an RLBot match: our bot vs an opponent bot's .toml (e.g. from RLBotPack).
#   scripts/match.sh /path/to/opponent/bot.toml
# Requires the rlbot CLI and Rocket League runnable on this machine.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO/bot/rlbot-config"
OPP="${1:?Usage: scripts/match.sh /path/to/opponent/bot.toml}"
OPP="$(realpath "$OPP")"

[[ -f "$OPP" ]] || { echo "No such file: $OPP" >&2; exit 1; }
command -v rlbot > /dev/null || { echo "Install the RLBot CLI: pipx install rlbot" >&2; exit 1; }

GEN="$CONFIG_DIR/match-vs-generated.toml"
cat > "$GEN" <<EOF
# Generated by scripts/match.sh -- do not edit; regenerate instead.
[rlbot]
launcher = "Steam"
auto_start_agents = true
wait_for_agents = true

[match]
game_mode = "Soccar"
game_map_upk = "Stadium_P"
existing_match_behavior = "Restart"
enable_state_setting = true

[[cars]]
team = "Blue"
type = "RLBot"
config_file = "bot.toml"

[[cars]]
team = "Orange"
type = "RLBot"
config_file = "$OPP"
EOF

cd "$CONFIG_DIR"
exec rlbot run "$(basename "$GEN")"
```

Add `bot/rlbot-config/match-vs-generated.toml` to `.gitignore`.

- [ ] **Step 3: RUNLOG.md**

`runs/RUNLOG.md`:

```markdown
# Run log

One line per run that matters. Comparisons are only valid between labeled
runs recorded here. Append newest at the top.

Format: `date | label | config delta from previous entry | why | outcome`

| Date | Label | Config delta | Why | Outcome |
|---|---|---|---|---|
| _(none yet)_ | | | | |
```

- [ ] **Step 4: Make scripts executable, run the throughput sweep once**

```bash
chmod +x scripts/throughput.sh scripts/match.sh
scripts/throughput.sh 64 128 256
```

Record the results as the first RUNLOG entry (label `throughput-*`, outcome = the steps/sec table; note the winning `--games` value). This is a real Phase-0 deliverable: it re-measures throughput at 1v1 width per the spec.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add throughput sweep, RLBot match script, run log"
```

---

### Task 12: Documentation rewrite

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/metrics.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything built in Tasks 1–11; the spec at `docs/superpowers/specs/2026-08-17-hivemind-roadmap-design.md`.

- [ ] **Step 1: Write docs/architecture.md**

Cover, briefly and only as verified against the code: single policy trained on 1v1 (kickoffs via curriculum — why the old two-policy split was removed, one paragraph); the padded observation at `maxPlayersPerTeam = 1` and why width changes invalidate checkpoints; training/deployment parity (tickSkip, actionDelay, ModelShape, the `verify` subcommand as the mechanized check); the deployment path (RLBot v5, `run.sh` env vars, PacketConvert's two silent-risk areas: pad mapping and flip availability); pointer to the spec for the phase roadmap. Target length: ~100 lines. No aspirational content — document what exists.

- [ ] **Step 2: Write docs/metrics.md (the metrics guide)**

A table-driven living document. For each metric family, three columns: what it measures, what healthy looks like (trend language, no absolute thresholds — per spec D7), which decision it feeds. Cover at minimum:

| Metric | Measures | Healthy | Feeds |
|---|---|---|---|
| `RewardShare/*` | Fraction of realized \|weighted reward\| per term (sampled; zero-sum terms measured pre-zero-sum) | Outcome terms (Goal, StrongTouch) growing over a phase; no shaping term dominating late | Phase-gate decisions; farming detection |
| `Scenario/*/Share` | Fraction of arenas running each curriculum scenario | Matches configured curriculum weights | Verifying setters actually run |
| `Scenario/*/EndedInGoal` | Episode outcome per scenario | Rising within a phase | Whether a scenario is being learned |
| `Player/Ball Touch Ratio` | Touch frequency | Stable or rising; a collapse after a reward change = new degenerate behavior | Reward-change rollback |
| `Player/Touch Height` | Air game development | Rising from P2 onward without touch ratio falling | P2/P3 gates |
| `Phase/*` | Time share per play phase (metric labels, not curriculum) | Neutral dominant; shifts tracking curriculum changes | Curriculum tuning |
| `Rating/*` (skill tracker) | Elo vs. past versions | Monotonic rise; falling while reward rises = farming | The primary gate signal |
| `Game/Goal Speed` | Shot power at goals | Rising through P1–P2 | Striking quality |

Add a short "how to watch a run" section: check RewardShare first, then Rating, then watch RocketSimVis for a few minutes — always within the first hour after any reward or curriculum change.

- [ ] **Step 3: Update README.md and CLAUDE.md**

README: fix the policy table (one policy), the quickstart commands (`scripts/train.sh` with no target argument; remove the stray `%command%` line), the layout section (PlayPhase/Curriculum/Policy files, tests, new scripts, `verify`/`eval` subcommands), the doc links (architecture.md, metrics.md, the spec), and the measured-throughput table (append the Task 11 sweep results; keep old rows labeled as pre-1v1). Keep "never judge a run by its reward curve".

CLAUDE.md: replace the "two policies, not a mixture of experts" section with the single-policy decision and a pointer to the spec ("this was re-decided 2026-08-17; the kickoff/general split was removed deliberately"); update layout notes (tests exist, `HiveTests` target); update the parity-traps paragraph to mention `HivemindBot verify`; update "Verified working" numbers only with values actually measured in this phase; keep the build gotchas and external-patch sections untouched.

- [ ] **Step 4: Verify doc claims against reality**

Every command named in the docs gets run once (`scripts/train.sh --max-steps 100000 --games 8 --no-metrics --label doccheck`, `HivemindBot verify ...`, `./HiveTests`); every file path named gets `ls`-checked. Delete the doccheck checkpoint.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Rewrite docs for single-policy architecture, add metrics guide"
```

---

### Task 13: Live deployment verification (user-in-the-loop)

**Files:**
- Modify: `runs/RUNLOG.md` (record outcomes)
- Possibly modify: `bot/src/rlbot/*`, `bot/rlbot-config/*` (fixes found live)

**Interfaces:**
- Consumes: everything. This task discharges the spec's biggest open risk: the RLBot client has never run a live match.

**This task cannot be completed by an agent alone** — it needs Rocket League running and the user present. The agent's job: prepare everything, drive the checklist, fix what breaks.

- [ ] **Step 1: Install the RLBot CLI**

```bash
pipx install rlbot || pip install --user rlbot
rlbot --help
```

- [ ] **Step 2: Produce a real throwaway checkpoint**

```bash
cd bot/build
./HivemindBot train --max-steps 30000000 --games 128 --label deploy-probe
./HivemindBot verify $(ls -d checkpoints/main-deploy-probe/*/ | tail -1)
```

(~30M steps ≈ tens of minutes; enough that the bot visibly chases the ball, which makes parity failures obvious — a bot that drives at the ball in sim but wanders in-game is the signature of an obs mismatch.)

- [ ] **Step 3: Live match checklist (with the user)**

1. User launches Rocket League (Steam, per `match-1v1.toml`; adjust `launcher` if they use Epic/Legendary).
2. `HIVE_MODEL=$(ls -d bot/build/checkpoints/main-deploy-probe/*/ | tail -1) scripts/play.sh`.
3. Confirm, in order: RLBot connects; the bot's car moves at all; the bot chases the ball roughly as it did in RocketSimVis at this checkpoint; no error spam in the bot's console.
4. If behavior diverges from sim: suspect `PacketConvert` (pad mapping warning in the console? flip logic?) and the cadence fields; the Task 8 tests plus `verify` narrow it. Fix, rebuild, retry.
5. Record the session and any fixes in `RUNLOG.md`.

- [ ] **Step 4: Nexto/v4-bridge probe (with the user)**

1. Download Nexto from the RLBotPack (github.com/RLBot/RLBotPack).
2. `scripts/match.sh /path/to/Nexto/bot.toml`.
3. Outcome A — it runs: record "v4 bots work under v5" in RUNLOG; the benchmark ladder is open.
4. Outcome B — it doesn't: check RLBot v5's current guidance on legacy-bot support (the v4 adapter situation changes; consult wiki.rlbot.org), record findings and the workaround path in RUNLOG. Do not sink more than an evening into this; it gates nothing else in Phase 0.

- [ ] **Step 5: Commit whatever changed**

```bash
git add -A && git commit -m "Record live deployment verification; fixes from first real match"
```

---

## Self-review notes

- Spec coverage: restructure (T2–T3), metrics additions (T7), throughput re-measure (T11), deployment verification + Nexto probe (T13), match script + reference-pool tool + RUNLOG (T10–T11), metrics guide + docs + CLAUDE.md (T12), unit tests for setters/rewards/metrics (T4–T7), parity as executable code (T8–T9). Short validation runs to re-derive P1 weights are an *operational* Phase-0 item the user drives once this plan lands; the tooling for it (RewardShare metrics, RUNLOG) is T7/T11.
- The `eval` goal-callback and flatbuffers details are pinned to the actual generated headers with explicit "verify ctor order" notes where the header is the authority.
- Type consistency: `Policy::InferBatch` drops the `Regime` parameter everywhere (T3 HivemindBot, T9 verify, T10 eval all call the same 4-arg form).
