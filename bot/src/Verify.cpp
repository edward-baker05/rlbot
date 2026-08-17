#include "Verify.h"

#include "Config.h"
#include "env/Obs.h"
#include "policy/Policy.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cstdio>
#include <cstdlib>
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
