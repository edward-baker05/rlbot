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
