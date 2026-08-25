#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace Dash::Test {

// The test binary lands in bot/build/ next to the copied meshes, but is
// usually invoked from the repo root, so a bare relative "collision_meshes"
// does not resolve. Mirrors main.cpp's FindCollisionMeshes rather than forcing
// every caller to cd first.
inline std::string FindCollisionMeshes() {
	const char* env = std::getenv("DASH_COLLISION_MESHES");
	if (!env)
		env = std::getenv("HIVE_COLLISION_MESHES");
	if (env && std::filesystem::exists(std::filesystem::path(env) / "soccar"))
		return env;

	for (const char* p : {"collision_meshes", "bot/build/collision_meshes",
	                      "tools/collision_meshes", "../tools/collision_meshes",
	                      "../collision_meshes"}) {
		if (std::filesystem::exists(std::filesystem::path(p) / "soccar"))
			return p;
	}
	return "collision_meshes";
}

// RocketSim asserts if initialized twice and Arena creation asserts if never
// initialized; every test that touches an Arena funnels through here.
inline void EnsureRocketSim() {
	static bool done = false;
	if (!done) {
		RocketSim::Init(FindCollisionMeshes());
		done = true;
	}
}

} // namespace Dash::Test
