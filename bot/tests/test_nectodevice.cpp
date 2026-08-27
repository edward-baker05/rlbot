#include "doctest/doctest.h"
#include "TestCommon.h"

#include <opponents/NectoPolicy.h>

#include <RLGymCPP/CommonValues.h>

#include <torch/cuda.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace RLGC;
using namespace Dash;

namespace {

// Moving Necto onto the GPU is the kind of change that keeps running while it
// quietly plays worse -- the same failure NectoSelfTest exists to catch on the
// observation side. This covers the other half: given identical inputs, the two
// devices must choose identical actions.
std::filesystem::path FindNectoModel() {
	if (const char *env = std::getenv("DASH_NECTO_MODEL"))
		if (std::filesystem::is_regular_file(env))
			return env;

	for (const char *p : {
			 "libs/opponents/NectoFamily/necto/necto-model.pt",
			 "../libs/opponents/NectoFamily/necto/necto-model.pt",
			 "../../libs/opponents/NectoFamily/necto/necto-model.pt",
		 }) {
		if (std::filesystem::is_regular_file(p))
			return p;
	}
	return {};
}

// Distinct, awkward, and deliberately asymmetric in x, y and z, so a transposed
// column or a dropped sign flip cannot pass by symmetry. `k` walks the states
// across the ground/air, boost/no-boost and demoed/alive branches, since those
// change which columns are non-zero.
GameState MakeVariedState(int k) {
	const float f = static_cast<float>(k);

	GameState gs = {};
	gs.ball.pos = Vec(-431.5f + 137.f * f, 1287.25f - 211.f * f,
					  120.f + 97.f * (k % 7));
	gs.ball.vel = Vec(920.5f - 83.f * f, -1455.f + 179.f * f, 233.25f - 31.f * f);
	gs.ball.angVel = Vec(0.75f + 0.13f * f, -2.125f, 3.5f - 0.07f * f);

	gs.players.resize(2);

	Player &blue = gs.players[0];
	blue.index = 0;
	blue.carId = 1;
	blue.team = Team::BLUE;
	blue.pos = Vec(-1024.f + 211.f * f, -2048.5f + 97.f * f, 17.25f);
	blue.vel = Vec(455.f - 61.f * f, 1200.25f - 143.f * f, -30.5f);
	blue.angVel = Vec(-1.25f, 0.5f + 0.11f * f, 2.75f);
	blue.rotMat =
		RotMat(Vec(0.6f, -0.8f, 0.f), Vec(0.8f, 0.6f, 0.f), Vec(0.f, 0.f, 1.f));
	blue.boost = (k % 3 == 0) ? 0.f : 47.f;
	blue.isOnGround = true;
	blue.demoRespawnTimer = 0.f;

	Player &orange = gs.players[1];
	orange.index = 1;
	orange.carId = 2;
	orange.team = Team::ORANGE;
	orange.pos = Vec(2200.75f - 173.f * f, 3100.f - 229.f * f, 845.5f);
	orange.vel = Vec(-1310.f + 91.f * f, -220.5f, 640.f - 53.f * f);
	orange.angVel = Vec(3.25f, 1.75f - 0.09f * f, -0.5f);
	orange.rotMat = RotMat(Vec(0.f, 0.28f, 0.96f), Vec(1.f, 0.f, 0.f),
						   Vec(0.f, 0.96f, -0.28f));
	orange.boost = 88.f;
	orange.isOnGround = false;
	orange.hasJumped = true;
	orange.airTimeSinceJump = 0.05f;
	orange.demoRespawnTimer = (k % 5 == 0) ? 1.75f : 0.f;

	gs.boostPads.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	gs.boostPadTimers.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 0.f);
	gs.boostPads[(3 + k) % CommonValues::BOOST_LOCATIONS_AMOUNT] = false;
	gs.boostPadTimers[(3 + k) % CommonValues::BOOST_LOCATIONS_AMOUNT] = 7.5f;
	gs.boostPads[(7 + k) % CommonValues::BOOST_LOCATIONS_AMOUNT] = false;
	gs.boostPadTimers[(7 + k) % CommonValues::BOOST_LOCATIONS_AMOUNT] = 1.25f;

	return gs;
}

} // namespace

TEST_CASE("NectoPolicy picks the same actions on CPU and GPU") {
	const std::filesystem::path model = FindNectoModel();
	if (model.empty()) {
		MESSAGE("necto-model.pt not found; skipping device equivalence check");
		return;
	}
	if (!torch::cuda::is_available()) {
		MESSAGE("CUDA unavailable; skipping device equivalence check");
		return;
	}

	// beta 1.0 is argmax, so SampleAction consumes no RNG and the comparison is
	// a property of the forward pass alone rather than of two RNG streams
	// happening to agree.
	NectoPolicy cpu(model, 1.0f, 0, /*useGPU=*/false);
	NectoPolicy gpu(model, 1.0f, 0, /*useGPU=*/true);

	CHECK_FALSE(cpu.OnGPU());
	REQUIRE(gpu.OnGPU());

	constexpr int NUM_STATES = 24;
	std::vector<GameState> states;
	states.reserve(NUM_STATES);
	for (int k = 0; k < NUM_STATES; k++)
		states.push_back(MakeVariedState(k));

	// One request per car, batched exactly as NectoDriver batches them.
	std::vector<NectoRequest> requests;
	for (const GameState &gs : states) {
		for (int p = 0; p < static_cast<int>(gs.players.size()); p++) {
			Action prev = {};
			for (size_t e = 0; e < Action::ELEM_AMOUNT; e++)
				prev[e] = 0.25f * static_cast<float>((p + e) % 5) - 0.5f;
			requests.push_back({&gs, p, prev});
		}
	}
	REQUIRE(requests.size() == NUM_STATES * 2);

	std::vector<Action> cpuActions, gpuActions;
	cpu.InferBatch(requests, cpuActions);
	gpu.InferBatch(requests, gpuActions);

	REQUIRE(cpuActions.size() == requests.size());
	REQUIRE(gpuActions.size() == requests.size());

	// Exact equality: argmax over five heads yields values in {-1, 0, 1}, and
	// the products SampleAction forms from them are exact in binary floating
	// point. A device that disagreed at all would mean the two are seeing
	// different logits, not that one rounded differently.
	int mismatches = 0;
	for (size_t i = 0; i < cpuActions.size(); i++) {
		for (size_t e = 0; e < Action::ELEM_AMOUNT; e++) {
			if (cpuActions[i][e] != gpuActions[i][e]) {
				mismatches++;
				MESSAGE("car " << i << " element " << e << ": cpu "
							   << cpuActions[i][e] << " vs gpu "
							   << gpuActions[i][e]);
				break;
			}
		}
	}
	CHECK(mismatches == 0);

	// A model that returned a constant would pass the comparison above while
	// telling us nothing, so confirm the batch actually varies.
	bool anyDifference = false;
	for (size_t i = 1; i < cpuActions.size() && !anyDifference; i++)
		for (size_t e = 0; e < Action::ELEM_AMOUNT; e++)
			if (cpuActions[i][e] != cpuActions[0][e]) {
				anyDifference = true;
				break;
			}
	CHECK(anyDifference);
}
