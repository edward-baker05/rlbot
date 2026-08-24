#pragma once

#include "NectoArena.h"
#include "NectoPolicy.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace Dash {

// Drives every Necto-controlled car across the whole EnvSet.
//
// Fills the two hooks the patched learner exposes: one says which players the
// learner does not own, the other runs just before each step and queues the
// opponent's controls.
class NectoDriver {
  public:
	NectoDriver(const std::filesystem::path &modelPath, float beta,
				int64_t seed);

	// externalPlayerMaskFn. Flags every car on Necto's side, in the learner's
	// flat player indexing.
	void BuildMask(RLGC::EnvSet *envSet, std::vector<uint8_t> &outMask);

	// preStepFn. One batched forward for every Necto car in the set.
	void Step(RLGC::EnvSet *envSet);

	int NumControlledCars() const { return numControlledCars; }

  private:
	// Where a queued action belongs, so the scatter matches the gather.
	struct Slot {
		NectoArenaState *arena;
		int playerIdx;
	};

	NectoPolicy policy;

	// Reused every step; never reallocated once the run is warm.
	std::vector<NectoRequest> requests;
	std::vector<Slot> slots;
	std::vector<RLGC::Action> actions;

	int numControlledCars = 0;
};

} // namespace Dash
