#include "StateSetters.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>

using namespace RLGC;
using RocketSim::Math::RandFloat;

namespace Dash {

void InfiniteBoostState::ResetArena(Arena* arena) {
	inner->ResetArena(arena);

	lastWasInfinite = RandFloat() < chance;

	MutatorConfig cfg = arena->GetMutatorConfig();
	cfg.boostUsedPerSecond =
		lastWasInfinite ? 0.f : RLConst::BOOST_USED_PER_SECOND;
	arena->SetMutatorConfig(cfg);

	if (lastWasInfinite) {
		for (Car* car : arena->_cars) {
			CarState state = car->GetState();
			state.boost = 100;
			car->SetState(state);
		}
	}
}

}  // namespace Dash
