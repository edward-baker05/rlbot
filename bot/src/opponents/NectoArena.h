#pragma once

#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/Gamestates/Player.h>

#include <cmath>
#include <vector>

namespace Dash {

// Which arenas Necto plays in, and on which side.
//
// Hung off EnvCreateResult::userInfo, so the action parser (which injects the
// controls) and the training driver (which computes them) reach the same
// object without either knowing about the other.
struct NectoArenaState {
	bool active = false;
	Team nectoTeam = Team::ORANGE;

	// Controls for the current step, indexed by player index within the arena.
	// Written by the driver before StepSecondHalf, read by the parser during it.
	std::vector<RLGC::Action> pending;
};

// The assignment is STATIC for the whole run, and exactly half the Necto arenas
// put Necto on each side.
//
// Static matters for correctness, not just simplicity: the learner splits
// players into "mine" and "not mine" once per iteration, so an assignment that
// changed mid-iteration -- per episode, say -- would leave that split stale and
// silently train on Necto's actions. Fixing the sides also makes the symmetry
// exact rather than something that averages out over enough random draws, which
// is the failure this project has already been bitten by.
inline bool NectoArenaAssignment(int arenaIndex, float fraction,
								 Team *outTeam) {
	if (fraction <= 0.f || arenaIndex < 0)
		return false;

	// One arena in every `stride` gets Necto.
	const int stride =
		(fraction >= 1.f) ? 1 : static_cast<int>(std::lround(1.f / fraction));
	if (stride <= 0 || arenaIndex % stride != 0)
		return false;

	if (outTeam) {
		const int nth = arenaIndex / stride;
		*outTeam = (nth % 2 == 0) ? Team::ORANGE : Team::BLUE;
	}
	return true;
}

} // namespace Dash
