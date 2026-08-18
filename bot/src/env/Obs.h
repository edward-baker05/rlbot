#pragma once

#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <memory>

namespace Hive {

// Uses RLGymCPP's DefaultObsPadded: teammate/opponent slots are zero-padded to
// maxPlayersPerTeam and shuffled each step, and orange is mirrored so both
// teams share one policy. Padding + shuffling also means a slot cannot be
// told human from bot from empty except by content, so humans need no special
// handling anywhere in this codebase.

// Create the observation builder. Caller owns the result. maxPlayersPerTeam
// must match between training and deployment: it is baked into the network's
// input layer.
std::unique_ptr<RLGC::DefaultObsPadded> MakeObsBuilder(int maxPlayersPerTeam);

// Measure the observation width by building a throwaway arena and asking the
// builder how many floats it produces, rather than computing it by hand (which
// would silently drift if RLGymCPP changes what DefaultObs emits per car).
// Requires RocketSim::Init() to have been called first.
int ProbeObsSize(int maxPlayersPerTeam);

} // namespace Hive
