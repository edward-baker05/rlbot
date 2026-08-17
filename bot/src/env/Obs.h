#pragma once

#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <memory>

namespace Hive {

// ============================================================================
// Observation
// ============================================================================
// The bot uses RLGymCPP's DefaultObsPadded, which is what makes a single policy
// work across 1s, 2s and 3s -- and, for free, across human players.
//
// How it satisfies the requirement:
//
//   * FIXED WIDTH. Teammate and opponent slots are padded with zeros up to
//     maxPlayersPerTeam, so the observation is the same size in 1s as in 3s and
//     the network never has to be resized.
//
//   * SLOT SHUFFLING. Teammates and opponents are shuffled every step, so the
//     policy cannot learn "the car in slot 2 is the good one". This is what
//     lets it treat an unfamiliar teammate -- a human, a different bot, a
//     demoed car -- as just another occupant of a slot.
//
//   * TEAM INVERSION. Everything is mirrored for orange, so the policy always
//     plays "towards +Y" and both teams share one set of weights.
//
// Humans need NO special handling anywhere in the codebase. From the packet's
// point of view a human is a car with a position and a velocity, and the padded
// observation cannot tell whether a slot is filled by a human, a bot, or
// nothing at all. That is the property that makes "humans in any number on any
// team" fall out for free rather than needing a code path.
//
// One consequence worth knowing: an empty slot is all zeros, which is not a
// physically reachable car state (a real car always has a non-zero rotation
// basis), so the network can distinguish "empty" from "car at the origin"
// without an explicit occupancy flag.
// ============================================================================

// Create the observation builder. Caller owns the result.
//
// maxPlayersPerTeam must match between training and deployment. It is baked
// into the observation width and therefore into the network's input layer.
std::unique_ptr<RLGC::DefaultObsPadded> MakeObsBuilder(int maxPlayersPerTeam);

// Measure the observation width by building one.
//
// Computing this by hand from the field layout is possible but fragile -- it
// silently breaks if RLGymCPP changes what DefaultObs emits per car, and the
// symptom is a shape mismatch deep inside libtorch. Building a throwaway arena
// and asking the builder how many floats it produced cannot drift.
//
// Requires RocketSim::Init() to have been called first.
int ProbeObsSize(int maxPlayersPerTeam);

} // namespace Hive
