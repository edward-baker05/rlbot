#pragma once

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <cstdint>
#include <memory>

namespace Hive {

// Which observation layout the policy sees. Lives here rather than in
// TrainConfig so the RLBot client can name it without pulling in the whole
// training config; both sides need it, because it determines the input layer.
enum class ObsMode {
	// DefaultObsPadded plus car-frame relative geometry. See RelativeObs.h.
	Relative,
	// RLGymCPP's DefaultObsPadded: absolute world-frame everything. Every run
	// up to and including p8ref used this.
	Default,
};

// Two observations, selected by ObsMode.
//
// `Default` is RLGymCPP's DefaultObsPadded: absolute, world-frame everything,
// with teammate/opponent slots zero-padded to maxPlayersPerTeam and shuffled
// each step, and orange mirrored so both teams share one policy. This is what
// every run up to and including p8ref used.
//
// `Relative` is that plus car-frame relative geometry for the ball and every
// other car -- see RelativeObs.h for why.
//
// Padding and shuffling also mean a slot cannot be told human from bot from
// empty except by content, so humans need no special handling anywhere.

// Create the observation builder. Caller owns the result. Both the mode and
// maxPlayersPerTeam must match between training and deployment: together they
// determine the network's input layer.
std::unique_ptr<RLGC::ObsBuilder> MakeObsBuilder(int maxPlayersPerTeam,
                                                 ObsMode mode);

// Non-finite observation values seen since the last call, and the total
// checked. Reset by reading.
//
// EXISTS BECAUSE p11boost died at 29.8M steps with a CUDA device-side assert
// ("probability tensor contains either inf, nan or element < 0"). Everything
// upstream was finite -- entropy 0.538, KL 0.0053, reward 0.0624, GAE/Returns
// STD 2.113, Critic/V All 2.72 -- and the only GAE outputs that went NaN were
// the two that depend on critic value predictions. A NaN in an observation
// produces exactly that signature and nothing else in the telemetry would show
// it. This turns a crash into a number.
struct ObsHealth {
	uint64_t checked = 0;
	uint64_t nonFinite = 0;
};
ObsHealth ConsumeObsHealth();

// Measure the observation width by building a throwaway arena and asking the
// builder how many floats it produces, rather than computing it by hand (which
// would silently drift if the layout changes).
// Requires RocketSim::Init() to have been called first.
int ProbeObsSize(int maxPlayersPerTeam, ObsMode mode);

} // namespace Hive
