#pragma once

#include "PredictiveObs.h"

namespace Dash {

// PredictiveObs plus boost pads the network can actually locate.
//
// The inherited block reports 34 availability scalars in fixed index order with
// no geometry, so "an available pad 800uu to my left" is only representable by
// first learning 34 index->position associations. This appends car-local
// positions: all six big pads, always, plus the nearest small pads.
//
// The 34 legacy scalars are left in place. They are redundant now, but keeping
// them makes this a pure append, which is what lets migrate-obs carry an
// existing run across instead of restarting it.
class PadGeometryObs : public PredictiveObs {
public:
	// Indices into CommonValues::BOOST_LOCATIONS, which is symmetric under the
	// 180-degree inversion, so pad slot i sits at BOOST_LOCATIONS[i] in the
	// player's own frame whichever team they are on.
	static constexpr int BIG_PADS[6] = {3, 4, 15, 18, 29, 30};

	// Ranked by distance from the car, not by availability: a set that
	// reshuffled as pads respawned would reintroduce the slot churn this obs
	// exists to remove.
	static constexpr int NEAREST_SMALL = 6;

	// (position + availability) per reported pad.
	static constexpr int PAD_BLOCK =
		(6 + NEAREST_SMALL) * 4;

	explicit PadGeometryObs(int maxPlayers = 3) : PredictiveObs(maxPlayers) {}

	RLGC::FList BuildObs(const RLGC::Player& player,
	                     const RLGC::GameState& state) override;
};

}  // namespace Dash
