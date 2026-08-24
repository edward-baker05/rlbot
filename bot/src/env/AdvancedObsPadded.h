#pragma once

#include <RLGymCPP/ObsBuilders/AdvancedObs.h>

namespace Dash {

class AdvancedObsPadded : public RLGC::AdvancedObs {
public:
	int maxPlayers;

	explicit AdvancedObsPadded(int maxPlayers = 3) : maxPlayers(maxPlayers) {}

	virtual RLGC::FList BuildObs(const RLGC::Player& player,
	                             const RLGC::GameState& state) override;
};

}  // namespace Dash
