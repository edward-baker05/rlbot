#pragma once

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <cstdint>

namespace Dash {

class RelativeObs : public RLGC::ObsBuilder {
public:
	static constexpr float RELATIVE_POS_SCALE = RLGC::CommonValues::BACK_WALL_Y;

	explicit RelativeObs(int maxPlayers = 1) : maxPlayers(maxPlayers) {}

	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;

	static constexpr int RELATIVE_BLOCK = 10;

private:
	int maxPlayers;
};

}  // namespace Dash
