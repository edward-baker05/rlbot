#pragma once

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>

#include <cstdint>

namespace Hive {

void NoteObsHealth(uint64_t checked, uint64_t nonFinite);

class RelativeObs : public RLGC::ObsBuilder {
public:
	static constexpr float RELATIVE_POS_SCALE = RLGC::CommonValues::BACK_WALL_Y;

	explicit RelativeObs(int maxPlayers) : maxPlayers(maxPlayers) {}

	RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override;

	static constexpr int RELATIVE_BLOCK = 10;

private:
	int maxPlayers;
};

}  // namespace Hive
