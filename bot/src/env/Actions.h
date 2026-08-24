#pragma once

#include "../opponents/NectoArena.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <memory>

namespace Dash {

// The standard 90-action lookup table, with two additions.
//
// Optional masking: `masked` false reports every action as available, which is
// the old UnmaskedAction behaviour.
//
// Optional Necto injection: when this arena has Necto in it, cars on Necto's
// team ignore the sampled action index and take the controls the Necto driver
// queued for this step. That is the whole injection mechanism -- EnvSet calls
// the parser once per car inside StepSecondHalf, so nothing in RLGymCPP has to
// change. It is needed because Necto's factored action head does not map onto
// this table: its ground actions set pitch from throttle, and its yaw and roll
// split one output on the handbrake bit.
class DashAction : public RLGC::DefaultAction {
  public:
	DashAction(bool masked, const NectoArenaState *nectoArena)
		: masked(masked), nectoArena(nectoArena) {}

	RLGC::Action ParseAction(int index, const RLGC::Player &player,
							 const RLGC::GameState &state) override {
		if (nectoArena && nectoArena->active &&
			player.team == nectoArena->nectoTeam) {
			const int i = player.index;
			if (i >= 0 && i < static_cast<int>(nectoArena->pending.size()))
				return nectoArena->pending[i];

			// Nothing queued for this car. Coast rather than fall through to the
			// sampled index, which would hand a learner action to the opponent
			// and corrupt the comparison without any visible error.
			return RLGC::Action{};
		}
		return RLGC::DefaultAction::ParseAction(index, player, state);
	}

	std::vector<uint8_t> GetActionMask(const RLGC::Player &player,
									   const RLGC::GameState &state) override {
		if (!masked)
			return std::vector<uint8_t>(GetActionAmount(), true);
		return RLGC::DefaultAction::GetActionMask(player, state);
	}

  private:
	bool masked;
	const NectoArenaState *nectoArena;
};

std::unique_ptr<RLGC::DefaultAction>
MakeActionParser(bool masked, const NectoArenaState *nectoArena = nullptr);

} // namespace Dash
