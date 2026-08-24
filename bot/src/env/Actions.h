#pragma once

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <memory>

namespace Dash {

class UnmaskedAction : public RLGC::DefaultAction {
public:
	std::vector<uint8_t> GetActionMask(const RLGC::Player& player,
	                                   const RLGC::GameState& state) override {
		return std::vector<uint8_t>(GetActionAmount(), true);
	}
};

std::unique_ptr<RLGC::DefaultAction> MakeActionParser(bool masked);

}  // namespace Dash
