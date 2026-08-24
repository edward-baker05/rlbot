#pragma once

#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

namespace Dash {

class TimeoutCondition : public RLGC::TerminalCondition {
public:
	float maxTime;

	explicit TimeoutCondition(float maxTime) : maxTime(maxTime) {}

	virtual void Reset(const RLGC::GameState &initialState) override {}

	virtual bool IsTerminal(const RLGC::GameState &currentState) override {
		return currentState.deltaTime >= maxTime;
	}

	virtual bool IsTruncation() override { return true; }
};

} // namespace Dash
