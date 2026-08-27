#pragma once

#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

namespace Dash {

class TimeoutCondition : public RLGC::TerminalCondition {
public:
	float maxTime;
	float elapsed = 0.f;

	explicit TimeoutCondition(float maxTime) : maxTime(maxTime) {}

	virtual void Reset(const RLGC::GameState &initialState) override {
		elapsed = 0.f;
	}

	// deltaTime is the per-step delta, not elapsed episode time.
	virtual bool IsTerminal(const RLGC::GameState &currentState) override {
		elapsed += currentState.deltaTime;
		return elapsed >= maxTime;
	}

	virtual bool IsTruncation() override { return true; }
};

} // namespace Dash
