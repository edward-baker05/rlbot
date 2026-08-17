#pragma once

#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

namespace Hive {

// ============================================================================
// FirstTouchCondition
// ============================================================================
// Ends the episode as soon as any car touches the ball. This is what makes
// kickoff training a well-posed problem rather than a slow way to train a
// general policy.
//
// The kickoff model's job finishes at first touch -- that is exactly where the
// deployed bot hands control to the general model (see policy/Regime.h). If
// episodes ran past that point, the kickoff model would be credited for play it
// will never actually perform, and it would learn to optimise a trajectory that
// something else is going to fly.
//
// Truncating rather than terminating matters: a truncation tells PPO the
// episode was cut short by the harness, so it bootstraps the value estimate
// from the final state instead of assuming a return of zero. Without that, the
// critic would learn that every kickoff ends in nothing.
class FirstTouchCondition : public RLGC::TerminalCondition {
public:
	// Small grace period so a car spawned in contact with the ball (which
	// state setters can occasionally produce) does not end the episode on
	// step zero.
	float graceSeconds;

	explicit FirstTouchCondition(float graceSeconds = 0.05f)
		: graceSeconds(graceSeconds) {}

	void Reset(const RLGC::GameState& initialState) override {
		elapsed = 0.f;
	}

	bool IsTerminal(const RLGC::GameState& currentState) override {
		elapsed += currentState.deltaTime;
		if (elapsed < graceSeconds)
			return false;

		for (const auto& player : currentState.players) {
			if (player.ballTouchedStep)
				return true;
		}
		return false;
	}

	bool IsTruncation() override { return true; }

private:
	float elapsed = 0.f;
};

// ============================================================================
// TimeoutCondition
// ============================================================================
// Plain wall-clock cap on episode length. NoTouchCondition only fires when
// nobody touches the ball at all, which a policy can game by tapping the ball
// occasionally while making no progress. This bounds the episode regardless.
class TimeoutCondition : public RLGC::TerminalCondition {
public:
	float maxSeconds;

	explicit TimeoutCondition(float maxSeconds) : maxSeconds(maxSeconds) {}

	void Reset(const RLGC::GameState& initialState) override {
		elapsed = 0.f;
	}

	bool IsTerminal(const RLGC::GameState& currentState) override {
		elapsed += currentState.deltaTime;
		return elapsed >= maxSeconds;
	}

	bool IsTruncation() override { return true; }

private:
	float elapsed = 0.f;
};

} // namespace Hive
