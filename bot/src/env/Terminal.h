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
	// Small grace period so a car spawned in contact with the ball (which state
	// setters can occasionally produce) does not end the episode on step zero.
	float startGraceSeconds;

	// How long to keep the episode alive AFTER first contact.
	//
	// Without this the episode ends on the very step the ball is touched, and
	// the ball's post-hit velocity is never observed -- which makes "did the
	// kickoff go towards their goal" almost pure noise, despite being the thing
	// a kickoff is actually for. A short window is enough to register the
	// direction of the strike without straying into general play, which is not
	// this policy's job.
	float holdAfterTouchSeconds;

	explicit FirstTouchCondition(float holdAfterTouchSeconds = 0.3f,
	                             float startGraceSeconds = 0.05f)
		: startGraceSeconds(startGraceSeconds),
		  holdAfterTouchSeconds(holdAfterTouchSeconds) {}

	void Reset(const RLGC::GameState& initialState) override {
		elapsed = 0.f;
		sinceTouch = -1.f;
	}

	bool IsTerminal(const RLGC::GameState& currentState) override {
		elapsed += currentState.deltaTime;

		if (sinceTouch >= 0.f) {
			// Already touched; run out the hold window.
			sinceTouch += currentState.deltaTime;
			return sinceTouch >= holdAfterTouchSeconds;
		}

		if (elapsed < startGraceSeconds)
			return false;

		for (const auto& player : currentState.players) {
			if (player.ballTouchedStep) {
				sinceTouch = 0.f;
				// Hold for at least one more step so the post-hit ball state is
				// observed even if the window is shorter than a step.
				return false;
			}
		}
		return false;
	}

	bool IsTruncation() override { return true; }

private:
	float elapsed = 0.f;
	float sinceTouch = -1.f;
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
