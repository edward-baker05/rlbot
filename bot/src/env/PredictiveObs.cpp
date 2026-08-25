#include "PredictiveObs.h"

#include <RLGymCPP/Gamestates/StateUtil.h>

#include <cmath>

using namespace RLGC;

namespace Dash {

void PredictiveObs::Reset(const GameState& initialState) {
	AdvancedObsPadded::Reset(initialState);
	predictor.Reset();
}

FList PredictiveObs::BuildObs(const Player& player, const GameState& state) {
	FList result = AdvancedObsPadded::BuildObs(player, state);

	const bool inv = player.team == Team::ORANGE;

	// Predict in world space once per arena tick -- both cars in a 1v1 share
	// this call, and the cache makes the second one free -- then invert per
	// player, the same way every other block in this obs is built.
	const BallTrajectory& traj = predictor.Get(state);

	const PhysState self = InvertPhys(player, inv);

	for (int k = 0; k < BallPredictor::NUM_SAMPLES; k++) {
		PhysState slice = {};
		slice.pos = traj.pos[BallPredictor::SAMPLE_TICKS[k]];
		slice.vel = traj.vel[BallPredictor::SAMPLE_TICKS[k]];
		const PhysState predicted = InvertPhys(slice, inv);

		// Car-local, matching AdvancedObs::AddPlayerToObs's convention for the
		// current ball position.
		result += self.rotMat.Dot(predicted.pos - self.pos) * POS_COEF;
	}

	// --- Event features ---

	if (traj.bounceTick >= 0) {
		const float bounceTime = (float)traj.bounceTick / TIME_NORM_TICKS;
		result += bounceTime > 1.f ? 1.f : bounceTime;

		PhysState bounce = {};
		bounce.pos = traj.bouncePos;
		result += InvertPhys(bounce, inv).pos * POS_COEF;
	} else {
		result += 1.f;                  // saturated: no bounce in horizon
		result += Vec(0, 0, 0);         // no position to report
	}

	// Only report goals inside the sample horizon; past that the prediction is
	// counterfactual enough that a confident goal flag would be a lie.
	const bool goalInHorizon =
		traj.goalTick >= 0 &&
		(float)traj.goalTick <= TIME_NORM_TICKS;

	if (goalInHorizon) {
		// traj.goalTeam is 1 for orange's net. From blue's perspective that is
		// scoring (+1); inverting flips it, so orange sees its own +1 the same
		// way.
		const float scoring = (traj.goalTeam == 1) ? 1.f : -1.f;
		result += inv ? -scoring : scoring;
		result += (float)traj.goalTick / TIME_NORM_TICKS;
	} else {
		result += 0.f;
		result += 1.f;
	}

	// A non-finite value silently poisons the whole batch; the existing
	// RelativeObs guards the same way.
	for (float& v : result) {
		if (!std::isfinite(v))
			v = 0.f;
	}

	return result;
}

}  // namespace Dash
