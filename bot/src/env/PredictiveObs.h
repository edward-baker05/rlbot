#pragma once

#include "AdvancedObsPadded.h"
#include "BallPredictor.h"

namespace Dash {

// AdvancedObsPadded plus where the ball is going.
//
// The policy is a memoryless MLP seeing one ball snapshot per step. Future ball
// position is computable from that snapshot plus fixed geometry, so this adds
// no information -- but the function is piecewise (bounces are
// discontinuities), which is exactly what an MLP approximates badly. Supplying
// it turns a hard approximation problem into a lookup.
class PredictiveObs : public AdvancedObsPadded {
public:
	// 6 samples x 3 (car-local position) + 6 event features.
	static constexpr int PREDICT_BLOCK = BallPredictor::NUM_SAMPLES * 3 + 6;

	// Event times are normalized against the deepest sample, not the 6s
	// simulation horizon -- the extra simulation is a caching buffer, not part
	// of the feature space.
	static constexpr float TIME_NORM_TICKS =
		(float)BallPredictor::SAMPLE_TICKS.back();

	explicit PredictiveObs(int maxPlayers = 3) : AdvancedObsPadded(maxPlayers) {}

	void Reset(const RLGC::GameState& initialState) override;

	RLGC::FList BuildObs(const RLGC::Player& player,
	                     const RLGC::GameState& state) override;

	// Full trajectory simulations run so far. For the performance gate: the
	// amortized cost is entirely a function of how often the cache misses.
	uint64_t SimulationCount() const { return predictor.SimulationCount(); }

private:
	BallPredictor predictor;
};

}  // namespace Dash
