#pragma once

#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

namespace Dash {

// One car Necto is asked to drive this step.
struct NectoRequest {
	const RLGC::GameState *state;
	int playerIdx; // Index into state->players.
	RLGC::Action prevAction;
};

// Runs the Necto bot as an opponent inside RocketSim arenas.
//
// Necto is not weight-compatible with anything else in this project: it has its
// own observation (a token matrix over ball/cars/boost pads, consumed by an
// attention model) and its own factored action head, so it cannot ride the
// old-version self-play path and cannot use GGL::InferUnit. This class is the
// whole bridge.
//
// The model file is TorchScript, so libtorch loads it directly -- no port of
// the network itself. Verified against libtorch 2.13.
class NectoPolicy {
  public:
	// Feature columns per token, and the query width (24 features + 8 previous
	// action values). Both are fixed by the trained model.
	static constexpr int FEATURES = 24;
	static constexpr int Q_SIZE = FEATURES + RLGC::Action::ELEM_AMOUNT;

	// beta shapes the action distribution, matching necto/agent.py:
	//   1.0  argmax (deterministic)
	//   0.5  exactly on-policy sampling -- the logit scale works out to 1.0
	//   0.0  uniform random
	// Training uses 0.5: a deterministic opponent is memorisable, and memorising
	// it is the failure mode this is meant to avoid. The benchmark uses 1.0,
	// which wants the opposite -- low variance.
	NectoPolicy(const std::filesystem::path &modelPath, float beta,
				int64_t seed);
	~NectoPolicy();

	NectoPolicy(const NectoPolicy &) = delete;
	NectoPolicy &operator=(const NectoPolicy &) = delete;

	// One forward pass for every request, so the whole training step costs a
	// single batched inference rather than one per arena.
	//
	// Requests are grouped by token count before batching. Padding ragged
	// batches would be near-exact but not exact (the traced model's mask is not
	// fully wired -- padded rows still perturb the logits slightly), and
	// grouping costs nothing: at 1v1 every arena has the same token count, so
	// this is one forward.
	void InferBatch(const std::vector<NectoRequest> &requests,
					std::vector<RLGC::Action> &outActions);

	// Builds the (q, kv) observation for one car. Exposed for the self-test that
	// diffs it against the Python NectoObsBuilder -- a silent mismatch here
	// makes Necto play badly while everything still runs.
	static void BuildObs(const RLGC::GameState &state, int playerIdx,
						 const RLGC::Action &prevAction, float *qOut,
						 float *kvOut);

	// 1 ball + cars + boost pads.
	static int TokenCount(const RLGC::GameState &state);

	float Beta() const { return beta; }

  private:
	// Decodes the model's five logit heads into a car control input.
	RLGC::Action SampleAction(const float *const *heads, const int *headSizes);

	struct Module;
	std::unique_ptr<Module> module;

	float beta;
	float logitScale;
	std::mt19937 rng;

	// Reused across steps so a training loop does not reallocate every tick.
	std::vector<float> qBuf, kvBuf, maskBuf;
};

} // namespace Dash
