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

// Which member of the Necto family a model file holds.
//
// They share a trunk (EARL over the same 24-column token matrix) and the same
// normalization, field inversion and q layout, so most of this class is common.
// They differ in exactly three places, and all three are silent if you get them
// wrong -- the model still runs and just plays badly:
//
//   * ACTION HEAD. Necto emits five factored logit heads (throttle, steer,
//     jump, boost, handbrake) that are combined into controls. Nexto emits one
//     90-way categorical over a lookup table of whole control sets.
//   * RELATIVE PASS. Necto makes kv positions AND linear velocities relative to
//     the main car. Nexto subtracts position only, then rotates every x/y pair
//     -- position, velocity, forward, up, angular velocity -- into the main
//     car's yaw frame.
//   * COLUMN 21. Necto reads real timers (demo respawn, boost pad cooldown).
//     Nexto was trained through `NextoObsBuilder.batched_build_obs`, which puts
//     the raw binary `is_demoed` and pad-availability flags there instead. Those
//     are marked FIXME upstream, but the same function ran at training time, so
//     the flags are what the weights actually expect.
//
// Token ORDER also differs (Nexto puts cars before the ball) and is reproduced
// faithfully below, even though it cannot matter: the traced EARLPerceiver has
// no positional encoding, so kv is a set. Matching it anyway keeps the self-test
// an exact elementwise diff rather than one with a permutation baked into it.
enum class NectoFamily { Necto, Nexto };

const char *NectoFamilyName(NectoFamily family);

// Runs a Necto-family bot (Necto or Nexto) as an opponent inside RocketSim
// arenas.
//
// Neither is weight-compatible with anything else in this project: they have
// their own observation (a token matrix over ball/cars/boost pads, consumed by
// an attention model) and their own action heads, so they cannot ride the
// old-version self-play path and cannot use GGL::InferUnit. This class is the
// whole bridge.
//
// The model file is TorchScript, so libtorch loads it directly -- no port of
// the network itself. Verified against libtorch 2.13.
//
// Which family a file holds is detected from the traced forward's return type
// rather than configured, so pointing DASH_NECTO_MODEL at either one is all it
// takes. See NectoFamily for what actually differs.
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
	//
	// useGPU puts the model on CUDA when one is available. During training that
	// is what you want: this forward runs in preStepFn, which is serial on the
	// collection thread with every worker idle, and the GPU measures ~11% busy
	// through collection. It falls back to CPU silently if CUDA is absent, so
	// callers that must not touch the GPU should pass false rather than rely on
	// that.
	NectoPolicy(const std::filesystem::path &modelPath, float beta,
				int64_t seed, bool useGPU);
	~NectoPolicy();

	// Where the model actually ended up -- false when the GPU was asked for and
	// CUDA was not available.
	bool OnGPU() const { return onGPU; }

	// Which family the loaded file turned out to hold.
	NectoFamily Family() const { return family; }

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

	// Builds the (q, kv) observation for one car. Exposed for the self-tests that
	// diff it against the real Python NectoObsBuilder / NextoObsBuilder -- a
	// silent mismatch here makes the opponent play badly while everything still
	// runs, which is the most expensive failure mode this bridge has.
	static void BuildObs(NectoFamily family, const RLGC::GameState &state,
						 int playerIdx, const RLGC::Action &prevAction,
						 float *qOut, float *kvOut);

	// 1 ball + cars + boost pads. Same count for both families; only the row
	// order differs.
	static int TokenCount(const RLGC::GameState &state);

	// Nexto's 90 control sets, in the order its head indexes them. Exposed for
	// the self-test: an off-by-one or a reordered loop here is invisible in the
	// observation and produces a Nexto that plays a scrambled version of what it
	// decided, which reads as "weaker than expected" rather than as a bug.
	static std::vector<RLGC::Action> NextoLookupTable();

	float Beta() const { return beta; }

  private:
	// Necto: decodes the five factored logit heads into a car control input.
	RLGC::Action SampleAction(const float *const *heads, const int *headSizes);

	// Nexto: decodes one 90-way categorical into a car control input.
	RLGC::Action SampleLookupAction(const float *logits, int count);

	struct Module;
	std::unique_ptr<Module> module;

	NectoFamily family = NectoFamily::Necto;
	float beta;
	float logitScale;
	bool onGPU = false;
	std::mt19937 rng;

	// Nexto's action space, empty for Necto. Index into this is exactly the
	// index the model emits.
	std::vector<RLGC::Action> lookupTable;

	// Reused across steps so a training loop does not reallocate every tick.
	std::vector<float> qBuf, kvBuf, maskBuf, logitBuf;
};

} // namespace Dash
