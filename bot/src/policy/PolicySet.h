#pragma once

#include "Regime.h"

#include <GigaLearnCPP/Util/InferUnit.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Hive {

// Shape of a policy network, needed to rebuild it for inference.
//
// These MUST match the values the model was trained with. Load a checkpoint
// with the wrong layer sizes and libtorch will either throw a shape error or --
// worse, if the sizes happen to line up -- load silently and play badly.
//
// This struct is the single source of truth for both sides: TrainConfig and
// BotSettings each hold a default-constructed ModelShape, so editing the
// defaults here changes training and deployment together. Do not duplicate
// these numbers anywhere else.
//
// Changing them invalidates every existing checkpoint. Start a fresh run.
struct ModelShape {
	std::vector<int> sharedHeadLayers = {512, 512};
	std::vector<int> policyLayers = {512, 512, 512};
	GGL::ModelActivationType activation = GGL::ModelActivationType::RELU;
	bool addLayerNorm = true;
};

// ============================================================================
// PolicySet -- holds the kickoff model and the general model
// ============================================================================
// Both policies share one obs builder and one action parser. That is not an
// implementation shortcut, it is the property that makes the split safe: the
// two models are interchangeable at any step because they read the same
// observation and emit the same action space. Only the weights differ.
//
// A kickoff model is optional. With only a general model loaded, the set routes
// everything to it and the bot behaves exactly like a single-policy bot -- which
// is what you should ship first, before a kickoff model exists.
// ============================================================================

class PolicySet {
public:
	// obsBuilder and actionParser are borrowed; the caller keeps them alive.
	PolicySet(RLGC::ObsBuilder* obsBuilder,
	          int obsSize,
	          RLGC::ActionParser* actionParser,
	          const ModelShape& shape,
	          bool useGPU);

	~PolicySet();

	PolicySet(const PolicySet&) = delete;
	PolicySet& operator=(const PolicySet&) = delete;

	// Load the general policy. Required. Throws std::runtime_error on failure.
	void LoadGeneral(const std::filesystem::path& checkpointFolder);

	// Load the kickoff policy. Optional -- without it, kickoffs are driven by
	// the general model.
	void LoadKickoff(const std::filesystem::path& checkpointFolder);

	bool HasKickoffModel() const { return kickoff != nullptr; }
	bool HasGeneralModel() const { return general != nullptr; }

	// ---- Inference ---------------------------------------------------------
	// Cars are grouped by regime so each model runs one batched forward pass.
	// On a 6 GB card, per-car inference is dominated by kernel launch overhead,
	// so batching a 3v3 hivemind into one call is a real saving.
	//
	// `regimes` must be the same length as `players` and `states`.
	std::vector<RLGC::Action> InferBatch(const std::vector<RLGC::Player>& players,
	                                     const std::vector<RLGC::GameState>& states,
	                                     const std::vector<Regime>& regimes,
	                                     bool deterministic,
	                                     float temperature = 1.f);

	RLGC::Action Infer(const RLGC::Player& player,
	                   const RLGC::GameState& state,
	                   Regime regime,
	                   bool deterministic,
	                   float temperature = 1.f);

private:
	std::unique_ptr<GGL::InferUnit> Make(const std::filesystem::path& folder, const char* what);

	RLGC::ObsBuilder* obsBuilder;
	int obsSize;
	RLGC::ActionParser* actionParser;
	ModelShape shape;
	bool useGPU;

	std::unique_ptr<GGL::InferUnit> general;
	std::unique_ptr<GGL::InferUnit> kickoff;
};

} // namespace Hive
