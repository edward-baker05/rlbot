#pragma once

#include <GigaLearnCPP/Util/InferUnit.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace Hive {

// Shape of the policy network. Single source of truth for training and
// deployment: both sides default-construct this struct. Changing it
// invalidates every existing checkpoint.
struct ModelShape {
	std::vector<int> sharedHeadLayers = {512, 512};
	std::vector<int> policyLayers = {512, 512, 512};
	GGL::ModelActivationType activation = GGL::ModelActivationType::RELU;
	bool addLayerNorm = true;
};

class Policy {
public:
	// obsBuilder and actionParser are borrowed; the caller keeps them alive.
	Policy(RLGC::ObsBuilder* obsBuilder,
	       int obsSize,
	       RLGC::ActionParser* actionParser,
	       const ModelShape& shape,
	       bool useGPU);

	Policy(const Policy&) = delete;
	Policy& operator=(const Policy&) = delete;

	// Throws std::runtime_error on a missing/invalid checkpoint folder.
	void Load(const std::filesystem::path& checkpointFolder);
	bool Loaded() const { return unit != nullptr; }

	std::vector<RLGC::Action> InferBatch(const std::vector<RLGC::Player>& players,
	                                     const std::vector<RLGC::GameState>& states,
	                                     bool deterministic,
	                                     float temperature = 1.f);

private:
	RLGC::ObsBuilder* obsBuilder;
	int obsSize;
	RLGC::ActionParser* actionParser;
	ModelShape shape;
	bool useGPU;
	std::unique_ptr<GGL::InferUnit> unit;
};

} // namespace Hive
