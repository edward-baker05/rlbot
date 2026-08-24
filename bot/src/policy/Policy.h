#pragma once

#include <GigaLearnCPP/Util/InferUnit.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace Dash {

struct ModelShape {
	std::vector<int> sharedHeadLayers = {1024, 1024, 512};
	std::vector<int> policyLayers = {512};
	GGL::ModelActivationType activation = GGL::ModelActivationType::RELU;
	bool addLayerNorm = true;
};

class Policy {
  public:
	Policy(RLGC::ObsBuilder *obsBuilder, int obsSize,
		   RLGC::ActionParser *actionParser, const ModelShape &shape,
		   bool useGPU);

	Policy(const Policy &) = delete;
	Policy &operator=(const Policy &) = delete;

	void Load(const std::filesystem::path &checkpointFolder);
	bool Loaded() const { return unit != nullptr; }

	std::vector<RLGC::Action>
	InferBatch(const std::vector<RLGC::Player> &players,
			   const std::vector<RLGC::GameState> &states, bool deterministic,
			   float temperature = 1.f);

  private:
	RLGC::ObsBuilder *obsBuilder;
	int obsSize;
	RLGC::ActionParser *actionParser;
	ModelShape shape;
	bool useGPU;
	std::unique_ptr<GGL::InferUnit> unit;
};

} // namespace Dash
