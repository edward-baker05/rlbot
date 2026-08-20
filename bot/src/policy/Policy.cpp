#include "Policy.h"

#include <stdexcept>

using namespace RLGC;

namespace Hive {

Policy::Policy(ObsBuilder* obsBuilder,
               int obsSize,
               ActionParser* actionParser,
               const ModelShape& shape,
               bool useGPU)
	: obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser),
	  shape(shape), useGPU(useGPU) {}

void Policy::Load(const std::filesystem::path& folder) {
	if (!std::filesystem::is_directory(folder)) {
		throw std::runtime_error(
			std::string("Policy: checkpoint folder does not exist: ") + folder.string());
	}

	bool anyModelFile = false;
	for (const auto& entry : std::filesystem::directory_iterator(folder)) {
		if (entry.path().extension() == ".lt") {
			anyModelFile = true;
			break;
		}
	}
	if (!anyModelFile) {
		throw std::runtime_error(
			std::string("Policy: no .lt model files in ") + folder.string() +
			" (expected a GigaLearn checkpoint folder, e.g. checkpoints/main/50000000)");
	}

	GGL::PartialModelConfig sharedCfg = {};
	sharedCfg.layerSizes = shape.sharedHeadLayers;
	sharedCfg.activationType = shape.activation;
	sharedCfg.addLayerNorm = shape.addLayerNorm;
	sharedCfg.addOutputLayer = false;

	GGL::PartialModelConfig policyCfg = {};
	policyCfg.layerSizes = shape.policyLayers;
	policyCfg.activationType = shape.activation;
	policyCfg.addLayerNorm = shape.addLayerNorm;

	unit = std::make_unique<GGL::InferUnit>(
		obsBuilder, obsSize, actionParser,
		sharedCfg, policyCfg, folder, useGPU);
}

std::vector<Action> Policy::InferBatch(const std::vector<Player>& players,
                                       const std::vector<GameState>& states,
                                       bool deterministic,
                                       float temperature) {
	if (players.size() != states.size())
		throw std::runtime_error("Policy::InferBatch(): mismatched input sizes");
	if (!unit)
		throw std::runtime_error("Policy::InferBatch(): no model loaded");
	if (players.empty())
		return {};
	return unit->BatchInferActions(players, states, deterministic, temperature);
}

}  // namespace Hive
