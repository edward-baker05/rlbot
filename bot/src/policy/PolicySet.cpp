#include "PolicySet.h"

#include <stdexcept>

using namespace RLGC;

namespace Hive {

PolicySet::PolicySet(ObsBuilder* obsBuilder,
                     int obsSize,
                     ActionParser* actionParser,
                     const ModelShape& shape,
                     bool useGPU)
	: obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser),
	  shape(shape), useGPU(useGPU) {}

PolicySet::~PolicySet() = default;

std::unique_ptr<GGL::InferUnit> PolicySet::Make(const std::filesystem::path& folder, const char* what) {
	if (!std::filesystem::is_directory(folder)) {
		throw std::runtime_error(
			std::string("PolicySet: ") + what + " checkpoint folder does not exist: " + folder.string());
	}

	// GigaLearn writes each sub-model as an uppercased <NAME>.lt file. Checking
	// here turns an opaque libtorch failure into an actionable message.
	bool anyModelFile = false;
	for (const auto& entry : std::filesystem::directory_iterator(folder)) {
		if (entry.path().extension() == ".lt") {
			anyModelFile = true;
			break;
		}
	}
	if (!anyModelFile) {
		throw std::runtime_error(
			std::string("PolicySet: no .lt model files in ") + folder.string() +
			" (expected a GigaLearn checkpoint folder, e.g. checkpoints/general/50000000)");
	}

	GGL::PartialModelConfig sharedCfg = {};
	sharedCfg.layerSizes = shape.sharedHeadLayers;
	sharedCfg.activationType = shape.activation;
	sharedCfg.addLayerNorm = shape.addLayerNorm;
	sharedCfg.addOutputLayer = false; // Shared head feeds the policy, no logits

	GGL::PartialModelConfig policyCfg = {};
	policyCfg.layerSizes = shape.policyLayers;
	policyCfg.activationType = shape.activation;
	policyCfg.addLayerNorm = shape.addLayerNorm;

	return std::make_unique<GGL::InferUnit>(
		obsBuilder, obsSize, actionParser,
		sharedCfg, policyCfg, folder, useGPU);
}

void PolicySet::LoadGeneral(const std::filesystem::path& folder) {
	general = Make(folder, "general");
}

void PolicySet::LoadKickoff(const std::filesystem::path& folder) {
	kickoff = Make(folder, "kickoff");
}

Action PolicySet::Infer(const Player& player,
                        const GameState& state,
                        Regime regime,
                        bool deterministic,
                        float temperature) {
	return InferBatch({player}, {state}, {regime}, deterministic, temperature)[0];
}

std::vector<Action> PolicySet::InferBatch(const std::vector<Player>& players,
                                          const std::vector<GameState>& states,
                                          const std::vector<Regime>& regimes,
                                          bool deterministic,
                                          float temperature) {
	if (players.size() != states.size() || players.size() != regimes.size())
		throw std::runtime_error("PolicySet::InferBatch(): mismatched input sizes");

	if (!general)
		throw std::runtime_error("PolicySet::InferBatch(): no general model loaded");

	std::vector<Action> results(players.size());
	if (players.empty())
		return results;

	// Bucket cars by the model that will drive them. Without a kickoff model
	// everything falls through to the general bucket.
	std::vector<size_t> kickoffIdx, generalIdx;
	for (size_t i = 0; i < players.size(); i++) {
		if (regimes[i] == Regime::Kickoff && kickoff)
			kickoffIdx.push_back(i);
		else
			generalIdx.push_back(i);
	}

	auto runBucket = [&](GGL::InferUnit* unit, const std::vector<size_t>& idx) {
		if (idx.empty())
			return;

		std::vector<Player> bPlayers;
		std::vector<GameState> bStates;
		bPlayers.reserve(idx.size());
		bStates.reserve(idx.size());
		for (size_t i : idx) {
			bPlayers.push_back(players[i]);
			bStates.push_back(states[i]);
		}

		auto actions = unit->BatchInferActions(bPlayers, bStates, deterministic, temperature);
		for (size_t k = 0; k < idx.size(); k++)
			results[idx[k]] = actions[k];
	};

	runBucket(kickoff.get(), kickoffIdx);
	runBucket(general.get(), generalIdx);

	return results;
}

} // namespace Hive
