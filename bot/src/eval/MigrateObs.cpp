#include "MigrateObs.h"
#include "Checkpoints.h"

#include <private/GigaLearnCPP/Util/Models.h>

#include <torch/torch.h>

#include <cstdio>
#include <fstream>

namespace Dash {

namespace {

GGL::ModelConfig MakeHeadConfig(int numInputs, const ModelShape& shape) {
	// Must match PPOLearner::MakeModels exactly, or the loaded parameter
	// shapes will not line up with the saved ones.
	GGL::PartialModelConfig partial = {};
	partial.layerSizes = shape.sharedHeadLayers;
	partial.activationType = shape.activation;
	partial.addLayerNorm = shape.addLayerNorm;
	partial.addOutputLayer = false;

	GGL::ModelConfig config = partial;
	config.numInputs = numInputs;
	config.numOutputs = 0;
	return config;
}

}  // namespace

MigrateResult MigrateSharedHead(const std::filesystem::path& srcFolder,
                                const std::filesystem::path& dstFolder,
                                int oldObsSize,
                                int newObsSize,
                                const ModelShape& shape) {
	RG_NO_GRAD;

	if (newObsSize < oldObsSize)
		return {false, "new obs size is smaller than the old one; this tool "
		               "only widens"};

	if (!std::filesystem::exists(srcFolder / "SHARED_HEAD.lt"))
		return {false, "no SHARED_HEAD.lt in " + srcFolder.string()};

	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObsSize, shape),
	                   torch::kCPU);
	try {
		oldHead.Load(srcFolder, /*allowNotExist=*/false, /*loadOptim=*/false);
	} catch (const std::exception& e) {
		return {false, std::string("failed to load old head: ") + e.what()};
	}

	GGL::Model newHead("shared_head", MakeHeadConfig(newObsSize, shape),
	                   torch::kCPU);

	auto from = oldHead.seq->parameters();
	auto to = newHead.seq->parameters();

	if (from.size() != to.size())
		return {false, "parameter count differs between old and new head"};

	for (size_t i = 0; i < from.size(); i++) {
		if (from[i].sizes() == to[i].sizes()) {
			to[i].copy_(from[i]);
			continue;
		}

		// The only legitimate mismatch is the first Linear's weight, whose
		// input dimension we are widening.
		const bool isFirstWeight =
			i == 0 && from[i].dim() == 2 &&
			from[i].size(0) == to[i].size(0) &&
			from[i].size(1) == oldObsSize && to[i].size(1) == newObsSize;

		if (!isFirstWeight)
			return {false, "unexpected parameter shape mismatch at index " +
			               std::to_string(i)};

		to[i].zero_();
		to[i].slice(1, 0, oldObsSize).copy_(from[i]);
	}

	std::filesystem::create_directories(dstFolder);
	newHead.Save(dstFolder, /*saveOptim=*/false);

	return {true, "migrated " + std::to_string(oldObsSize) + " -> " +
	              std::to_string(newObsSize)};
}

int ReadSavedObsSize(const std::filesystem::path& checkpointFolder,
                     const ModelShape& shape) {
	RG_NO_GRAD;

	const std::filesystem::path path = checkpointFolder / "SHARED_HEAD.lt";
	if (!std::filesystem::exists(path))
		return -1;

	// Deliberately torch::load on the Sequential rather than Model::Load: the
	// latter hard-checks parameter sizes and aborts, which is the very failure
	// this function exists to pre-empt. Torch resizes the modules to whatever
	// the archive holds, so the placeholder width below is irrelevant.
	GGL::Model probe("shared_head", MakeHeadConfig(1, shape), torch::kCPU);
	try {
		std::ifstream in(path, std::ios::binary);
		if (!in.good())
			return -1;
		torch::load(probe.seq, in, torch::kCPU);
	} catch (const std::exception&) {
		return -1;
	}

	const auto params = probe.seq->parameters();
	if (params.empty() || params[0].dim() != 2)
		return -1;

	return (int)params[0].size(1);
}

int RunMigrateObs(const std::filesystem::path& srcRun,
                  const std::filesystem::path& dstRun,
                  int oldObsSize,
                  int newObsSize) {
	const ModelShape shape = {};  // t1's shape is the ModelShape default

	const std::filesystem::path latest = FindLatestCheckpoint(srcRun);
	if (latest.empty()) {
		std::fprintf(stderr, "No complete checkpoint under %s\n",
		             srcRun.string().c_str());
		return EXIT_FAILURE;
	}

	// Migrate the head, then carry across everything the head does not own.
	// POLICY and CRITIC take sharedHeadLayers.back() as input, not obsSize
	// (PPOLearner.cpp:62-63), so they need no surgery.
	auto MigrateOne = [&](const std::filesystem::path& src,
	                      const std::filesystem::path& dst) -> bool {
		const MigrateResult r =
			MigrateSharedHead(src, dst, oldObsSize, newObsSize, shape);
		if (!r.ok) {
			std::fprintf(stderr, "  %s: %s\n", src.string().c_str(),
			             r.message.c_str());
			return false;
		}

		for (const auto& entry : std::filesystem::directory_iterator(src)) {
			if (entry.path().filename() == "SHARED_HEAD.lt")
				continue;
			// Optimizer state for the head is intentionally dropped; see the
			// note printed at the end.
			if (entry.path().filename() == "SHARED_HEAD_OPTIM.lt")
				continue;
			std::filesystem::copy(
				entry.path(), dst / entry.path().filename(),
				std::filesystem::copy_options::overwrite_existing);
		}
		return true;
	};

	std::filesystem::create_directories(dstRun);

	std::printf("Migrating newest checkpoint %s\n", latest.string().c_str());
	if (!MigrateOne(latest, dstRun / latest.filename()))
		return EXIT_FAILURE;

	// Run-level files sit beside the numbered checkpoint folders.
	for (const auto& entry : std::filesystem::directory_iterator(srcRun)) {
		if (entry.is_directory())
			continue;
		std::filesystem::copy(
			entry.path(), dstRun / entry.path().filename(),
			std::filesystem::copy_options::overwrite_existing);
	}

	// The self-play opponent ladder. Model::Load hard-checks parameter sizes,
	// so an unmigrated snapshot would abort the run at load time.
	const std::filesystem::path versions = srcRun / "policy_versions";
	if (std::filesystem::exists(versions)) {
		int migrated = 0;
		for (const auto& entry : std::filesystem::directory_iterator(versions)) {
			if (!entry.is_directory())
				continue;
			const auto dst =
				dstRun / "policy_versions" / entry.path().filename();
			if (!MigrateOne(entry.path(), dst))
				return EXIT_FAILURE;
			migrated++;
		}
		std::printf("Migrated %d policy version(s)\n", migrated);
	}

	std::printf(
		"\nDone: %s\n"
		"NOTE: shared-head optimizer state was NOT carried across. Adam's\n"
		"      moments re-estimate within a few hundred iterations; at lr 1e-4\n"
		"      the transient is small. Resume with a reduced --lr for the\n"
		"      first few iterations if you want to be careful.\n",
		dstRun.string().c_str());

	return EXIT_SUCCESS;
}

}  // namespace Dash
