#include "doctest/doctest.h"
#include "TestCommon.h"

#include <eval/MigrateObs.h>
#include <policy/Policy.h>

#include <private/GigaLearnCPP/Util/Models.h>

#include <torch/torch.h>

#include <filesystem>

using namespace Dash;

namespace {

GGL::ModelConfig MakeHeadConfig(int numInputs, const ModelShape& shape) {
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

} // namespace

TEST_CASE("MigrateSharedHead preserves the network's output exactly") {
	RG_NO_GRAD;

	// A small shape keeps the test fast; the mechanism is size-independent.
	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const int oldObs = 12;
	const int newObs = 20;

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-test";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	// Build and save a randomly initialized old-width head.
	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObs, shape),
	                   torch::kCPU);
	oldHead.Save(tmp / "src", /*saveOptim=*/false);

	const MigrateResult result = MigrateSharedHead(
		tmp / "src", tmp / "dst", oldObs, newObs, shape);
	REQUIRE_MESSAGE(result.ok, result.message);

	GGL::Model newHead("shared_head", MakeHeadConfig(newObs, shape),
	                   torch::kCPU);
	newHead.Load(tmp / "dst", /*allowNotExist=*/false, /*loadOptim=*/false);

	// The same input, zero-padded, must produce the same output.
	torch::Tensor input = torch::randn({4, oldObs});
	torch::Tensor padded = torch::cat(
		{input, torch::zeros({4, newObs - oldObs})}, 1);

	torch::Tensor before = oldHead.Forward(input, false);
	torch::Tensor after = newHead.Forward(padded, false);

	CHECK(torch::allclose(before, after, 1e-6, 1e-6));

	std::filesystem::remove_all(tmp);
}

TEST_CASE("MigrateSharedHead zeroes only the new input columns") {
	RG_NO_GRAD;

	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const int oldObs = 12;
	const int newObs = 20;

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-cols";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	GGL::Model oldHead("shared_head", MakeHeadConfig(oldObs, shape),
	                   torch::kCPU);
	oldHead.Save(tmp / "src", /*saveOptim=*/false);

	REQUIRE(MigrateSharedHead(tmp / "src", tmp / "dst", oldObs, newObs,
	                          shape).ok);

	GGL::Model newHead("shared_head", MakeHeadConfig(newObs, shape),
	                   torch::kCPU);
	newHead.Load(tmp / "dst", /*allowNotExist=*/false, /*loadOptim=*/false);

	torch::Tensor oldW = oldHead.seq->parameters()[0];
	torch::Tensor newW = newHead.seq->parameters()[0];

	REQUIRE(newW.size(1) == newObs);
	CHECK(torch::equal(newW.slice(1, 0, oldObs), oldW));
	CHECK(newW.slice(1, oldObs, newObs).abs().sum().item<float>() == 0.f);

	std::filesystem::remove_all(tmp);
}

TEST_CASE("MigrateSharedHead refuses to shrink the input") {
	ModelShape shape = {};
	shape.sharedHeadLayers = {32, 16};

	const auto tmp = std::filesystem::temp_directory_path() /
	                 "dash-migrate-shrink";
	std::filesystem::remove_all(tmp);
	std::filesystem::create_directories(tmp / "src");

	GGL::Model head("shared_head", MakeHeadConfig(20, shape), torch::kCPU);
	head.Save(tmp / "src", /*saveOptim=*/false);

	const MigrateResult result =
		MigrateSharedHead(tmp / "src", tmp / "dst", 20, 12, shape);
	CHECK_FALSE(result.ok);

	std::filesystem::remove_all(tmp);
}
