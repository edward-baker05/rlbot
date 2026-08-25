#pragma once

#include "../policy/Policy.h"

#include <filesystem>
#include <string>

namespace Dash {

struct MigrateResult {
	bool ok = false;
	std::string message;
};

// Widens a saved shared head's input layer, zero-filling the new columns.
//
// Exactly behaviour-preserving: LayerNorm sits after each hidden Linear, never
// on the input (Models.cpp:17-22), so the raw obs feeds straight into
// Linear(obsSize, N) and zero columns contribute nothing. This would NOT hold
// if standardizeObs were enabled, since a zero input standardizes to
// -mean/std.
MigrateResult MigrateSharedHead(const std::filesystem::path& srcFolder,
                                const std::filesystem::path& dstFolder,
                                int oldObsSize,
                                int newObsSize,
                                const ModelShape& shape);

// Migrates a whole run folder: the newest checkpoint plus every snapshot under
// policy_versions/, which are the self-play opponent ladder and would fail
// Model::Load's size check otherwise.
int RunMigrateObs(const std::filesystem::path& srcRun,
                  const std::filesystem::path& dstRun,
                  int oldObsSize,
                  int newObsSize);

}  // namespace Dash
