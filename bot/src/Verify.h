#pragma once

#include <filesystem>

namespace Hive {

// Checkpoint/deployment parity check: loads a checkpoint the way the RLBot
// client does and asserts it infers sanely. Returns 0 on pass, 1 on failure.
int RunVerify(const std::filesystem::path& checkpointFolder);

} // namespace Hive
