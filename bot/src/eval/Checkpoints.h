#pragma once

#include <filesystem>

namespace Hive {

// Newest loadable checkpoint inside a run folder (checkpoints/main-<label>).
// "Newest" is by step count parsed as a number, not lexicographically, and
// folders missing a required file are skipped (the trainer may still be
// mid-write to the newest one).
std::filesystem::path FindLatestCheckpoint(const std::filesystem::path& runFolder);

} // namespace Hive
