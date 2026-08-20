#pragma once

#include <filesystem>

namespace Hive {

std::filesystem::path FindLatestCheckpoint(const std::filesystem::path& runFolder);

}  // namespace Hive
