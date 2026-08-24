#pragma once

#include <filesystem>

namespace Dash {

std::filesystem::path FindLatestCheckpoint(const std::filesystem::path& runFolder);

}  // namespace Dash
