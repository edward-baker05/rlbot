#pragma once

#include <filesystem>

namespace Hive {

// Newest loadable checkpoint inside a run folder (checkpoints/main-<label>),
// or an empty path if there is none.
//
// "Newest" is by step count parsed as a number, not by name: GigaLearn names
// checkpoint folders after the step they were saved at, so lexicographic order
// puts 9000000 after 10000000 and silently freezes a follower at 9M.
//
// Folders missing any file the policy load needs are skipped. That is not
// defensive padding -- the trainer writes a new checkpoint every tsPerSave
// while a spectator is reading, so the newest folder on disk is regularly a
// partial one.
std::filesystem::path FindLatestCheckpoint(const std::filesystem::path& runFolder);

} // namespace Hive
