#include "Checkpoints.h"

#include <string>

namespace fs = std::filesystem;

namespace Hive {

namespace {

bool IsAllDigits(const std::string& s) {
	if (s.empty())
		return false;
	for (char c : s)
		if (c < '0' || c > '9')
			return false;
	return true;
}

// What Policy::Load -> GGL::InferUnit actually opens. RUNNING_STATS.json is
// included because loading without it gives an unnormalized observation: the
// bot runs, and is quietly wrong, which is the failure mode this project
// keeps hitting.
bool IsComplete(const fs::path& dir) {
	std::error_code ec;
	for (const char* f : {"POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json"})
		if (!fs::exists(dir / f, ec))
			return false;
	return true;
}

} // namespace

fs::path FindLatestCheckpoint(const fs::path& runFolder) {
	std::error_code ec;
	if (!fs::is_directory(runFolder, ec))
		return {};

	fs::path best;
	unsigned long long bestStep = 0;
	bool found = false;

	for (const auto& entry : fs::directory_iterator(runFolder, ec)) {
		if (ec)
			break;
		if (!entry.is_directory(ec))
			continue;

		// Skips policy_versions (the self-play snapshot pool) for free.
		const std::string name = entry.path().filename().string();
		if (!IsAllDigits(name))
			continue;

		unsigned long long step = 0;
		try {
			step = std::stoull(name);
		} catch (const std::exception&) {
			continue; // Digits but wider than 64 bits; not a real checkpoint.
		}

		if (found && step <= bestStep)
			continue;
		if (!IsComplete(entry.path()))
			continue;

		best = entry.path();
		bestStep = step;
		found = true;
	}

	return best;
}

} // namespace Hive
