#include "doctest/doctest.h"

#include <eval/Checkpoints.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Hive;
namespace fs = std::filesystem;

namespace {

// A scratch run folder that cleans itself up.
struct TempRun {
	fs::path root;

	TempRun() {
		root = fs::temp_directory_path() /
		       ("hive-ckpt-test-" + std::to_string(::rand()));
		fs::create_directories(root);
	}
	~TempRun() {
		std::error_code ec;
		fs::remove_all(root, ec);
	}

	// A checkpoint folder GigaLearn would consider finished.
	fs::path Complete(const std::string& name) {
		fs::path p = root / name;
		fs::create_directories(p);
		for (const char* f : {"POLICY.lt", "SHARED_HEAD.lt", "RUNNING_STATS.json",
		                      "CRITIC.lt", "POLICY_OPTIM.lt"})
			std::ofstream(p / f) << "x";
		return p;
	}

	// A folder the trainer is midway through writing.
	fs::path Partial(const std::string& name,
	                 const std::vector<std::string>& files) {
		fs::path p = root / name;
		fs::create_directories(p);
		for (const auto& f : files)
			std::ofstream(p / f) << "x";
		return p;
	}
};

} // namespace

TEST_CASE("FindLatestCheckpoint returns nothing for a missing or empty folder") {
	CHECK(FindLatestCheckpoint("/definitely/not/a/real/path").empty());

	TempRun run;
	CHECK(FindLatestCheckpoint(run.root).empty());
}

TEST_CASE("FindLatestCheckpoint picks the highest step count") {
	TempRun run;
	run.Complete("1000000");
	fs::path newest = run.Complete("2000000");
	run.Complete("1500000");

	CHECK(FindLatestCheckpoint(run.root) == newest);
}

TEST_CASE("FindLatestCheckpoint orders numerically, not lexicographically") {
	// The bug this guards: string ordering puts "9000000" after "10000000",
	// so a lexicographic pick silently freezes at 9M for the rest of a run.
	TempRun run;
	run.Complete("9000000");
	fs::path newest = run.Complete("10000000");

	CHECK(FindLatestCheckpoint(run.root) == newest);
}

TEST_CASE("FindLatestCheckpoint skips policy_versions") {
	// policy_versions sits beside the numbered folders and is a snapshot pool,
	// not a loadable checkpoint. It also sorts last alphabetically, which is
	// exactly how it gets picked by a naive glob.
	TempRun run;
	fs::path newest = run.Complete("5000000");
	run.Complete("policy_versions");

	CHECK(FindLatestCheckpoint(run.root) == newest);
}

TEST_CASE("FindLatestCheckpoint ignores a half-written checkpoint") {
	// The trainer writes into a new numbered folder while we are reading. A
	// folder missing any file the policy load needs must not be chosen, or the
	// spectator crashes every time training checkpoints.
	TempRun run;
	fs::path complete = run.Complete("4000000");

	SUBCASE("missing RUNNING_STATS.json") {
		run.Partial("5000000", {"POLICY.lt", "SHARED_HEAD.lt"});
	}
	SUBCASE("missing SHARED_HEAD.lt") {
		run.Partial("5000000", {"POLICY.lt", "RUNNING_STATS.json"});
	}
	SUBCASE("missing POLICY.lt") {
		run.Partial("5000000", {"SHARED_HEAD.lt", "RUNNING_STATS.json"});
	}
	SUBCASE("entirely empty") {
		run.Partial("5000000", {});
	}

	CHECK(FindLatestCheckpoint(run.root) == complete);
}

TEST_CASE("FindLatestCheckpoint ignores non-numeric and non-directory entries") {
	TempRun run;
	fs::path newest = run.Complete("3000000");
	run.Complete("latest");
	run.Complete("30000000-old");
	std::ofstream(run.root / "40000000") << "a file, not a folder";

	CHECK(FindLatestCheckpoint(run.root) == newest);
}
