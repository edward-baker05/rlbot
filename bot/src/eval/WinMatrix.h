#pragma once

#include "../env/Obs.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Dash {

// Round-robin win-rate matrix over a lineage's saved policy versions.
//
// Why this exists, when Rating/* already reports an Elo:
//
// GigaLearn's skill tracker only ever plays CURRENT against ONE randomly chosen
// old version (PolicyVersionManager::RunSkillMatches). It never plays two old
// versions against each other, so it cannot observe the relation that matters
// most for a self-play plateau: whether the lineage is TRANSITIVE.
//
// Elo assumes transitivity. If v30 beats v20, v20 beats v10, and v10 beats v30,
// Elo cannot represent that -- it averages the cycle away and reports three
// similar numbers, or worse, a rising line. A rock-paper-scissors lineage looks
// exactly like a converged one through that lens, and the two call for opposite
// responses: cycling means the opponent pool needs curating, convergence means
// the pool is exhausted and something outside it has to change.
//
// This plays every pair and reports the structure directly.

struct WinMatrixConfig {
	std::filesystem::path runFolder;   // checkpoints/<label>
	int maxVersions = 8;               // subsample evenly to bound cost; 0 = all
	int gamesPerPair = 40;
	int arenas = 64;
	float gameDuration = 300.f;
	bool deterministic = true;
	bool useGPU = true;
	std::string jsonOutput = {};
};

struct WinMatrixEntry {
	int i = 0, j = 0;          // version indices, i < j
	int winsI = 0, winsJ = 0, ties = 0;
	int goalsI = 0, goalsJ = 0;
	float winRateI = 0.f;      // excluding ties
};

struct WinMatrixResult {
	bool valid = false;
	std::vector<uint64_t> versionSteps;      // timestep label per version
	std::vector<WinMatrixEntry> pairs;

	// Tournament structure
	int cyclicTriples = 0;
	int totalTriples = 0;
	float consistency = 1.f;        // Kendall's zeta: 1 = fully transitive, 0 = max cycling
	float monotonicity = 0.f;       // fraction of (earlier, later) pairs the LATER one wins
	std::vector<int> tournamentWins; // out-degree per version
};

WinMatrixResult RunWinMatrix(const WinMatrixConfig &cfg);
void PrintWinMatrix(const WinMatrixResult &res);

} // namespace Dash
