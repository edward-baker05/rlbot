#pragma once

#include "../Config.h"
#include <RLGymCPP/Gamestates/Player.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Dash {

struct MatchGameResult {
	int gameId = 0;
	int score1 = 0;
	int score2 = 0;
	int winner = 0; // 1 = bot1, 2 = bot2, 0 = tie
	bool overtime = false;
	float overtimeDuration = 0.f;
	Team bot1Team = Team::BLUE;
};

struct MatchSummaryResult {
	bool valid = false;
	std::string label1;
	std::string label2;
	std::filesystem::path model1Path;
	std::filesystem::path model2Path;

	int totalGames = 0;
	int bot1Wins = 0;
	int bot2Wins = 0;
	int ties = 0;

	float bot1WinRate = 0.f;
	float bot2WinRate = 0.f;
	float winRateCiLower95 = 0.f;
	float winRateCiUpper95 = 0.f;
	double binomialPValue = 1.0;

	int bot1Goals = 0;
	int bot2Goals = 0;
	float bot1MeanGoals = 0.f;
	float bot2MeanGoals = 0.f;
	float bot1StdGoals = 0.f;
	float bot2StdGoals = 0.f;

	float meanGoalDiff = 0.f; // (bot2 - bot1)
	float stdGoalDiff = 0.f;
	float goalDiffCiLower95 = 0.f;
	float goalDiffCiUpper95 = 0.f;
	double pairedTPValue = 1.0;
	double wilcoxonPValue = 1.0;

	int overtimeGames = 0;
	int bot1OtWins = 0;
	int bot2OtWins = 0;
	float meanOtDuration = 0.f;

	// Side statistics
	int bot1BlueGames = 0;
	int bot1BlueWins = 0;
	int bot1OrangeGames = 0;
	int bot1OrangeWins = 0;

	int bot2BlueGames = 0;
	int bot2BlueWins = 0;
	int bot2OrangeGames = 0;
	int bot2OrangeWins = 0;

	float blueWinRate = 0.f;
	float orangeWinRate = 0.f;

	std::vector<MatchGameResult> games;
};

struct MatchRunnerConfig {
	std::filesystem::path model1;
	std::filesystem::path model2;
	std::string label1 = "t1";
	std::string label2 = "t2";

	int totalGames = 500;
	int arenas = 64;
	float gameDuration = 300.0f; // 5 minutes in seconds
	bool suddenDeathOvertime = true;
	float maxOvertime = 300.0f; // max 5 min OT
	bool deterministic = true;
	float temperature = 1.0f;
	bool useGPU = true;
	ObsMode obs = ObsMode::Advanced;
	int maxPlayersPerTeam = 3;
	bool maskActions = true;
	int tickSkip = 8;
	int actionDelay = 7;
	std::string jsonOutput = "";
};

class MatchBench {
  public:
	explicit MatchBench(const MatchRunnerConfig &cfg);
	~MatchBench();

	MatchBench(const MatchBench &) = delete;
	MatchBench &operator=(const MatchBench &) = delete;

	MatchSummaryResult Run();

  private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

std::filesystem::path ResolveCheckpoint(const std::string &input);
ModelShape LoadModelShapeFromConfig(const std::filesystem::path &checkpointPath);

} // namespace Dash
