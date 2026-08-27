#include "MatchBench.h"
#include "Checkpoints.h"
#include "../env/Actions.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"

#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;
using namespace RLGC;

namespace Dash {

namespace {

// Cumulative distribution function of standard normal distribution
double NormalCDF(double x) {
	return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Student's t-distribution two-tailed p-value approximation
double StudentTTwoTailedPValue(double t_stat, int df) {
	if (df <= 0)
		return 1.0;
	double t = std::abs(t_stat);
	// For large degrees of freedom (df >= 30), t-distribution closely approximates normal distribution
	// We use standard Hill's approximation / normal approximation with Cornish-Fisher expansion
	double x = t * (1.0 - 1.0 / (4.0 * df)) / std::sqrt(1.0 + t * t / (2.0 * df));
	double p = 2.0 * (1.0 - NormalCDF(x));
	return std::clamp(p, 0.0, 1.0);
}

// Log of combination ln(n choose k)
double LogComb(int n, int k) {
	if (k < 0 || k > n)
		return -1e30;
	return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
}

// Exact two-tailed Binomial test against p = 0.5
double ExactBinomialTestPValue(int wins, int total) {
	if (total <= 0)
		return 1.0;
	if (wins * 2 == total)
		return 1.0;

	int k = std::max(wins, total - wins);
	double log2n = total * std::log(2.0);
	double tailSum = 0.0;

	for (int i = k; i <= total; ++i) {
		double logTerm = LogComb(total, i) - log2n;
		tailSum += std::exp(logTerm);
	}

	double p = 2.0 * tailSum;
	return std::clamp(p, 0.0, 1.0);
}

// Wilson Score 95% Confidence Interval for a proportion
std::pair<float, float> WilsonScoreCI(int successes, int total, double z = 1.95996398454) {
	if (total <= 0)
		return {0.f, 0.f};
	double p_hat = static_cast<double>(successes) / total;
	double z2 = z * z;
	double n = total;

	double denom = 1.0 + z2 / n;
	double center = (p_hat + z2 / (2.0 * n)) / denom;
	double margin = (z * std::sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom;

	float lower = static_cast<float>(std::max(0.0, center - margin));
	float upper = static_cast<float>(std::min(1.0, center + margin));
	return {lower, upper};
}

// Wilcoxon Signed-Rank Test for paired differences
double WilcoxonSignedRankTest(const std::vector<int> &diffs) {
	std::vector<std::pair<int, int>> nonZero; // {absDiff, sign}
	for (int d : diffs) {
		if (d != 0) {
			nonZero.push_back({std::abs(d), d > 0 ? 1 : -1});
		}
	}

	int nr = static_cast<int>(nonZero.size());
	if (nr < 5)
		return 1.0;

	std::sort(nonZero.begin(), nonZero.end(), [](const auto &a, const auto &b) {
		return a.first < b.first;
	});

	std::vector<double> ranks(nr, 0.0);
	std::vector<int> tieCounts;

	int i = 0;
	while (i < nr) {
		int j = i;
		while (j < nr && nonZero[j].first == nonZero[i].first) {
			j++;
		}
		int count = j - i;
		tieCounts.push_back(count);
		double avgRank = (i + 1 + j) / 2.0;
		for (int k = i; k < j; k++) {
			ranks[k] = avgRank;
		}
		i = j;
	}

	double W_pos = 0.0;
	for (int k = 0; k < nr; k++) {
		if (nonZero[k].second > 0) {
			W_pos += ranks[k];
		}
	}

	double meanW = nr * (nr + 1) / 4.0;
	double tieCorrection = 0.0;
	for (int t : tieCounts) {
		tieCorrection += (std::pow(t, 3) - t);
	}
	double varW = (nr * (nr + 1.0) * (2.0 * nr + 1.0) - 0.5 * tieCorrection) / 24.0;
	if (varW <= 0.0)
		return 1.0;

	double z = (W_pos - meanW) / std::sqrt(varW);
	double p = 2.0 * (1.0 - NormalCDF(std::abs(z)));
	return std::clamp(p, 0.0, 1.0);
}

} // namespace

fs::path ResolveCheckpoint(const std::string &input) {
	fs::path p = input;
	std::error_code ec;

	// Check if already direct checkpoint path
	if (fs::is_directory(p, ec) && fs::exists(p / "POLICY.lt", ec))
		return p;

	// Check via FindLatestCheckpoint
	fs::path found = FindLatestCheckpoint(p);
	if (!found.empty() && fs::exists(found / "POLICY.lt", ec))
		return found;

	// Check in bot/build/checkpoints/<input>
	fs::path inBuildCkpts = fs::path("bot/build/checkpoints") / input;
	found = FindLatestCheckpoint(inBuildCkpts);
	if (!found.empty() && fs::exists(found / "POLICY.lt", ec))
		return found;

	// Check in checkpoints/<input>
	fs::path inCkpts = fs::path("checkpoints") / input;
	found = FindLatestCheckpoint(inCkpts);
	if (!found.empty() && fs::exists(found / "POLICY.lt", ec))
		return found;

	return {};
}

struct BotSpec {
	ModelShape shape;
	ObsMode obsMode = ObsMode::Advanced;
	int maxPlayersPerTeam = 3;
	bool maskActions = true;
};

BotSpec LoadBotSpecFromConfig(const fs::path &checkpointPath) {
	BotSpec spec = {};
	fs::path configPath = checkpointPath.parent_path() / "CONFIG.json";
	std::error_code ec;
	if (!fs::exists(configPath, ec)) {
		configPath = checkpointPath / "CONFIG.json";
	}
	if (!fs::exists(configPath, ec)) {
		// A saved policy version lives at <run>/policy_versions/<timesteps>/, so the
		// run's CONFIG.json is two levels up, not one. Without this the spec falls
		// back to defaults -- Advanced obs -- and a Predictive-obs lineage then
		// loads into the wrong input width, or worse, the right width with the
		// wrong meaning. Silent, and it would poison every matrix cell.
		configPath = checkpointPath.parent_path().parent_path() / "CONFIG.json";
	}
	if (fs::exists(configPath, ec)) {
		try {
			std::ifstream in(configPath);
			nlohmann::json j;
			in >> j;
			if (j.contains("model")) {
				auto &m = j["model"];
				if (m.contains("sharedHeadLayers") && m["sharedHeadLayers"].is_array()) {
					spec.shape.sharedHeadLayers = m["sharedHeadLayers"].get<std::vector<int>>();
				}
				if (m.contains("policyLayers") && m["policyLayers"].is_array()) {
					spec.shape.policyLayers = m["policyLayers"].get<std::vector<int>>();
				}
				if (m.contains("addLayerNorm") && m["addLayerNorm"].is_boolean()) {
					spec.shape.addLayerNorm = m["addLayerNorm"].get<bool>();
				}
			}
			if (j.contains("env")) {
				auto &e = j["env"];
				if (e.contains("obs") && e["obs"].is_string()) {
					std::string obsStr = e["obs"].get<std::string>();
					if (obsStr == "Relative" || obsStr == "relative")
						spec.obsMode = ObsMode::Relative;
					else if (obsStr == "Default" || obsStr == "default")
						spec.obsMode = ObsMode::Default;
					else if (obsStr == "Advanced" || obsStr == "advanced")
						spec.obsMode = ObsMode::Advanced;
					else if (obsStr == "Predictive" || obsStr == "predictive")
						spec.obsMode = ObsMode::Predictive;
				}
				if (e.contains("maxPlayersPerTeam") && e["maxPlayersPerTeam"].is_number_integer()) {
					spec.maxPlayersPerTeam = e["maxPlayersPerTeam"].get<int>();
				}
				if (e.contains("maskActions") && e["maskActions"].is_boolean()) {
					spec.maskActions = e["maskActions"].get<bool>();
				}
			}
		} catch (const std::exception &e) {
			std::cerr << "Warning: Could not parse CONFIG.json at " << configPath << ": " << e.what() << "\n";
		}
	}
	return spec;
}

ModelShape LoadModelShapeFromConfig(const fs::path &checkpointPath) {
	return LoadBotSpecFromConfig(checkpointPath).shape;
}

struct MatchBench::Impl {
	MatchRunnerConfig cfg;

	std::vector<Arena *> arenas;
	std::vector<GameState> states;
	GoalScoreCondition goalScoredCondition;

	std::unique_ptr<ObsBuilder> obsBuilder1;
	std::unique_ptr<ObsBuilder> obsBuilder2;
	std::unique_ptr<DefaultAction> parser1;
	std::unique_ptr<DefaultAction> parser2;
	int obsSize1 = 0;
	int obsSize2 = 0;

	std::unique_ptr<Policy> policy1;
	std::unique_ptr<Policy> policy2;

	explicit Impl(const MatchRunnerConfig &cfg) : cfg(cfg) {
		BotSpec spec1 = LoadBotSpecFromConfig(cfg.model1);
		BotSpec spec2 = LoadBotSpecFromConfig(cfg.model2);

		obsSize1 = ProbeObsSize(spec1.maxPlayersPerTeam, spec1.obsMode);
		obsBuilder1 = MakeObsBuilder(spec1.maxPlayersPerTeam, spec1.obsMode);
		parser1 = MakeActionParser(spec1.maskActions);

		obsSize2 = ProbeObsSize(spec2.maxPlayersPerTeam, spec2.obsMode);
		obsBuilder2 = MakeObsBuilder(spec2.maxPlayersPerTeam, spec2.obsMode);
		parser2 = MakeActionParser(spec2.maskActions);

		policy1 = std::make_unique<Policy>(obsBuilder1.get(), obsSize1, parser1.get(), spec1.shape, cfg.useGPU);
		policy2 = std::make_unique<Policy>(obsBuilder2.get(), obsSize2, parser2.get(), spec2.shape, cfg.useGPU);

		policy1->Load(cfg.model1);
		policy2->Load(cfg.model2);

		int numArenas = std::max(1, cfg.arenas);
		for (int i = 0; i < numArenas; i++) {
			Arena *arena = Arena::Create(GameMode::SOCCAR);
			arena->AddCar(Team::BLUE);
			arena->AddCar(Team::ORANGE);
			arenas.push_back(arena);
		}
		states.resize(arenas.size());
	}

	~Impl() {
		for (Arena *arena : arenas)
			delete arena;
	}
};

MatchBench::MatchBench(const MatchRunnerConfig &cfg)
	: impl(std::make_unique<Impl>(cfg)) {}

MatchBench::~MatchBench() = default;

MatchSummaryResult MatchBench::Run() {
	MatchSummaryResult summary = {};
	Impl &s = *impl;
	const MatchRunnerConfig &cfg = s.cfg;

	summary.label1 = cfg.label1;
	summary.label2 = cfg.label2;
	summary.model1Path = cfg.model1;
	summary.model2Path = cfg.model2;
	summary.totalGames = cfg.totalGames;

	const size_t numArenas = s.arenas.size();
	const float stepTime = static_cast<float>(cfg.tickSkip) / 120.f;

	// Per-arena state tracking
	std::vector<bool> arenaActive(numArenas, false);
	std::vector<int> gameId(numArenas, -1);
	std::vector<Team> bot1Team(numArenas, Team::BLUE);
	std::vector<float> gameClock(numArenas, 0.f);
	std::vector<bool> inOvertime(numArenas, false);
	std::vector<float> otClock(numArenas, 0.f);
	std::vector<int> score1(numArenas, 0);
	std::vector<int> score2(numArenas, 0);

	int nextGameIndex = 0;
	int completedGamesCount = 0;

	auto startNewGameInArena = [&](size_t a) {
		if (nextGameIndex >= cfg.totalGames) {
			arenaActive[a] = false;
			return;
		}

		int g = nextGameIndex++;
		arenaActive[a] = true;
		gameId[a] = g;
		// Balance sides strictly: even games Bot1 is BLUE, odd games Bot1 is ORANGE
		bot1Team[a] = (g % 2 == 0) ? Team::BLUE : Team::ORANGE;
		gameClock[a] = 0.f;
		inOvertime[a] = false;
		otClock[a] = 0.f;
		score1[a] = 0;
		score2[a] = 0;

		s.arenas[a]->ResetToRandomKickoff();
		s.states[a] = GameState(s.arenas[a]);
		s.states[a].UpdateFromArena(s.arenas[a], std::vector<Action>(2), nullptr);
	};

	for (size_t a = 0; a < numArenas; a++) {
		startNewGameInArena(a);
	}

	// Scratch buffers for batch inference
	std::vector<Player> p1Players, p2Players;
	std::vector<GameState> p1States, p2States;
	std::vector<std::pair<size_t, int>> p1Slots, p2Slots;
	std::vector<std::vector<Action>> applied(numArenas, std::vector<Action>(2));

	int lastReportedPercent = -1;

	while (completedGamesCount < cfg.totalGames) {
		p1Players.clear();
		p1States.clear();
		p1Slots.clear();
		p2Players.clear();
		p2States.clear();
		p2Slots.clear();

		for (size_t a = 0; a < numArenas; a++) {
			if (!arenaActive[a])
				continue;

			const GameState &gs = s.states[a];
			int p1Idx = (bot1Team[a] == Team::BLUE) ? 0 : 1;
			int p2Idx = (bot1Team[a] == Team::BLUE) ? 1 : 0;

			p1Players.push_back(gs.players[p1Idx]);
			p1States.push_back(gs);
			p1Slots.push_back({a, p1Idx});

			p2Players.push_back(gs.players[p2Idx]);
			p2States.push_back(gs);
			p2Slots.push_back({a, p2Idx});
		}

		if (p1Players.empty())
			break;

		std::vector<Action> p1Actions = s.policy1->InferBatch(p1Players, p1States, cfg.deterministic, cfg.temperature);
		std::vector<Action> p2Actions = s.policy2->InferBatch(p2Players, p2States, cfg.deterministic, cfg.temperature);

		for (size_t i = 0; i < p1Slots.size(); i++)
			applied[p1Slots[i].first][p1Slots[i].second] = p1Actions[i];
		for (size_t i = 0; i < p2Slots.size(); i++)
			applied[p2Slots[i].first][p2Slots[i].second] = p2Actions[i];

		for (size_t a = 0; a < numArenas; a++) {
			if (!arenaActive[a])
				continue;

			Arena *arena = s.arenas[a];
			s.states[a].ResetBeforeStep();

			arena->Step(cfg.actionDelay);
			auto carItr = arena->_cars.begin();
			(*carItr)->controls = (CarControls)applied[a][0];
			carItr++;
			(*carItr)->controls = (CarControls)applied[a][1];

			arena->Step(cfg.tickSkip - cfg.actionDelay);
			s.states[a].UpdateFromArena(arena, applied[a], nullptr);

			if (!inOvertime[a]) {
				gameClock[a] += stepTime;
			} else {
				otClock[a] += stepTime;
			}

			// Check Goal
			if (s.states[a].goalScored) {
				Team conceding = RS_TEAM_FROM_Y(s.states[a].ball.pos.y);
				Team scoringTeam = (conceding == Team::BLUE) ? Team::ORANGE : Team::BLUE;
				bool bot1Scored = (scoringTeam == bot1Team[a]);

				if (bot1Scored)
					score1[a]++;
				else
					score2[a]++;

				if (inOvertime[a]) {
					// Sudden death overtime goal!
					MatchGameResult gr = {};
					gr.gameId = gameId[a];
					gr.score1 = score1[a];
					gr.score2 = score2[a];
					gr.winner = bot1Scored ? 1 : 2;
					gr.overtime = true;
					gr.overtimeDuration = otClock[a];
					gr.bot1Team = bot1Team[a];
					summary.games.push_back(gr);

					completedGamesCount++;
					startNewGameInArena(a);
				} else {
					// Regulation goal -> Reset to kickoff and continue
					arena->ResetToRandomKickoff();
					s.states[a] = GameState(arena);
					s.states[a].UpdateFromArena(arena, std::vector<Action>(2), nullptr);
				}
			} else if (!inOvertime[a] && gameClock[a] >= cfg.gameDuration) {
				// End of regulation
				if (score1[a] != score2[a]) {
					// Decisive regulation finish
					MatchGameResult gr = {};
					gr.gameId = gameId[a];
					gr.score1 = score1[a];
					gr.score2 = score2[a];
					gr.winner = (score1[a] > score2[a]) ? 1 : 2;
					gr.overtime = false;
					gr.overtimeDuration = 0.f;
					gr.bot1Team = bot1Team[a];
					summary.games.push_back(gr);

					completedGamesCount++;
					startNewGameInArena(a);
				} else {
					// Tied regulation -> Enter sudden-death overtime!
					inOvertime[a] = true;
					otClock[a] = 0.f;
					arena->ResetToRandomKickoff();
					s.states[a] = GameState(arena);
					s.states[a].UpdateFromArena(arena, std::vector<Action>(2), nullptr);
				}
			} else if (inOvertime[a] && otClock[a] >= cfg.maxOvertime) {
				// Overtime timeout (Tie)
				MatchGameResult gr = {};
				gr.gameId = gameId[a];
				gr.score1 = score1[a];
				gr.score2 = score2[a];
				gr.winner = 0;
				gr.overtime = true;
				gr.overtimeDuration = otClock[a];
				gr.bot1Team = bot1Team[a];
				summary.games.push_back(gr);

				completedGamesCount++;
				startNewGameInArena(a);
			}
		}

		int percent = (completedGamesCount * 100) / cfg.totalGames;
		if (percent != lastReportedPercent && percent % 10 == 0) {
			std::printf("  Progress: %d%% (%d / %d games completed)\n", percent, completedGamesCount, cfg.totalGames);
			lastReportedPercent = percent;
		}
	}

	if (summary.games.empty()) {
		summary.valid = false;
		return summary;
	}

	// Sort games by gameId
	std::sort(summary.games.begin(), summary.games.end(), [](const auto &a, const auto &b) {
		return a.gameId < b.gameId;
	});

	summary.valid = true;
	summary.totalGames = static_cast<int>(summary.games.size());

	std::vector<int> diffs;
	std::vector<float> otDurations;

	for (const auto &g : summary.games) {
		if (g.winner == 1) {
			summary.bot1Wins++;
			if (g.overtime)
				summary.bot1OtWins++;
		} else if (g.winner == 2) {
			summary.bot2Wins++;
			if (g.overtime)
				summary.bot2OtWins++;
		} else {
			summary.ties++;
		}

		summary.bot1Goals += g.score1;
		summary.bot2Goals += g.score2;
		diffs.push_back(g.score2 - g.score1);

		if (g.overtime) {
			summary.overtimeGames++;
			otDurations.push_back(g.overtimeDuration);
		}

		// Side stats
		if (g.bot1Team == Team::BLUE) {
			summary.bot1BlueGames++;
			summary.bot2OrangeGames++;
			if (g.winner == 1) {
				summary.bot1BlueWins++;
			} else if (g.winner == 2) {
				summary.bot2OrangeWins++;
			}
		} else {
			summary.bot1OrangeGames++;
			summary.bot2BlueGames++;
			if (g.winner == 1) {
				summary.bot1OrangeWins++;
			} else if (g.winner == 2) {
				summary.bot2BlueWins++;
			}
		}
	}

	const int n = summary.totalGames;
	summary.bot1WinRate = static_cast<float>(summary.bot1Wins) / n;
	summary.bot2WinRate = static_cast<float>(summary.bot2Wins) / n;

	auto [ciLower, ciUpper] = WilsonScoreCI(summary.bot2Wins, n);
	summary.winRateCiLower95 = ciLower;
	summary.winRateCiUpper95 = ciUpper;

	summary.binomialPValue = ExactBinomialTestPValue(summary.bot2Wins, n);

	summary.bot1MeanGoals = static_cast<float>(summary.bot1Goals) / n;
	summary.bot2MeanGoals = static_cast<float>(summary.bot2Goals) / n;

	double var1 = 0.0, var2 = 0.0, varDiff = 0.0;
	double sumDiff = 0.0;
	for (const auto &g : summary.games) {
		double d1 = g.score1 - summary.bot1MeanGoals;
		double d2 = g.score2 - summary.bot2MeanGoals;
		var1 += d1 * d1;
		var2 += d2 * d2;
		sumDiff += (g.score2 - g.score1);
	}
	summary.bot1StdGoals = n > 1 ? static_cast<float>(std::sqrt(var1 / (n - 1))) : 0.f;
	summary.bot2StdGoals = n > 1 ? static_cast<float>(std::sqrt(var2 / (n - 1))) : 0.f;

	summary.meanGoalDiff = static_cast<float>(sumDiff / n);
	for (int d : diffs) {
		double delta = d - summary.meanGoalDiff;
		varDiff += delta * delta;
	}
	summary.stdGoalDiff = n > 1 ? static_cast<float>(std::sqrt(varDiff / (n - 1))) : 0.f;

	double seDiff = (n > 1) ? (summary.stdGoalDiff / std::sqrt(static_cast<double>(n))) : 0.0;
	double z95 = 1.95996398454;
	summary.goalDiffCiLower95 = static_cast<float>(summary.meanGoalDiff - z95 * seDiff);
	summary.goalDiffCiUpper95 = static_cast<float>(summary.meanGoalDiff + z95 * seDiff);

	if (seDiff > 1e-7) {
		double tStat = summary.meanGoalDiff / seDiff;
		summary.pairedTPValue = StudentTTwoTailedPValue(tStat, n - 1);
	} else {
		summary.pairedTPValue = 1.0;
	}

	summary.wilcoxonPValue = WilcoxonSignedRankTest(diffs);

	if (!otDurations.empty()) {
		float totalOtTime = std::accumulate(otDurations.begin(), otDurations.end(), 0.f);
		summary.meanOtDuration = totalOtTime / otDurations.size();
	}

	int totalBlueWins = summary.bot1BlueWins + summary.bot2BlueWins;
	int totalBlueGames = summary.bot1BlueGames + summary.bot2BlueGames;
	summary.blueWinRate = totalBlueGames > 0 ? static_cast<float>(totalBlueWins) / totalBlueGames : 0.5f;
	summary.orangeWinRate = 1.0f - summary.blueWinRate;

	if (!cfg.jsonOutput.empty()) {
		nlohmann::json j;
		j["model1"] = {
			{"label", summary.label1},
			{"path", summary.model1Path.string()},
			{"wins", summary.bot1Wins},
			{"win_rate", summary.bot1WinRate},
			{"total_goals", summary.bot1Goals},
			{"mean_goals", summary.bot1MeanGoals},
			{"std_goals", summary.bot1StdGoals},
			{"blue_wins", summary.bot1BlueWins},
			{"orange_wins", summary.bot1OrangeWins},
			{"ot_wins", summary.bot1OtWins}
		};
		j["model2"] = {
			{"label", summary.label2},
			{"path", summary.model2Path.string()},
			{"wins", summary.bot2Wins},
			{"win_rate", summary.bot2WinRate},
			{"win_rate_ci_95", {summary.winRateCiLower95, summary.winRateCiUpper95}},
			{"total_goals", summary.bot2Goals},
			{"mean_goals", summary.bot2MeanGoals},
			{"std_goals", summary.bot2StdGoals},
			{"blue_wins", summary.bot2BlueWins},
			{"orange_wins", summary.bot2OrangeWins},
			{"ot_wins", summary.bot2OtWins}
		};
		j["comparison"] = {
			{"total_games", summary.totalGames},
			{"ties", summary.ties},
			{"overtime_games", summary.overtimeGames},
			{"mean_ot_duration_sec", summary.meanOtDuration},
			{"mean_goal_diff_m2_minus_m1", summary.meanGoalDiff},
			{"std_goal_diff", summary.stdGoalDiff},
			{"goal_diff_ci_95", {summary.goalDiffCiLower95, summary.goalDiffCiUpper95}},
			{"binomial_test_p_value", summary.binomialPValue},
			{"paired_t_test_p_value", summary.pairedTPValue},
			{"wilcoxon_test_p_value", summary.wilcoxonPValue},
			{"statistically_significant_p05", summary.binomialPValue < 0.05}
		};

		nlohmann::json gamesJson = nlohmann::json::array();
		for (const auto &g : summary.games) {
			nlohmann::json gObj;
			gObj["game_id"] = g.gameId;
			gObj["bot1_score"] = g.score1;
			gObj["bot2_score"] = g.score2;
			gObj["winner"] = g.winner;
			gObj["overtime"] = g.overtime;
			gObj["ot_duration"] = g.overtimeDuration;
			gObj["bot1_side"] = (g.bot1Team == Team::BLUE ? "BLUE" : "ORANGE");
			gamesJson.push_back(gObj);
		}
		j["games"] = gamesJson;

		std::ofstream out(cfg.jsonOutput);
		if (out) {
			out << j.dump(2) << "\n";
			std::printf("Saved full match data to: %s\n", cfg.jsonOutput.c_str());
		}
	}

	return summary;
}

} // namespace Dash
