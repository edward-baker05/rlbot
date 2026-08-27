#include "WinMatrix.h"
#include "MatchBench.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace Dash {

namespace {

// Numerically-named subdirectories of policy_versions/, ascending.
std::vector<fs::path> FindPolicyVersions(const fs::path &runFolder,
                                         std::vector<uint64_t> &outSteps) {
	std::vector<std::pair<uint64_t, fs::path>> found;
	const fs::path versionsDir = runFolder / "policy_versions";

	std::error_code ec;
	if (!fs::is_directory(versionsDir, ec))
		return {};

	for (const auto &entry : fs::directory_iterator(versionsDir, ec)) {
		if (!entry.is_directory())
			continue;
		const std::string name = entry.path().filename().string();
		if (name.empty() ||
		    !std::all_of(name.begin(), name.end(),
		                 [](unsigned char c) { return std::isdigit(c) != 0; }))
			continue;
		found.emplace_back(std::stoull(name), entry.path());
	}

	std::sort(found.begin(), found.end(),
	          [](const auto &a, const auto &b) { return a.first < b.first; });

	std::vector<fs::path> paths;
	outSteps.clear();
	for (auto &p : found) {
		outSteps.push_back(p.first);
		paths.push_back(p.second);
	}
	return paths;
}

// Evenly-spaced subsample that always keeps the first and last version.
//
// Cost is quadratic in the count -- 16 versions is 120 pairs -- so the default
// trims to something that finishes in a sitting. Keeping the endpoints matters:
// oldest-vs-newest is the single most informative cell in the matrix.
template <typename T>
void SubsampleEvenly(std::vector<T> &v, int keep) {
	if (keep <= 0 || static_cast<int>(v.size()) <= keep)
		return;
	std::vector<T> out;
	out.reserve(keep);
	for (int k = 0; k < keep; k++) {
		const double f = (keep == 1) ? 0.0
		                             : static_cast<double>(k) / (keep - 1);
		out.push_back(v[static_cast<size_t>(f * (v.size() - 1) + 0.5)]);
	}
	v = std::move(out);
}

// Obs mode in effect at a given timestep, from the run's CONFIG_HISTORY.json.
//
// This guard exists because the t3 lineage silently changed obs mode (Advanced ->
// Predictive at 1.0265B) partway through. LoadBotSpecFromConfig resolves a saved
// version's spec from the RUN's current CONFIG.json, so without this every
// pre-switch version would be loaded under the current obs builder -- same tensor
// shapes, different meaning, and no error anywhere. Every matrix cell involving
// such a version would be quietly meaningless, which is worse than not having the
// tool at all.
//
// Returns an empty string if the history is missing or unreadable, in which case
// the caller proceeds and says so rather than refusing to run.
std::string ObsModeAtStep(const fs::path &runFolder, uint64_t steps) {
	const fs::path histPath = runFolder / "CONFIG_HISTORY.json";
	std::error_code ec;
	if (!fs::exists(histPath, ec))
		return {};

	try {
		std::ifstream in(histPath);
		nlohmann::json j;
		in >> j;
		if (!j.is_array())
			return {};

		// The history is not guaranteed monotonic in timesteps -- a resumed run can
		// append an entry with a lower start than its predecessor -- so sort rather
		// than scanning in file order.
		std::vector<std::pair<uint64_t, std::string>> timeline;
		for (const auto &e : j) {
			if (!e.contains("config") || !e["config"].contains("env"))
				continue;
			const auto &env = e["config"]["env"];
			if (!env.contains("obs") || !env["obs"].is_string())
				continue;
			uint64_t at = e.value("total_timesteps_at_start", 0ull);
			timeline.emplace_back(at, env["obs"].get<std::string>());
		}
		std::sort(timeline.begin(), timeline.end(),
		          [](const auto &a, const auto &b) { return a.first < b.first; });

		std::string mode;
		for (const auto &t : timeline) {
			if (t.first <= steps)
				mode = t.second;
			else
				break;
		}
		return mode;
	} catch (const std::exception &) {
		return {};
	}
}

std::string ShortSteps(uint64_t steps) {
	std::ostringstream ss;
	if (steps >= 1'000'000'000ull)
		ss << std::fixed << std::setprecision(2) << (steps / 1e9) << "B";
	else
		ss << std::fixed << std::setprecision(0) << (steps / 1e6) << "M";
	return ss.str();
}

} // namespace

WinMatrixResult RunWinMatrix(const WinMatrixConfig &cfg) {
	WinMatrixResult res = {};

	std::vector<uint64_t> steps;
	std::vector<fs::path> versions = FindPolicyVersions(cfg.runFolder, steps);

	if (versions.size() < 3) {
		std::cout << "Need at least 3 policy versions to build a matrix; found "
		          << versions.size() << " in " << (cfg.runFolder / "policy_versions")
		          << "\n";
		// Below 3 there are no triples, so the cycling question is not even
		// expressible -- fail loudly rather than print a degenerate table.
		return res;
	}

	// Drop versions trained under a different observation than the newest one.
	// See ObsModeAtStep for why this is not optional.
	{
		const std::string newestObs = ObsModeAtStep(cfg.runFolder, steps.back());
		if (newestObs.empty()) {
			std::cout << "WARNING: no readable CONFIG_HISTORY.json; cannot verify that "
			             "these versions\n         share an observation space. Results are "
			             "only meaningful if they do.\n\n";
		} else {
			std::vector<fs::path> keptV;
			std::vector<uint64_t> keptS;
			std::vector<uint64_t> dropped;
			for (size_t k = 0; k < versions.size(); k++) {
				if (ObsModeAtStep(cfg.runFolder, steps[k]) == newestObs) {
					keptV.push_back(versions[k]);
					keptS.push_back(steps[k]);
				} else {
					dropped.push_back(steps[k]);
				}
			}
			if (!dropped.empty()) {
				std::cout << "Excluding " << dropped.size() << " version(s) trained under a "
				          << "different obs than '" << newestObs << "':\n  ";
				for (uint64_t d : dropped)
					std::cout << ShortSteps(d) << " ";
				std::cout << "\n  (their weights would load at the right shape but read the "
				             "wrong features)\n\n";
			}
			versions = std::move(keptV);
			steps = std::move(keptS);
		}

		if (versions.size() < 3) {
			std::cout << "Fewer than 3 comparable versions remain; nothing to measure.\n";
			return res;
		}
	}

	if (cfg.maxVersions > 0) {
		SubsampleEvenly(versions, cfg.maxVersions);
		SubsampleEvenly(steps, cfg.maxVersions);
	}

	const int n = static_cast<int>(versions.size());
	const int totalPairs = n * (n - 1) / 2;

	std::cout << "Win matrix over " << n << " versions from " << cfg.runFolder << "\n"
	          << "  " << totalPairs << " pairs x " << cfg.gamesPerPair
	          << " games = " << (totalPairs * cfg.gamesPerPair) << " games total\n\n";

	res.versionSteps = steps;
	res.tournamentWins.assign(n, 0);

	// Full pairwise round robin. MatchBench already alternates sides within a
	// pairing, so no need to play each pair twice for colour balance.
	int pairIdx = 0;
	std::vector<std::vector<float>> winRate(n, std::vector<float>(n, 0.5f));
	std::vector<std::vector<float>> goalDiff(n, std::vector<float>(n, 0.f));

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			pairIdx++;
			std::cout << "[" << pairIdx << "/" << totalPairs << "] "
			          << ShortSteps(steps[i]) << " vs " << ShortSteps(steps[j])
			          << std::flush;

			MatchRunnerConfig mc = {};
			mc.model1 = versions[i];
			mc.model2 = versions[j];
			mc.label1 = ShortSteps(steps[i]);
			mc.label2 = ShortSteps(steps[j]);
			mc.totalGames = cfg.gamesPerPair;
			mc.arenas = cfg.arenas;
			mc.gameDuration = cfg.gameDuration;
			mc.deterministic = cfg.deterministic;
			mc.useGPU = cfg.useGPU;

			MatchBench bench(mc);
			MatchSummaryResult m = bench.Run();

			WinMatrixEntry e = {};
			e.i = i;
			e.j = j;
			e.winsI = m.bot1Wins;
			e.winsJ = m.bot2Wins;
			e.ties = m.ties;
			e.goalsI = m.bot1Goals;
			e.goalsJ = m.bot2Goals;

			const int decisive = m.bot1Wins + m.bot2Wins;
			e.winRateI = decisive > 0
			                 ? static_cast<float>(m.bot1Wins) / decisive
			                 : 0.5f;

			winRate[i][j] = e.winRateI;
			winRate[j][i] = 1.f - e.winRateI;
			goalDiff[i][j] = static_cast<float>(e.goalsI - e.goalsJ);
			goalDiff[j][i] = -goalDiff[i][j];

			res.pairs.push_back(e);

			std::cout << "  ->  " << m.bot1Wins << "-" << m.bot2Wins;
			if (m.ties)
				std::cout << " (" << m.ties << " tied)";
			std::cout << "\n";
		}
	}

	// Orient each edge. Ties on win rate fall through to goal difference, then to
	// training order, so the tournament is always complete -- the triple counting
	// below assumes exactly one edge per pair.
	auto beats = [&](int a, int b) {
		if (winRate[a][b] != winRate[b][a])
			return winRate[a][b] > winRate[b][a];
		if (goalDiff[a][b] != goalDiff[b][a])
			return goalDiff[a][b] > goalDiff[b][a];
		return a > b; // arbitrary but deterministic: later version wins the coin flip
	};

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (i != j && beats(i, j))
				res.tournamentWins[i]++;

	// Cyclic triples, exactly.
	//
	// In a complete tournament every triple is either transitive or a 3-cycle,
	// and a triple is transitive iff exactly one of its members beats the other
	// two. So summing C(wins_i, 2) counts each transitive triple once, from its
	// dominant member, and the rest are cycles. No enumeration needed.
	res.totalTriples = n * (n - 1) * (n - 2) / 6;
	long long transitive = 0;
	for (int w : res.tournamentWins)
		transitive += static_cast<long long>(w) * (w - 1) / 2;
	res.cyclicTriples = res.totalTriples - static_cast<int>(transitive);

	// Kendall's coefficient of consistency: 1 - cycles / max possible cycles.
	const double maxCycles = (n % 2 == 1)
	                             ? (static_cast<double>(n) * n * n - n) / 24.0
	                             : (static_cast<double>(n) * n * n - 4.0 * n) / 24.0;
	res.consistency = maxCycles > 0
	                      ? static_cast<float>(1.0 - res.cyclicTriples / maxCycles)
	                      : 1.f;

	// Monotonic progress: over pairs in training order, how often the LATER
	// version wins. 1.0 is a strictly improving lineage; 0.5 is a lineage whose
	// later members are no better than its earlier ones.
	int laterWins = 0;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			if (beats(j, i))
				laterWins++;
	res.monotonicity = totalPairs > 0
	                       ? static_cast<float>(laterWins) / totalPairs
	                       : 0.f;

	res.valid = true;

	if (!cfg.jsonOutput.empty()) {
		nlohmann::json j;
		j["run_folder"] = cfg.runFolder.string();
		j["games_per_pair"] = cfg.gamesPerPair;
		j["deterministic"] = cfg.deterministic;
		j["versions"] = res.versionSteps;
		j["cyclic_triples"] = res.cyclicTriples;
		j["total_triples"] = res.totalTriples;
		j["consistency"] = res.consistency;
		j["monotonicity"] = res.monotonicity;
		j["tournament_wins"] = res.tournamentWins;
		for (const auto &e : res.pairs) {
			j["pairs"].push_back({{"i", e.i},
			                      {"j", e.j},
			                      {"steps_i", res.versionSteps[e.i]},
			                      {"steps_j", res.versionSteps[e.j]},
			                      {"wins_i", e.winsI},
			                      {"wins_j", e.winsJ},
			                      {"ties", e.ties},
			                      {"goals_i", e.goalsI},
			                      {"goals_j", e.goalsJ},
			                      {"win_rate_i", e.winRateI}});
		}
		std::ofstream out(cfg.jsonOutput);
		out << j.dump(2) << "\n";
		std::cout << "\nWrote " << cfg.jsonOutput << "\n";
	}

	return res;
}

void PrintWinMatrix(const WinMatrixResult &res) {
	if (!res.valid)
		return;

	const int n = static_cast<int>(res.versionSteps.size());

	std::vector<std::vector<float>> wr(n, std::vector<float>(n, -1.f));
	for (const auto &e : res.pairs) {
		wr[e.i][e.j] = e.winRateI;
		wr[e.j][e.i] = 1.f - e.winRateI;
	}

	std::cout << "\nRow win rate vs column (deterministic play)\n\n";
	std::cout << std::setw(8) << "";
	for (int j = 0; j < n; j++)
		std::cout << std::setw(8) << ShortSteps(res.versionSteps[j]);
	std::cout << std::setw(9) << "wins\n";

	for (int i = 0; i < n; i++) {
		std::cout << std::setw(8) << ShortSteps(res.versionSteps[i]);
		for (int j = 0; j < n; j++) {
			if (i == j)
				std::cout << std::setw(8) << "-";
			else
				std::cout << std::setw(8) << std::fixed << std::setprecision(2) << wr[i][j];
		}
		std::cout << std::setw(9) << res.tournamentWins[i] << "\n";
	}

	std::cout << "\n"
	          << "Cyclic triples:  " << res.cyclicTriples << " / " << res.totalTriples << "\n"
	          << "Consistency:     " << std::fixed << std::setprecision(3) << res.consistency
	          << "   (1.00 = fully transitive, 0.00 = maximally cyclic)\n"
	          << "Monotonicity:    " << std::fixed << std::setprecision(3) << res.monotonicity
	          << "   (1.00 = every later version beats every earlier one, 0.50 = no trend)\n\n";

	// The interpretation is the whole point of the tool, so state it rather than
	// leaving three numbers on the screen.
	if (res.consistency > 0.9f && res.monotonicity > 0.8f) {
		std::cout << "Reading: lineage is transitive and improving. The pool is a ladder;\n"
		             "rating-weighted opponent sampling should behave sensibly on it.\n";
	} else if (res.monotonicity < 0.35f) {
		// Distinct from convergence and much worse. Convergence sits near 0.5 --
		// later versions neither reliably beat nor lose to earlier ones. Down here
		// the ORDER IS REVERSED: training is reliably making the policy worse, and
		// the pool's best member is one of its oldest.
		std::cout << "Reading: REGRESSION, not convergence. Later versions reliably LOSE to\n"
		             "earlier ones -- the lineage has been going backwards, and its strongest\n"
		             "member is among its oldest. This is not a plateau to be escaped with\n"
		             "better opponents; something in the training loop is actively degrading\n"
		             "the policy. Find and fix that before spending compute on anything else.\n"
		             "Start by bisecting the config history for what changed near the peak,\n"
		             "and confirm with --stochastic (see below).\n";
	} else if (res.consistency > 0.9f && res.monotonicity < 0.65f) {
		std::cout << "Reading: transitive but NOT improving -- later versions do not reliably\n"
		             "beat earlier ones. Self-play has converged; more of the same will not\n"
		             "move it. Change something outside the pool (opponents, states, rewards).\n";
	} else if (res.consistency < 0.75f) {
		std::cout << "Reading: substantial CYCLING. Strategies beat each other in a loop, which\n"
		             "a single Elo cannot represent and will average away. Uniform sampling from\n"
		             "this pool trains against a moving target; curate the opponents (PFSP /\n"
		             "rating-weighted sampling) rather than adding more versions.\n";
	} else {
		std::cout << "Reading: mixed. Some non-transitivity, some progress. Worth re-running with\n"
		             "more games per pair before drawing conclusions -- see the caveat below.\n";
	}

	std::cout << "\nCaveats:\n"
	             " - Each cell is a finite sample, so cells near 0.5 mean little and their edge\n"
	             "   orientation is close to a coin flip. The aggregate numbers are sturdier\n"
	             "   than any single cell; raise --games-per-pair before acting on a marginal\n"
	             "   result.\n"
	             " - These are DETERMINISTIC games by default, while training optimises the\n"
	             "   STOCHASTIC policy. Those can disagree. Re-run with --stochastic to check\n"
	             "   that a surprising ordering is not an artefact of argmax play.\n";
}

} // namespace Dash
