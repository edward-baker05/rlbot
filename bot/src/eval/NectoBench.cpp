#include "NectoBench.h"

#include "../env/Actions.h"
#include "../env/Obs.h"
#include "../opponents/NectoPolicy.h"
#include "../policy/Policy.h"

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace RLGC;

namespace Dash {

namespace {

// Plain Elo, K per goal, lifted from PolicyVersionManager's skill tracker so the
// number is comparable in spirit to the existing Rating/* metric.
//
// The one change: Necto's own rating is FROZEN at 0 rather than drifting. That
// turns a relative pairing into an absolute scale -- "Elo above Necto" -- which
// is the whole point of anchoring to a fixed opponent. With the anchor frozen
// this converges to 400*log10(p/(1-p)); the incremental form is just the robust
// version of that, well defined at p = 0 and p = 1.
void EloUpdate(float &rating, bool won, float k) {
	const float expected = 1.f / (std::pow(10.f, (0.f - rating) / 400.f) + 1.f);
	rating += k * ((won ? 1.f : 0.f) - expected);
}

float LoadRating(const std::filesystem::path &path) {
	std::ifstream in(path);
	if (!in)
		return 0.f;
	try {
		nlohmann::json j;
		in >> j;
		return j.value("elo_vs_necto", 0.f);
	} catch (const std::exception &) {
		return 0.f; // Corrupt or half-written: start over rather than fail a run.
	}
}

void SaveRating(const std::filesystem::path &path, float rating, int64_t games) {
	nlohmann::json j;
	j["elo_vs_necto"] = rating;
	j["decisive_games"] = games;
	std::ofstream out(path);
	if (out)
		out << j.dump(2) << "\n";
}

} // namespace

struct NectoBench::Impl {
	TrainConfig cfg;

	std::vector<Arena *> arenas;
	std::vector<GameState> states;
	std::vector<Team> nectoTeams;
	std::vector<float> episodeTime;

	std::unique_ptr<ObsBuilder> obsBuilder;
	std::unique_ptr<DefaultAction> parser;
	std::unique_ptr<NectoPolicy> necto;
	std::unique_ptr<FuzzedKickoffState> spawner;
	GoalScoreCondition goalScored;

	int obsSize = 0;

	float rating = 0.f;
	int64_t decisiveGames = 0;
	bool ratingLoaded = false;

	explicit Impl(const TrainConfig &cfg) : cfg(cfg) {
		obsSize = ProbeObsSize(cfg.maxPlayersPerTeam, cfg.obs);
		obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam, cfg.obs);
		parser = MakeActionParser(cfg.maskActions);
		spawner = std::make_unique<FuzzedKickoffState>();

		// CPU on purpose, matching the learner-side policy below: this blocks
		// collection while it runs, and the GPU's memory is training's.
		necto = std::make_unique<NectoPolicy>(
			cfg.necto.modelPath, cfg.necto.benchBeta, 0, /*useGPU=*/false);

		// Half the arenas put Necto on each side. Same reasoning as the training
		// assignment: a fixed-side opponent measures a side, not a policy.
		const int n = RS_MAX(cfg.necto.benchArenas, 2);
		for (int i = 0; i < n; i++) {
			Arena *arena = Arena::Create(GameMode::SOCCAR);
			arena->AddCar(Team::BLUE);
			arena->AddCar(Team::ORANGE);
			arenas.push_back(arena);
			nectoTeams.push_back(i % 2 == 0 ? Team::ORANGE : Team::BLUE);
		}
		states.resize(arenas.size());
		episodeTime.assign(arenas.size(), 0.f);
	}

	~Impl() {
		for (Arena *arena : arenas)
			delete arena;
	}

	void ResetArena(size_t a) {
		spawner->ResetArena(arenas[a]);
		states[a] = GameState(arenas[a]);
		states[a].UpdateFromArena(arenas[a],
								  std::vector<Action>(arenas[a]->_cars.size()),
								  nullptr);
		episodeTime[a] = 0.f;
	}
};

NectoBench::NectoBench(const TrainConfig &cfg)
	: impl(std::make_unique<Impl>(cfg)) {}

NectoBench::~NectoBench() = default;

NectoBenchResult NectoBench::Run(const std::filesystem::path &checkpoint) {
	NectoBenchResult result = {};
	if (checkpoint.empty() || !std::filesystem::is_directory(checkpoint))
		return result;

	Impl &s = *impl;
	const TrainConfig &cfg = s.cfg;

	// The GPU belongs to training while a run is going; this is small enough
	// that CPU costs a few seconds every benchInterval iterations.
	Policy policy(s.obsBuilder.get(), s.obsSize, s.parser.get(), cfg.modelShape,
				  false);
	try {
		policy.Load(checkpoint);
	} catch (const std::exception &e) {
		std::cerr << "NectoBench: " << e.what() << "\n";
		return result;
	}

	const std::filesystem::path ratingPath =
		cfg.CheckpointFolder() / "necto_rating.json";
	if (!s.ratingLoaded) {
		s.rating = LoadRating(ratingPath);
		s.ratingLoaded = true;
	}

	for (size_t a = 0; a < s.arenas.size(); a++)
		s.ResetArena(a);

	const float stepTime = static_cast<float>(cfg.tickSkip) / 120.f;
	const int targetDecisive = static_cast<int>(s.arenas.size());
	float totalTime = 0.f;

	// Scratch, reused across steps.
	std::vector<Player> learnerPlayers;
	std::vector<GameState> learnerStates;
	std::vector<std::pair<size_t, int>> learnerSlots;
	std::vector<NectoRequest> nectoRequests;
	std::vector<std::pair<size_t, int>> nectoSlots;
	std::vector<Action> nectoActions;
	std::vector<std::vector<Action>> applied(s.arenas.size());

	while (result.decisive < targetDecisive &&
		   totalTime < cfg.necto.benchMaxSimTime) {

		learnerPlayers.clear();
		learnerStates.clear();
		learnerSlots.clear();
		nectoRequests.clear();
		nectoSlots.clear();

		for (size_t a = 0; a < s.arenas.size(); a++) {
			const GameState &gs = s.states[a];
			for (size_t p = 0; p < gs.players.size(); p++) {
				if (gs.players[p].team == s.nectoTeams[a]) {
					nectoRequests.push_back(
						{&gs, static_cast<int>(p), gs.players[p].prevAction});
					nectoSlots.push_back({a, static_cast<int>(p)});
				} else {
					learnerPlayers.push_back(gs.players[p]);
					learnerStates.push_back(gs);
					learnerSlots.push_back({a, static_cast<int>(p)});
				}
			}
		}

		// Deterministic on both sides: the benchmark wants low variance, which
		// is the opposite of what the training arenas want.
		std::vector<Action> learnerActions =
			policy.InferBatch(learnerPlayers, learnerStates, true);
		s.necto->InferBatch(nectoRequests, nectoActions);

		for (size_t a = 0; a < s.arenas.size(); a++)
			applied[a].assign(s.states[a].players.size(), Action{});
		for (size_t i = 0; i < learnerSlots.size(); i++)
			applied[learnerSlots[i].first][learnerSlots[i].second] =
				learnerActions[i];
		for (size_t i = 0; i < nectoSlots.size(); i++)
			applied[nectoSlots[i].first][nectoSlots[i].second] = nectoActions[i];

		for (size_t a = 0; a < s.arenas.size(); a++) {
			Arena *arena = s.arenas[a];
			s.states[a].ResetBeforeStep();

			// Action delay first, on the controls already on the cars, then the
			// rest of the tick skip on the new ones -- same split the training
			// env uses, so the comparison is like for like.
			arena->Step(cfg.actionDelay);

			auto carItr = arena->_cars.begin();
			for (size_t p = 0; p < applied[a].size(); p++, carItr++)
				(*carItr)->controls = (CarControls)applied[a][p];

			arena->Step(cfg.tickSkip - cfg.actionDelay);
			s.states[a].UpdateFromArena(arena, applied[a], nullptr);
			s.episodeTime[a] += stepTime;
		}

		totalTime += stepTime;

		for (size_t a = 0; a < s.arenas.size(); a++) {
			GameState &gs = s.states[a];
			const Team learnerTeam =
				(s.nectoTeams[a] == Team::BLUE) ? Team::ORANGE : Team::BLUE;

			if (s.goalScored.IsTerminal(gs)) {
				// Ball is in the conceding team's net.
				const Team conceding = RS_TEAM_FROM_Y(gs.ball.pos.y);
				const bool learnerScored = (conceding != learnerTeam);

				result.episodes++;
				result.decisive++;
				if (learnerScored)
					result.goalsFor++;
				else
					result.goalsAgainst++;

				EloUpdate(s.rating, learnerScored, cfg.necto.benchEloK);
				s.decisiveGames++;
				s.ResetArena(a);
			} else if (s.episodeTime[a] >= cfg.necto.benchSimTime) {
				// Nobody scored inside the budget. Counted, but not decisive.
				result.episodes++;
				s.ResetArena(a);
			}
		}
	}

	result.valid = result.episodes > 0;
	result.winRate = result.decisive > 0
						 ? static_cast<float>(result.goalsFor) / result.decisive
						 : 0.f;
	result.elo = s.rating;

	SaveRating(ratingPath, s.rating, s.decisiveGames);
	return result;
}

void ReportNectoBench(const NectoBenchResult &result, GGL::Report &report) {
	if (!result.valid)
		return;

	report["Necto/Bench/Elo"] = result.elo;
	report["Necto/Bench/WinRate"] = result.winRate;
	report["Necto/Bench/GoalsFor"] = result.goalsFor;
	report["Necto/Bench/GoalsAgainst"] = result.goalsAgainst;
	report["Necto/Bench/GoalDiff"] = result.goalsFor - result.goalsAgainst;
	report["Necto/Bench/Episodes"] = result.episodes;
	// Early on nearly every episode times out with nobody able to score. Without
	// this, the win rate above is a mean over a couple of samples reading as
	// noise -- the two together are what make it interpretable.
	report["Necto/Bench/DecisiveRate"] =
		result.episodes > 0
			? static_cast<float>(result.decisive) / result.episodes
			: 0.f;
}

} // namespace Dash
