#include "Eval.h"

#include "../Config.h"
#include "../env/Actions.h"
#include "../env/Env.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"

#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <stdexcept>

using namespace RLGC;

namespace Hive {

namespace {

// A checkpoint's CONFIG.json lives in the label folder, one level above the numbered save.
struct CheckpointConfig {
	ModelShape shape = {};
	ObsMode obs = ObsMode::Relative;
	bool maskActions = false;
	int maxPlayersPerTeam = 1;
	bool found = false;
};

CheckpointConfig ReadCheckpointConfig(const std::filesystem::path& checkpoint) {
	CheckpointConfig out = {};
	const std::filesystem::path configPath = checkpoint.parent_path() / "CONFIG.json";

	std::ifstream fIn(configPath);
	if (!fIn.good())
		return out;

	nlohmann::json j;
	try {
		fIn >> j;
	} catch (const std::exception& e) {
		std::fprintf(stderr, "WARNING: %s is unreadable (%s); assuming current defaults\n",
		             configPath.c_str(), e.what());
		return out;
	}

	if (j.contains("model")) {
		const auto& m = j["model"];
		if (m.contains("addLayerNorm")) out.shape.addLayerNorm = m["addLayerNorm"].get<bool>();
		if (m.contains("policyLayers")) out.shape.policyLayers = m["policyLayers"].get<std::vector<int>>();
		if (m.contains("sharedHeadLayers")) out.shape.sharedHeadLayers = m["sharedHeadLayers"].get<std::vector<int>>();
	}
	if (j.contains("env")) {
		const auto& e = j["env"];
		if (e.contains("obs"))
			out.obs = (e["obs"].get<std::string>() == "Default") ? ObsMode::Default : ObsMode::Relative;
		if (e.contains("maskActions")) out.maskActions = e["maskActions"].get<bool>();
		if (e.contains("maxPlayersPerTeam")) out.maxPlayersPerTeam = e["maxPlayersPerTeam"].get<int>();
	}

	out.found = true;
	return out;
}

const char* ObsModeName(ObsMode mode) {
	return mode == ObsMode::Relative ? "Relative" : "Default";
}

}  // namespace

EvalResult RunEval(const EvalConfig& ecfg) {
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	if (ecfg.seed >= 0)
		srand(static_cast<unsigned>(ecfg.seed));

	TrainConfig cfg = {};

	// Ancestors differ in ModelShape, so each side is built from its own checkpoint's config.
	const CheckpointConfig blueCfg = ReadCheckpointConfig(ecfg.blueModel);
	const CheckpointConfig orangeCfg = ReadCheckpointConfig(ecfg.orangeModel);

	// Each side builds its own pipeline from its own config, so two checkpoints with
	// different observations or action tables can still meet in the same arena.
	auto fnSide = [&](const CheckpointConfig& side, const char* name) {
		if (!side.found)
			std::fprintf(stderr, "WARNING: no CONFIG.json beside the %s checkpoint; assuming current defaults\n", name);
		return side;
	};
	const CheckpointConfig blueSide = fnSide(blueCfg, "blue");
	const CheckpointConfig orangeSide = fnSide(orangeCfg, "orange");

	auto blueObs = MakeObsBuilder(blueSide.maxPlayersPerTeam, blueSide.obs);
	auto orangeObs = MakeObsBuilder(orangeSide.maxPlayersPerTeam, orangeSide.obs);
	auto blueParser = MakeActionParser(blueSide.maskActions);
	auto orangeParser = MakeActionParser(orangeSide.maskActions);

	Policy blue(blueObs.get(), ProbeObsSize(blueSide.maxPlayersPerTeam, blueSide.obs),
	            blueParser.get(), blueSide.shape, ecfg.useGPU);
	Policy orange(orangeObs.get(), ProbeObsSize(orangeSide.maxPlayersPerTeam, orangeSide.obs),
	              orangeParser.get(), orangeSide.shape, ecfg.useGPU);
	blue.Load(ecfg.blueModel);
	orange.Load(ecfg.orangeModel);

	EvalResult res = {};
	// Kickoffs are seeded per reset: unseeded they repeat, and against a deterministic
	// policy that makes every game a replay of the same handful of trajectories.
	int kickoff = 0;
	const int kickoffBase = static_cast<int>(ecfg.seed >= 0 ? ecfg.seed : 0);

	// Random spawns reproduce the TRAINING distribution exactly, InfiniteBoostState included;
	// a kickoff is out-of-distribution for every checkpoint this project has made.
	std::unique_ptr<StateSetter> spawner(ecfg.randomSpawn ? BuildSpawner(cfg) : nullptr);
	NoTouchCondition noTouch(cfg.noTouchTimeoutSeconds);

	auto fnReset = [&](Arena* arena, GameState* gs) {
		if (spawner)
			spawner->ResetArena(arena);
		else
			arena->ResetToRandomKickoff(kickoffBase + kickoff++);
		if (gs) {
			gs->UpdateFromArena(arena, std::vector<Action>(2), nullptr);
			noTouch.Reset(*gs);
		}
	};
	for (int game = 0; game < ecfg.games; game++) {
		Arena* arena = Arena::Create(GameMode::SOCCAR);
		Car* blueCar = arena->AddCar(Team::BLUE);
		Car* orangeCar = arena->AddCar(Team::ORANGE);
		int scoreBlue = 0, scoreOrange = 0;

		struct GoalFlag { bool scored = false; Team team = Team::BLUE; } goalFlag;
		arena->SetGoalScoreCallback(
			[](Arena*, Team team, void* user) {
				auto* g = static_cast<GoalFlag*>(user);
				g->scored = true;
				g->team = team;
			},
			&goalFlag);

		// The observation contains prevAction (RelativeObs.cpp), so the GameState must
		// carry the applied actions forward -- a fresh GameState per step zeroes them.
		GameState gs;
		fnReset(arena, &gs);
		// GameState orders players by arena->_cars, which is NOT AddCar order.
		int blueIdx = -1, orangeIdx = -1;
		for (size_t i = 0; i < gs.players.size(); i++) {
			if (gs.players[i].carId == blueCar->id) blueIdx = static_cast<int>(i);
			else if (gs.players[i].carId == orangeCar->id) orangeIdx = static_cast<int>(i);
		}
		if (blueIdx < 0 || orangeIdx < 0)
			throw std::runtime_error("eval: could not map cars to GameState players");

		const int totalTicks = static_cast<int>(ecfg.maxSeconds * 120.f);
		for (int tick = 0; tick < totalTicks; tick += cfg.tickSkip) {
			auto actBlue = blue.InferBatch({gs.players[blueIdx]}, {gs}, ecfg.deterministic);
			auto actOrange = orange.InferBatch({gs.players[orangeIdx]}, {gs}, ecfg.deterministic);
			std::vector<Action> applied(2);
			applied[blueIdx] = actBlue[0];
			applied[orangeIdx] = actOrange[0];

			gs.ResetBeforeStep();
			arena->Step(cfg.actionDelay);
			blueCar->controls = (CarControls)actBlue[0];
			orangeCar->controls = (CarControls)actOrange[0];
			arena->Step(cfg.tickSkip - cfg.actionDelay);
			gs.UpdateFromArena(arena, applied, nullptr);

			// Training's own terminal, so episodes end -- and respawn -- at the same rate.
			const bool stalled = noTouch.IsTerminal(gs);
			if (gs.goalScored) res.stateGoals++;
			for (const auto& pl : gs.players) {
				res.steps++;
				if (pl.ballTouchedStep) res.touchSteps++;
			}

			if (goalFlag.scored) {
				if (goalFlag.team == Team::BLUE) scoreBlue++;
				else scoreOrange++;
				goalFlag.scored = false;
				res.episodes++;
				fnReset(arena, &gs);
			} else if (ecfg.randomSpawn && stalled) {
				res.episodes++;
				fnReset(arena, &gs);
			}
		}

		res.blueGoals += scoreBlue;
		res.orangeGoals += scoreOrange;
		if (scoreBlue > scoreOrange) res.blueWins++;
		else if (scoreOrange > scoreBlue) res.orangeWins++;
		else res.draws++;

		std::printf("Game %d/%d: blue %d - %d orange\n", game + 1, ecfg.games, scoreBlue, scoreOrange);
		delete arena;
	}

	std::printf("\nEval summary: blue %d wins, orange %d wins, %d draws | goals blue %d - %d orange\n",
	            res.blueWins, res.orangeWins, res.draws, res.blueGoals, res.orangeGoals);
	std::printf("Episodes %d | mean %.1f s | state-goal steps %d | Ball Touch Ratio %.4f\n", res.episodes,
	            res.episodes ? ecfg.games * ecfg.maxSeconds / res.episodes : 0.f, res.stateGoals,
	            res.steps ? static_cast<double>(res.touchSteps) / res.steps : 0.0);
	return res;
}

}  // namespace Hive
