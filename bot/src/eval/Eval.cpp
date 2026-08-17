#include "Eval.h"

#include "../Config.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace RLGC;

namespace Hive {

EvalResult RunEval(const EvalConfig& ecfg) {
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	if (ecfg.seed >= 0)
		srand(static_cast<unsigned>(ecfg.seed));

	TrainConfig cfg = {};
	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam);
	auto obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam);
	DefaultAction parser;

	Policy blue(obsBuilder.get(), obsSize, &parser, cfg.modelShape, ecfg.useGPU);
	Policy orange(obsBuilder.get(), obsSize, &parser, cfg.modelShape, ecfg.useGPU);
	blue.Load(ecfg.blueModel);
	orange.Load(ecfg.orangeModel);

	EvalResult res = {};
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

		arena->ResetToRandomKickoff();

		struct Held { Action queued = {}, applied = {}; };
		Held held[2];

		const int totalTicks = static_cast<int>(ecfg.maxSeconds * 120.f);
		for (int tick = 0; tick < totalTicks; tick += cfg.tickSkip) {
			GameState gs(arena);
			if (tick == 0)
				assert(gs.players[0].team == Team::BLUE);

			auto actBlue = blue.InferBatch({gs.players[0]}, {gs}, true);
			auto actOrange = orange.InferBatch({gs.players[1]}, {gs}, true);
			held[0].queued = actBlue[0];
			held[1].queued = actOrange[0];

			// actionDelay: hold the old action for the first actionDelay ticks
			// of this window, then apply the fresh one -- the same cadence the
			// policy trained with and HivemindBot::update replays.
			arena->Step(cfg.actionDelay);
			held[0].applied = held[0].queued;
			held[1].applied = held[1].queued;
			blueCar->controls = (CarControls)held[0].applied;
			orangeCar->controls = (CarControls)held[1].applied;
			arena->Step(cfg.tickSkip - cfg.actionDelay);

			if (goalFlag.scored) {
				if (goalFlag.team == Team::BLUE) scoreBlue++;
				else scoreOrange++;
				goalFlag.scored = false;
				arena->ResetToRandomKickoff();
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
	return res;
}

} // namespace Hive
