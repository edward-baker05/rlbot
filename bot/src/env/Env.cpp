#include "Env.h"

#include "Actions.h"
#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"
#include "StateSetters/CombinedState.h"
#include "StateSetters/FuzzedKickoffState.h"
#include "Terminal.h"

#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

using namespace RLGC;

namespace Dash {

StateSetter *BuildSpawner(const TrainConfig &cfg) {
	StateSetter *base =
		new CombinedState({{new RandomState(true, true, false), 0.9f},
						   {new FuzzedKickoffState(), 0.1f}});

	if (cfg.infiniteBoostChance <= 0.f)
		return base;

	return new InfiniteBoostState(base, cfg.infiniteBoostChance);
}

EnvCreateResult CreateEnv(int index, const TrainConfig &cfg) {
	int playersPerTeam =
		cfg.teamDistribution.SampleTeamSize(index, cfg.numGames);
	Arena *arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.arena = arena;
	result.actionParser = MakeActionParser(cfg.maskActions).release();
	result.obsBuilder =
		MakeObsBuilder(cfg.maxPlayersPerTeam, cfg.obs).release();
	result.stateSetter = BuildSpawner(cfg);
	result.rewards = BuildGeneralRewards(cfg);
	result.terminalConditions = {
		new NoTouchCondition(cfg.noTouchTimeoutSeconds),
		new GoalScoreCondition(),
		new TimeoutCondition(cfg.timeoutSeconds),
	};
	return result;
}

} // namespace Dash
