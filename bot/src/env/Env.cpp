#include "Env.h"

#include "Actions.h"
#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"
#include "StateSetters/CombinedState.h"
#include "StateSetters/FuzzedKickoffState.h"
#include "Terminal.h"
#include "../opponents/NectoArena.h"

#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

using namespace RLGC;

namespace Dash {

StateSetter *BuildSpawner(const TrainConfig &cfg) {
	StateSetter *groundBase =
		new CombinedState({{new RandomState(true, true, true), 0.7f},
						   {new FuzzedKickoffState(), 0.3f}});

	StateSetter *base = groundBase;

	if (cfg.aerial.aerialSpawnChance > 0.f) {
		float hoverWeight = RS_CLAMP(cfg.aerial.hoverFraction, 0.01f, 0.99f);
		StateSetter *aerialBase = new CombinedState({
			{new AerialHoverState(cfg.aerial.initialBoost), hoverWeight},
			{new HighBallPopUpState(cfg.aerial.initialBoost),
			 1.f - hoverWeight},
		});

		if (cfg.aerial.aerialSpawnChance >= 1.f) {
			base = aerialBase;
		} else {
			base = new CombinedState({
				{groundBase, 1.f - cfg.aerial.aerialSpawnChance},
				{aerialBase, cfg.aerial.aerialSpawnChance},
			});
		}
	}

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

	// Necto arenas are fixed at creation: a static assignment keeps the
	// learner's once-per-iteration player split valid, and pins exactly half the
	// Necto arenas to each side. Owned for the process lifetime, like the parser
	// and obs builder below.
	NectoArenaState *nectoArena = nullptr;
	if (cfg.necto.enabled) {
		Team nectoTeam = Team::ORANGE;
		if (NectoArenaAssignment(index, cfg.necto.arenaFraction, &nectoTeam)) {
			nectoArena = new NectoArenaState();
			nectoArena->active = true;
			nectoArena->nectoTeam = nectoTeam;
			nectoArena->pending.assign(playersPerTeam * 2, RLGC::Action{});
		}
	}

	EnvCreateResult result = {};
	result.arena = arena;
	result.userInfo = nectoArena;
	result.actionParser = MakeActionParser(cfg.maskActions, nectoArena).release();
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
