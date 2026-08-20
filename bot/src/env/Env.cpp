#include "Env.h"

#include "Actions.h"
#include "Curriculum.h"
#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

using namespace RLGC;

namespace Hive {

StateSetter* BuildSpawner(const TrainConfig& cfg) {
	StateSetter* base = (cfg.spawn == TrainConfig::SpawnMode::Curriculum)
		? BuildGeneralCurriculum(cfg.curriculum)
		: new RandomState(true, true, false);

	if (cfg.infiniteBoostChance <= 0.f)
		return base;

	return new InfiniteBoostState(base, cfg.infiniteBoostChance);
}

StateSetter* BuildGeneralCurriculum(const CurriculumWeights& w) {
	return new CurriculumState({
		{new NeutralPlayState(), w.neutralPlay, "NeutralPlay"},
		{new BallContactState(), w.ballContact, "BallContact"},
		{new DefendState(), w.defend, "Defend"},
		{new RecoverState(), w.recover, "Recover"},
		{new StrikeState(), w.strike, "Strike"},
		{new AerialState(), w.aerial, "Aerial"},
		{new AirDribbleState(), w.airDribble, "AirDribble"},
		{new FlipResetState(), w.flipReset, "FlipReset"},
		{new DemoState(), w.demo, "Demo"},
		{new FuzzedKickoffState(), w.kickoff, "Kickoff"},
	});
}

EnvCreateResult CreateEnv(int index, const TrainConfig& cfg) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.arena = arena;
	result.actionParser = MakeActionParser(cfg.maskActions).release();
	result.obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam, cfg.obs).release();
	result.stateSetter = BuildSpawner(cfg);
	result.rewards = BuildGeneralRewards(cfg);
	result.terminalConditions = {
		new NoTouchCondition(cfg.noTouchTimeoutSeconds),
		new GoalScoreCondition(),
	};
	return result;
}

}  // namespace Hive
