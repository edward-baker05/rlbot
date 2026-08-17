#include "Env.h"

#include "Curriculum.h"
#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

using namespace RLGC;

namespace Hive {

static StateSetter* BuildGeneralCurriculum(const CurriculumWeights& w) {
	// CurriculumState drops zero-weight entries and remembers which scenario
	// each reset came from, which is what the Scenario/* metrics report.
	return new CurriculumState({
		{new NeutralPlayState(), w.neutralPlay, "NeutralPlay"},
		{new BallContactState(), w.ballContact, "BallContact"},
		{new DefendState(), w.defend, "Defend"},
		{new RecoverState(), w.recover, "Recover"},
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
	result.actionParser = new DefaultAction();
	result.obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam).release();
	result.stateSetter = BuildGeneralCurriculum(cfg.curriculum);
	result.rewards = BuildGeneralRewards(cfg);
	result.terminalConditions = {
		new NoTouchCondition(cfg.noTouchTimeoutSeconds),
		new GoalScoreCondition(),
	};
	return result;
}

} // namespace Hive
