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

// (randBallSpeed, randCarSpeed, carsOnGround) = (true, true, false), which is
// the RandomState configuration Zealan's guide specifies: random positions and
// velocities for both the ball and the cars, with cars spawning airborne on
// half of resets so they learn to land.
StateSetter* BuildSpawner(const TrainConfig& cfg) {
	if (cfg.spawn == TrainConfig::SpawnMode::Curriculum)
		return BuildGeneralCurriculum(cfg.curriculum);

	return new RandomState(true, true, false);
}

StateSetter* BuildGeneralCurriculum(const CurriculumWeights& w) {
	// CurriculumState drops zero-weight entries and remembers which scenario
	// each reset came from, which is what the Scenario/* metrics report.
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
	result.obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam).release();
	result.stateSetter = BuildSpawner(cfg);
	result.rewards = BuildGeneralRewards(cfg);
	result.terminalConditions = {
		new NoTouchCondition(cfg.noTouchTimeoutSeconds),
		new GoalScoreCondition(),
	};
	return result;
}

} // namespace Hive
