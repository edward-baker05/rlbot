#include "Env.h"

#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

static StateSetter* BuildGeneralCurriculum(const CurriculumWeights& w) {
	// CombinedState picks one child per episode reset, weighted. Zero-weight
	// entries are dropped rather than passed through, because CombinedState
	// treats weights as a cumulative distribution and a zero-weight setter can
	// still be selected at an exact boundary.
	std::vector<std::pair<StateSetter*, float>> setters;

	auto add = [&](StateSetter* s, float weight) {
		if (weight > 0.f)
			setters.emplace_back(s, weight);
		else
			delete s;
	};

	add(new NeutralPlayState(), w.neutralPlay);
	add(new BallContactState(), w.ballContact);
	add(new DefendState(), w.defend);
	add(new RecoverState(), w.recover);
	add(new AerialState(), w.aerial);
	add(new AirDribbleState(), w.airDribble);
	add(new FlipResetState(), w.flipReset);
	add(new DemoState(), w.demo);
	add(new FuzzedKickoffState(), w.kickoff);

	if (setters.empty())
		throw std::runtime_error("BuildGeneralCurriculum(): every curriculum weight is zero");

	return new CombinedState(setters);
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
