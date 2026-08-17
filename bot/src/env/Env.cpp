#include "Env.h"

#include "Obs.h"
#include "Rewards.h"
#include "StateSetters.h"
#include "Terminal.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <algorithm>
#include <stdexcept>

using namespace RLGC;

namespace Hive {

TeamSizes PickTeamSizes(int index, const TrainConfig& cfg) {
	const auto& m = cfg.teamSizes;
	const float total = m.weight1v1 + m.weight2v2 + m.weight3v3 + m.weightAsymmetric;
	if (total <= 0.f)
		throw std::runtime_error("PickTeamSizes(): team size weights sum to zero");

	// Deterministic stratification: map the game index onto [0, total) evenly,
	// so the realised mix matches the configured mix exactly.
	const float pos = (static_cast<float>(index % 1000) / 1000.f) * total;

	float acc = m.weight1v1;
	if (pos < acc)
		return {1, 1};

	acc += m.weight2v2;
	if (pos < acc)
		return {2, 2};

	acc += m.weight3v3;
	if (pos < acc)
		return {3, 3};

	// Asymmetric. Cycle through the uneven pairings so all of them appear,
	// and alternate which side is short so neither team is systematically
	// advantaged.
	static const TeamSizes kAsym[] = {
		{1, 2}, {2, 1},
		{2, 3}, {3, 2},
		{1, 3}, {3, 1},
	};
	const int n = static_cast<int>(sizeof(kAsym) / sizeof(kAsym[0]));
	TeamSizes sizes = kAsym[index % n];

	// Never exceed what the observation reserves room for.
	sizes.blue = std::min(sizes.blue, cfg.maxPlayersPerTeam);
	sizes.orange = std::min(sizes.orange, cfg.maxPlayersPerTeam);
	return sizes;
}

// ---------------------------------------------------------------------------

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
	add(new GroundDribbleState(), w.groundDribble);
	add(new AirDribbleState(), w.airDribble);
	add(new FlipResetState(), w.flipReset);
	add(new DemoState(), w.demo);
	add(new FuzzedKickoffState(), w.kickoff);

	if (setters.empty())
		throw std::runtime_error("BuildGeneralCurriculum(): every curriculum weight is zero");

	return new CombinedState(setters);
}

EnvCreateResult CreateEnv(int index, const TrainConfig& cfg) {
	const TeamSizes sizes = PickTeamSizes(index, cfg);

	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < sizes.blue; i++)
		arena->AddCar(Team::BLUE);
	for (int i = 0; i < sizes.orange; i++)
		arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.arena = arena;
	result.actionParser = new DefaultAction();

	// Every game gets its own obs builder instance (GigaLearn expects one per
	// env), but they are all configured identically -- a fixed width regardless
	// of how many cars this particular game has. That is what lets one policy
	// train on 1s, 2s and 3s simultaneously.
	result.obsBuilder = MakeObsBuilder(cfg.maxPlayersPerTeam).release();

	if (cfg.target == TrainTarget::Kickoff) {
		result.stateSetter = new FuzzedKickoffState();
		result.rewards = BuildKickoffRewards(cfg.kickoffRewards);
		result.terminalConditions = {
			new FirstTouchCondition(),
			new TimeoutCondition(cfg.kickoffTimeoutSeconds),
			new GoalScoreCondition(),
		};
	} else {
		result.stateSetter = BuildGeneralCurriculum(cfg.curriculum);
		result.rewards = BuildGeneralRewards(cfg);
		result.terminalConditions = {
			new NoTouchCondition(cfg.noTouchTimeoutSeconds),
			new GoalScoreCondition(),
		};
	}

	return result;
}

} // namespace Hive
