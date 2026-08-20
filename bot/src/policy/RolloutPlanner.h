#pragma once

#include <RLGymCPP/Framework.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Math.h>

#include <vector>

namespace Hive {

struct PlannerConfig {
	int horizonTicks = 12;
	int numCandidates = 32;
	float temperature = 0.3f;

	float goalWeight = 1000.f;
	float ownGoalPenalty = -2000.f;
	float ballVelWeight = 2.5f;
	float ballProximityWeight = 0.5f;
	float touchWeight = 50.f;
	float recoveryWeight = 15.f;
	float ownGoalVetoPenalty = -1500.f;
};

class RolloutPlanner {
public:
	explicit RolloutPlanner(const PlannerConfig& config = {});
	~RolloutPlanner();

	RolloutPlanner(const RolloutPlanner&) = delete;
	RolloutPlanner& operator=(const RolloutPlanner&) = delete;

	RLGC::Action PlanAction(const RLGC::GameState& state,
	                        const RLGC::Player& player,
	                        const RLGC::Action& baseAction);

	float EvaluateRollout(RocketSim::Arena* arena,
	                      RocketSim::Car* car,
	                      RocketSim::Team team,
	                      const RocketSim::Vec& initialBallPos,
	                      bool hadContact,
	                      bool goalScored,
	                      RocketSim::Team scoringTeam) const;

	std::vector<RLGC::Action> GenerateCandidates(const RLGC::Action& baseAction,
	                                             const RLGC::Player& player) const;

	const PlannerConfig& GetConfig() const { return config; }
	void SetConfig(const PlannerConfig& cfg) { config = cfg; }

private:
	PlannerConfig config;
	RocketSim::Arena* simArena = nullptr;
	RocketSim::Car* simBlueCar = nullptr;
	RocketSim::Car* simOrangeCar = nullptr;

	void EnsureArenaInitialized();
};

}  // namespace Hive
