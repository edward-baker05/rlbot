#include "RolloutPlanner.h"

#include <RLGymCPP/CommonValues.h>

#include <algorithm>
#include <cmath>

using namespace RLGC;

namespace Hive {

RolloutPlanner::RolloutPlanner(const PlannerConfig& config)
	: config(config) {}

RolloutPlanner::~RolloutPlanner() {
	if (simArena) {
		delete simArena;
		simArena = nullptr;
		simBlueCar = nullptr;
		simOrangeCar = nullptr;
	}
}

void RolloutPlanner::EnsureArenaInitialized() {
	if (simArena) return;

	simArena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
	simBlueCar = simArena->AddCar(RocketSim::Team::BLUE);
	simOrangeCar = simArena->AddCar(RocketSim::Team::ORANGE);
}

std::vector<Action> RolloutPlanner::GenerateCandidates(const Action& baseAction,
                                                      const Player& player) const {
	std::vector<Action> candidates;
	candidates.reserve(config.numCandidates);

	candidates.push_back(baseAction);

	auto AddClamped = [&](Action a) {
		a.throttle = std::clamp(a.throttle, -1.f, 1.f);
		a.steer = std::clamp(a.steer, -1.f, 1.f);
		a.pitch = std::clamp(a.pitch, -1.f, 1.f);
		a.yaw = std::clamp(a.yaw, -1.f, 1.f);
		a.roll = std::clamp(a.roll, -1.f, 1.f);
		a.boost = a.boost > 0.5f ? 1.f : 0.f;
		a.jump = a.jump > 0.5f ? 1.f : 0.f;
		a.handbrake = a.handbrake > 0.5f ? 1.f : 0.f;

		if (candidates.size() < static_cast<size_t>(config.numCandidates)) {
			candidates.push_back(a);
		}
	};

	{
		Action a = baseAction;
		a.boost = (baseAction.boost > 0.5f) ? 0.f : 1.f;
		AddClamped(a);
	}

	{
		Action a = baseAction;
		a.jump = (baseAction.jump > 0.5f) ? 0.f : 1.f;
		AddClamped(a);
	}

	{
		Action a = baseAction;
		a.handbrake = (baseAction.handbrake > 0.5f) ? 0.f : 1.f;
		AddClamped(a);
	}

	for (float dSteer : {-0.4f, 0.4f, -0.8f, 0.8f}) {
		Action a = baseAction;
		a.steer += dSteer;
		AddClamped(a);
	}

	for (float dPitch : {-0.4f, 0.4f, -0.8f, 0.8f}) {
		Action a = baseAction;
		a.pitch += dPitch;
		AddClamped(a);
	}

	for (float dRoll : {-0.5f, 0.5f, 1.0f, -1.0f}) {
		Action a = baseAction;
		a.roll += dRoll;
		AddClamped(a);
	}

	for (float throttleVal : {1.0f, 0.0f, -1.0f}) {
		Action a = baseAction;
		a.throttle = throttleVal;
		AddClamped(a);
	}

	{
		Action a = baseAction;
		a.throttle = 1.0f;
		a.boost = 1.0f;
		AddClamped(a);
	}

	{
		Action a = baseAction;
		a.throttle = -1.0f;
		a.boost = 0.0f;
		a.handbrake = 1.0f;
		AddClamped(a);
	}

	if (!player.isOnGround) {
		Action a = {};
		a.throttle = 1.0f;
		a.pitch = (player.rotMat.forward.z > 0.1f) ? -0.5f : (player.rotMat.forward.z < -0.1f ? 0.5f : 0.f);
		a.roll = (player.rotMat.up.z < 0.5f) ? 1.0f : 0.0f;
		AddClamped(a);
	}

	return candidates;
}

float RolloutPlanner::EvaluateRollout(RocketSim::Arena* arena,
                                      RocketSim::Car* car,
                                      RocketSim::Team team,
                                      const Vec& initialBallPos,
                                      bool hadContact,
                                      bool goalScored,
                                      RocketSim::Team scoringTeam) const {
	float score = 0.f;
	const RocketSim::BallState ball = arena->ball->GetState();
	const RocketSim::CarState carState = car->GetState();

	if (goalScored) {
		if (scoringTeam == team) {
			score += config.goalWeight;
		} else {
			score += config.ownGoalPenalty;
		}
	}

	const float teamSign = (team == RocketSim::Team::BLUE) ? 1.f : -1.f;
	const float goalDirectedVel = ball.vel.y * teamSign;
	score += (goalDirectedVel / 1000.f) * config.ballVelWeight;

	if (goalDirectedVel < -400.f) {
		score += (goalDirectedVel / 1000.f) * (config.ballVelWeight * 2.0f);
		if (ball.pos.y * teamSign < -2000.f) {
			score += config.ownGoalVetoPenalty * 0.5f;
		}
	}

	if (hadContact) {
		score += config.touchWeight;
		if (goalDirectedVel > 200.f) {
			score += (goalDirectedVel / 500.f) * 10.f;
		}
	}

	const float distToBall = (ball.pos - carState.pos).Length();
	score -= (distToBall / 1000.f) * config.ballProximityWeight;

	if (carState.isOnGround) {
		score += config.recoveryWeight;
	} else {
		const float upright = carState.rotMat.up.z;
		score += upright * (config.recoveryWeight * 0.5f);
		if (upright < -0.3f) {
			score -= config.recoveryWeight;
		}
	}

	return score;
}

Action RolloutPlanner::PlanAction(const GameState& state,
                                  const Player& player,
                                  const Action& baseAction) {
	if (config.horizonTicks <= 0 || config.numCandidates <= 1) {
		return baseAction;
	}

	EnsureArenaInitialized();

	const auto candidates = GenerateCandidates(baseAction, player);
	if (candidates.empty()) {
		return baseAction;
	}

	RocketSim::Car* ourCar = (player.team == Team::BLUE) ? simBlueCar : simOrangeCar;
	RocketSim::Car* otherCar = (player.team == Team::BLUE) ? simOrangeCar : simBlueCar;

	simArena->ball->SetState(state.ball);
	ourCar->SetState(player);

	bool hasOpponent = false;
	for (const auto& p : state.players) {
		if (p.team != player.team) {
			otherCar->SetState(p);
			hasOpponent = true;
			break;
		}
	}
	if (!hasOpponent) {
		RocketSim::CarState neutralCar = {};
		neutralCar.pos = Vec(0, (player.team == Team::BLUE ? 4000.f : -4000.f), 17.f);
		neutralCar.isOnGround = true;
		otherCar->SetState(neutralCar);
	}

	const RocketSim::BallState snapBall = simArena->ball->GetState();
	const RocketSim::CarState snapOurCar = ourCar->GetState();
	const RocketSim::CarState snapOtherCar = otherCar->GetState();

	struct GoalTracker {
		bool scored = false;
		RocketSim::Team team = RocketSim::Team::BLUE;
	} goalTracker;

	simArena->SetGoalScoreCallback(
		[](RocketSim::Arena*, RocketSim::Team t, void* user) {
			auto* gt = static_cast<GoalTracker*>(user);
			gt->scored = true;
			gt->team = t;
		},
		&goalTracker);

	const uint64_t snapTick = simArena->tickCount;
	std::vector<float> scores(candidates.size(), 0.f);

	for (size_t i = 0; i < candidates.size(); i++) {
		simArena->ball->SetState(snapBall);
		ourCar->SetState(snapOurCar);
		otherCar->SetState(snapOtherCar);
		goalTracker.scored = false;

		ourCar->controls = (RocketSim::CarControls)candidates[i];
		otherCar->controls = RocketSim::CarControls{};

		simArena->Step(config.horizonTicks);

		const auto carState = ourCar->GetState();
		const bool hadContact = (carState.ballHitInfo.isValid && carState.ballHitInfo.tickCountWhenHit >= snapTick) ||
		                        (simArena->ball->GetState().pos - carState.pos).Length() < 180.f;

		scores[i] = EvaluateRollout(simArena, ourCar, static_cast<RocketSim::Team>(player.team),
		                            snapBall.pos, hadContact, goalTracker.scored, goalTracker.team);
	}

	if (config.temperature <= 0.05f) {
		size_t bestIdx = 0;
		float maxScore = scores[0];
		for (size_t i = 1; i < scores.size(); i++) {
			if (scores[i] > maxScore) {
				maxScore = scores[i];
				bestIdx = i;
			}
		}
		return candidates[bestIdx];
	}

	float maxScore = *std::max_element(scores.begin(), scores.end());
	std::vector<float> weights(scores.size(), 0.f);
	float totalWeight = 0.f;

	for (size_t i = 0; i < scores.size(); i++) {
		weights[i] = std::exp((scores[i] - maxScore) / config.temperature);
		totalWeight += weights[i];
	}

	if (totalWeight <= 1e-6f) {
		return candidates[0];
	}

	Action result = {};
	for (size_t i = 0; i < candidates.size(); i++) {
		const float w = weights[i] / totalWeight;
		result.throttle += candidates[i].throttle * w;
		result.steer += candidates[i].steer * w;
		result.pitch += candidates[i].pitch * w;
		result.yaw += candidates[i].yaw * w;
		result.roll += candidates[i].roll * w;
		result.boost += candidates[i].boost * w;
		result.jump += candidates[i].jump * w;
		result.handbrake += candidates[i].handbrake * w;
	}

	result.boost = (result.boost >= 0.5f) ? 1.f : 0.f;
	result.jump = (result.jump >= 0.5f) ? 1.f : 0.f;
	result.handbrake = (result.handbrake >= 0.5f) ? 1.f : 0.f;

	result.throttle = std::clamp(result.throttle, -1.f, 1.f);
	result.steer = std::clamp(result.steer, -1.f, 1.f);
	result.pitch = std::clamp(result.pitch, -1.f, 1.f);
	result.yaw = std::clamp(result.yaw, -1.f, 1.f);
	result.roll = std::clamp(result.roll, -1.f, 1.f);

	return result;
}

}  // namespace Hive
