#pragma once

#include "../Config.h"

#include "Rewards/ZeroSumReward.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <functional>
#include <string>
#include <vector>

using namespace RLGC;

namespace Dash {

// Whether a reward fires on discrete events (a touch, a goal, a demo) or
// accrues on most steps. Spectate's --rewards probe only arms its auto-pause
// on Event rewards: the Continuous ones swing constantly in normal play and
// would trip it every step, but they are still always displayed.
enum class RewardKind {
	Event,
	Continuous,
};

struct RewardSpec {
	std::string name;
	float weight;
	RewardKind kind;
	std::function<RLGC::Reward *()> make;
};

class DirectionalTouchReward : public Reward {
  public:
	constexpr static float MAX_REWARDED_HIT_VEL = RLGC::Math::KPHToVel(120);

	constexpr static float DIR_OFFSET = 0.f;

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!state.prev || !player.ballTouchedStep)
			return 0;

		Vec deltaVel = state.ball.vel - state.prev->ball.vel;

		bool targetOrangeGoal = player.team == Team::BLUE;

		Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
										 : CommonValues::BLUE_GOAL_BACK;

		Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();

		float hitVel = deltaVel.Length();
		float power = RS_MIN(1.f, hitVel / MAX_REWARDED_HIT_VEL);

		float alignment = ballDirToGoal.Dot(deltaVel.Normalized());
		float dirFactor = (alignment + DIR_OFFSET) / (1.f + DIR_OFFSET);

		return power * dirFactor;
	}
};

class AirFaceBallReward : public Reward {
  public:
	float minHeight;
	float maxHeight;
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	std::vector<float> launchZ;

	AirFaceBallReward(float minHeight = 500.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight) {}

	virtual void Reset(const GameState &initialState) override {
		launchZ.assign(initialState.players.size(), 0.f);
		for (const Player &player : initialState.players)
			launchZ[player.index] = player.pos.z;
	}

	virtual void PreStep(const GameState &state) override {
		launchZ.resize(state.players.size(), 0.f);
		for (const Player &player : state.players)
			if (player.isOnGround)
				launchZ[player.index] = player.pos.z;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.isOnGround || state.ball.pos.z <= minHeight)
			return 0.f;

		if (player.index >= (int)launchZ.size() ||
			launchZ[player.index] > MAX_GROUND_LAUNCH_Z)
			return 0.f;

		if ((player.pos - state.ball.pos).Length() >=
			(player.prev->pos - state.prev->ball.pos).Length())
			return 0.f;

		float heightFactor =
			(state.ball.pos.z - minHeight) / (maxHeight - minHeight);
		heightFactor = RS_CLAMP(heightFactor, 0.f, 1.f);

		Vec dirToBall = (state.ball.pos - player.pos).Normalized();
		float alignment = player.rotMat.forward.Dot(dirToBall);

		return heightFactor * RS_MAX(0.f, alignment);
	}
};

class AirVelToBallReward : public Reward {
  public:
	float minHeight;
	float maxHeight;
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	std::vector<float> launchZ;

	AirVelToBallReward(float minHeight = 500.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight) {}

	virtual void Reset(const GameState &initialState) override {
		launchZ.assign(initialState.players.size(), 0.f);
		for (const Player &player : initialState.players)
			launchZ[player.index] = player.pos.z;
	}

	virtual void PreStep(const GameState &state) override {
		launchZ.resize(state.players.size(), 0.f);
		for (const Player &player : state.players)
			if (player.isOnGround)
				launchZ[player.index] = player.pos.z;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.isOnGround || state.ball.pos.z <= minHeight)
			return 0.f;

		if (player.index >= (int)launchZ.size() ||
			launchZ[player.index] > MAX_GROUND_LAUNCH_Z)
			return 0.f;

		float heightFactor =
			(state.ball.pos.z - minHeight) / (maxHeight - minHeight);
		heightFactor = RS_CLAMP(heightFactor, 0.f, 1.f);

		Vec dirToBall = (state.ball.pos - player.pos).Normalized();
		Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
		float velDot = dirToBall.Dot(normVel);

		return heightFactor * RS_MAX(0.f, velDot);
	}
};

class AirLaunchReward : public Reward {
  public:
	float minHeight;
	constexpr static float MAX_AIR_TIME = 1.5f;
	constexpr static float MAX_REWARDED_Z_VEL = 1000.f;
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	std::vector<float> launchZ;

	AirLaunchReward(float minHeight = 500.f) : minHeight(minHeight) {}

	virtual void Reset(const GameState &initialState) override {
		launchZ.assign(initialState.players.size(), 0.f);
		for (const Player &player : initialState.players)
			launchZ[player.index] = player.pos.z;
	}

	virtual void PreStep(const GameState &state) override {
		launchZ.resize(state.players.size(), 0.f);
		for (const Player &player : state.players)
			if (player.isOnGround)
				launchZ[player.index] = player.pos.z;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.isOnGround || state.ball.pos.z <= minHeight)
			return 0.f;

		if (player.index >= (int)launchZ.size() ||
			launchZ[player.index] > MAX_GROUND_LAUNCH_Z)
			return 0.f;

		if (player.airTimeSinceJump > MAX_AIR_TIME || player.vel.z <= 0.f)
			return 0.f;

		if ((player.pos - state.ball.pos).Length() >=
			(player.prev->pos - state.prev->ball.pos).Length())
			return 0.f;

		Vec dirXY = Vec(state.ball.pos.x - player.pos.x,
						state.ball.pos.y - player.pos.y, 0.f)
						.Normalized();
		Vec fwdXY = Vec(player.rotMat.forward.x, player.rotMat.forward.y, 0.f)
						.Normalized();

		float xyAlign = RS_MAX(0.f, fwdXY.Dot(dirXY));
		float zScale = RS_CLAMP(player.vel.z / MAX_REWARDED_Z_VEL, 0.f, 1.f);

		return xyAlign * zScale;
	}
};

class ImprovedAirTouchReward : public Reward {
  public:
	float minHeight;
	float maxHeight;
	float heightSpan;

	constexpr static float DIR_OFFSET = 1.1f;
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	std::vector<float> launchZ;

	ImprovedAirTouchReward(float minHeight = 250.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight),
		  heightSpan(maxHeight - minHeight) {}

	virtual void Reset(const GameState &initialState) override {
		launchZ.assign(initialState.players.size(), 0.f);
		for (const Player &player : initialState.players)
			launchZ[player.index] = player.pos.z;
	}

	virtual void PreStep(const GameState &state) override {
		launchZ.resize(state.players.size(), 0.f);
		for (const Player &player : state.players)
			if (player.isOnGround)
				launchZ[player.index] = player.pos.z;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.ballTouchedStep || player.isOnGround ||
			state.ball.pos.z <= minHeight)
			return 0;

		if (player.index >= (int)launchZ.size() ||
			launchZ[player.index] > MAX_GROUND_LAUNCH_Z)
			return 0.f;

		float height = sqrt((state.ball.pos.z - minHeight) / heightSpan);
		height = RS_CLAMP(height, 0.f, 1.f);
		if (height <= 0)
			return 0;

		float climb = (player.pos.z - launchZ[player.index]) / heightSpan;
		climb = RS_CLAMP(climb, 0.f, 1.f);

		float touchReward = height * (0.2f + 0.8f * climb);

		if (state.prev) {
			Vec deltaVel = state.ball.vel - state.prev->ball.vel;
			if (deltaVel.Length() > 50.f) {
				bool targetOrangeGoal = player.team == Team::BLUE;
				Vec targetPos = targetOrangeGoal
									? CommonValues::ORANGE_GOAL_BACK
									: CommonValues::BLUE_GOAL_BACK;
				Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
				float alignment = ballDirToGoal.Dot(deltaVel.Normalized());
				touchReward *= alignment;
			}
		}

		return touchReward;
	}
};

class AwkwardContactPenalty : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.worldContact.hasContact)
			return 0;

		float upness = player.rotMat.up.Dot(player.worldContact.contactNormal);
		upness = RS_CLAMP(upness, -1.f, 1.f);

		return -(1.f - upness) / 2.f;
	}
};

class PossessionReward : public Reward {
  public:
	constexpr static float CHASE_ACCEL = 1600.f;

	constexpr static float MIN_TIME = 0.1f;
	constexpr static float MAX_TIME = 4.0f;

	static float TimeToBall(const Player &player, const GameState &state) {
		Vec toBall = state.ball.pos - player.pos;

		float surfaceDist = toBall.Length() - CommonValues::BALL_RADIUS;
		float dist = RS_MAX(0.f, surfaceDist);

		float v0 = (player.vel - state.ball.vel).Dot(toBall.Normalized());

		float t = (-v0 + sqrtf(v0 * v0 + 2 * CHASE_ACCEL * dist)) / CHASE_ACCEL;
		return RS_CLAMP(t, MIN_TIME, MAX_TIME);
	}

	virtual std::vector<float> GetAllRewards(const GameState &state,
											 bool isFinal) override {
		std::vector<float> rewards(state.players.size(), 0.f);

		float teamTime[2] = {MAX_TIME, MAX_TIME};
		bool teamExists[2] = {false, false};

		for (const Player &player : state.players) {
			int team = (int)player.team;
			teamExists[team] = true;

			if (player.isDemoed)
				continue;

			float t = TimeToBall(player, state);
			teamTime[team] = RS_MIN(teamTime[team], t);
		}

		if (!teamExists[0] || !teamExists[1])
			return rewards;

		for (int i = 0; i < state.players.size(); i++) {
			int team = (int)state.players[i].team;

			float selfTime = teamTime[team];
			float oppTime = teamTime[1 - team];

			rewards[i] = (oppTime - selfTime) / (oppTime + selfTime);
		}

		return rewards;
	}
};

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg);

std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig &cfg);

} // namespace Dash
