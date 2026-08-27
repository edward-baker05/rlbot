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

// Should return power * direction when ball is hit
class DirectionalTouchReward : public Reward {
  public:
	constexpr static float MAX_REWARDED_HIT_VEL = RLGC::Math::KPHToVel(120);

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

		return power * RS_MAX(alignment, 0.01f);
	}
};

// Should punish ball moving towards our goal if previous touch by opponent
class ConditionalVelocityBallToGoalReward : public Reward {
  public:
	bool ownGoal;
	ConditionalVelocityBallToGoalReward(bool ownGoal = false)
		: ownGoal(ownGoal) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		for (const Player &other : state.players) {
			if (state.lastTouchCarID == other.carId) {
				if (other.team == player.team)
					return 0;
				break;
			}
		}

		bool targetOrangeGoal = player.team == Team::BLUE;
		if (ownGoal)
			targetOrangeGoal = !targetOrangeGoal;

		Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
										 : CommonValues::BLUE_GOAL_BACK;

		Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
		return RS_CLAMP(
			-ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED),
			-1.f, 0.f);
	}
};

class AerialReward : public Reward {
  public:
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	constexpr static float MIN_CLIMB = 150.f;

	constexpr static float MIN_DODGE_CLIMB = 300.f;

	std::vector<float> launchZ;

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

	float GetClimb(const Player &player) const {
		if (player.isOnGround || player.worldContact.hasContact)
			return -1.f;

		if (player.index < 0 || player.index >= (int)launchZ.size())
			return -1.f;

		float launch = launchZ[player.index];
		if (launch > MAX_GROUND_LAUNCH_Z)
			return -1.f;

		return player.pos.z - launch;
	}

	bool IsAerialing(const Player &player, float minClimb = MIN_CLIMB) const {
		float climb = GetClimb(player);

		if (climb < minClimb)
			return false;

		return !player.hasFlipped || climb >= MIN_DODGE_CLIMB;
	}
};

class AirFaceBallReward : public AerialReward {
  public:
	float minHeight;
	float maxHeight;

	AirFaceBallReward(float minHeight = 500.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!IsAerialing(player) || state.ball.pos.z <= minHeight)
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

class AirVelToBallReward : public AerialReward {
  public:
	float minHeight;
	float maxHeight;

	AirVelToBallReward(float minHeight = 500.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!IsAerialing(player) || state.ball.pos.z <= minHeight)
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

class AirLaunchReward : public AerialReward {
  public:
	float minHeight;
	constexpr static float MAX_AIR_TIME = 1.5f;
	constexpr static float MAX_REWARDED_Z_VEL = 1000.f;

	AirLaunchReward(float minHeight = 500.f) : minHeight(minHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!IsAerialing(player, 0.f) || state.ball.pos.z <= minHeight)
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

		float pitchUp = RS_MAX(0.f, player.rotMat.forward.z);

		return xyAlign * zScale * pitchUp;
	}
};

// Rewards air touches above a certain height relative to their power and
// direction to net
class ImprovedAirTouchReward : public AerialReward {
  public:
	float minHeight;
	float maxHeight;
	float heightSpan;

	ImprovedAirTouchReward(float minHeight = 250.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight),
		  heightSpan(maxHeight - minHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.ballTouchedStep || !IsAerialing(player) ||
			state.ball.pos.z <= minHeight)
			return 0;

		float height = sqrt((state.ball.pos.z - minHeight) / heightSpan);
		height = RS_CLAMP(height, 0.f, 1.f);
		if (height <= 0)
			return 0;

		float climb = GetClimb(player) / heightSpan;
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

// Should punish touching surface with part of the car that isn't wheels
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

// Should roughly reward being in a better position to get to the ball
// TODO Probably worth giving full reward if the player is close to the
// ball, as possession ends up being calculated weirdly on dribbles where
// the ball is technically moving away from you
class PossessionReward : public Reward {
  public:
	constexpr static float CHASE_ACCEL = 1600.f;

	constexpr static float MIN_TIME = 0.1f;
	constexpr static float MAX_TIME = 4.0f;

	static float TimeToBall(const Player &player, const GameState &state) {
		Vec toBall = state.ball.pos - player.pos;
		if (toBall.Length() < 150)
			return MIN_TIME;

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
