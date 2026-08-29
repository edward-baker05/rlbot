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

using namespace CommonValues;

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

inline bool onTarget(const GameState &state, Team goalTeam,
					 bool checkZ = true) {
	float goalY =
		(goalTeam == Team::ORANGE) ? ORANGE_GOAL_CENTER.y : BLUE_GOAL_CENTER.y;
	float dy = goalY - state.ball.pos.y;

	if (dy * state.ball.vel.y <= 0.f)
		return false;

	float t = dy / state.ball.vel.y;

	if (std::abs(state.ball.pos.x + state.ball.vel.x * t) >=
		GOAL_WIDTH_FROM_CENTER - BALL_RADIUS)
		return false;

	if (!checkZ)
		return true;

	float projZ =
		state.ball.pos.z + state.ball.vel.z * t + 0.5f * GRAVITY_Z * t * t;

	return projZ < GOAL_HEIGHT;
}

// Should return power * direction when ball is hit
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

		if (dirFactor < 0)
			return 0;

		return power * dirFactor;
	}
};

class PadAwarePickupBoostReward : public Reward {
  public:
	constexpr static float SMALL_PAD_REWARD = 0.45f;

	constexpr static float FULL_BOOST = 99.99f;

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.prev || player.boost <= player.prev->boost)
			return 0.f;

		if (player.boost < FULL_BOOST)
			return SMALL_PAD_REWARD * (1 + state.ball.pos.IsZero());

		return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
	}
};

// Should punish ball moving towards our goal if previous touch by opponent
class ConditionalVelocityBallToGoalReward : public Reward {
  public:
	bool ownGoal;
	ConditionalVelocityBallToGoalReward(bool ownGoal = false)
		: ownGoal(ownGoal) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!state.prev)
			return 0.f;

		bool targetOrangeGoal = player.team == Team::BLUE;
		if (ownGoal)
			targetOrangeGoal = !targetOrangeGoal;

		Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
										 : CommonValues::BLUE_GOAL_BACK;

		Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
		return RS_CLAMP(
			ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED),
			0.f, 1.f);
	}
};

// Unwired, as are GoalsidePunishment/ImprovedSaveReward/ShotOnTargetReward.
class OwnGoalThreatPunishment : public Reward {
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ownGoal =
			(player.team == Team::BLUE) ? BLUE_GOAL_BACK : ORANGE_GOAL_BACK;
		Vec ballDirToGoal = (ownGoal - state.ball.pos).Normalized();
		return RS_MAX(0.f, ballDirToGoal.Dot(state.ball.vel) / BALL_MAX_SPEED);
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

	AirFaceBallReward(float minHeight = 300.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!state.prev || !player.prev)
			return 0.f;

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

	AirVelToBallReward(float minHeight = 300.f, float maxHeight = 1800.f)
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

		return heightFactor * heightFactor * RS_MAX(0.f, velDot);
	}
};

class AirLaunchReward : public AerialReward {
  public:
	float minHeight;
	constexpr static float MAX_AIR_TIME = 1.5f;
	constexpr static float MAX_REWARDED_Z_VEL = 1000.f;

	AirLaunchReward(float minHeight = 300.f) : minHeight(minHeight) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!state.prev || !player.prev)
			return 0.f;

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
	constexpr static float STREAK_GROWTH = 1.1f;
	constexpr static int MAX_STREAK = 15;

	float minHeight;
	float maxHeight;
	float heightSpan;

	// Air touches made by each player since they were last on a surface.
	std::vector<int> touchStreak;

	ImprovedAirTouchReward(float minHeight = 250.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight),
		  heightSpan(maxHeight - minHeight) {}

	bool IsAirTouch(const Player &player, const GameState &state) const {
		return player.ballTouchedStep && IsAerialing(player) &&
			   state.ball.pos.z > minHeight;
	}

	virtual void Reset(const GameState &initialState) override {
		AerialReward::Reset(initialState);
		touchStreak.assign(initialState.players.size(), 0);
	}

	virtual void PreStep(const GameState &state) override {
		AerialReward::PreStep(state);
		touchStreak.resize(state.players.size(), 0);

		for (const Player &player : state.players) {
			if (player.index < 0 ||
				static_cast<size_t>(player.index) >= touchStreak.size())
				continue;

			if (player.isOnGround)
				touchStreak[player.index] = 0;
			else if (IsAirTouch(player, state))
				touchStreak[player.index] =
					RS_MIN(touchStreak[player.index] + 1, MAX_STREAK);
		}
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!IsAirTouch(player, state))
			return 0;

		float height = (state.ball.pos.z - minHeight) / heightSpan;
		height = RS_CLAMP(height, 0.f, 1.f);
		if (height <= 0)
			return 0;

		float climb = GetClimb(player) / heightSpan;
		climb = RS_CLAMP(climb, 0.f, 1.f);

		float touchReward = height * (0.5f + 0.5f * climb);

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

		int streak = 1;
		if (player.index >= 0 &&
			static_cast<size_t>(player.index) < touchStreak.size())
			streak = RS_MAX(1, touchStreak[player.index]);

		touchReward *= powf(STREAK_GROWTH, (float)(streak - 1));

		return RS_CLAMP((player.HasFlipOrJump() / 2.f + 0.5f) * touchReward,
						-1.f, 1.f);
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

// Should roughly reward being in a better position to get to the ball. Known
// weak on dribbles, where the ball is technically moving away from the carrier.
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

class GoalsidePunishment : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		Team oppTeam = RS_OPPOSITE_TEAM(player.team);
		float goalY = (oppTeam == Team::ORANGE) ? ORANGE_GOAL_CENTER.y
												: BLUE_GOAL_CENTER.y;
		return std::abs(player.pos.y - goalY) <
			   std::abs(state.ball.pos.y - goalY);
	}
};

class ImprovedSaveReward : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		if (!state.prev ||
			!(player.ballTouchedStep &&
			  state.prev->ball.vel.Length() > RLGC::Math::KPHToVel(60.f)))
			return 0.f;

		return (onTarget(*state.prev, player.team, false) &&
				!onTarget(state, player.team, false));
	}
};

class ShotOnTargetReward : public Reward {
  public:
	const float MIN_SHOT_THRESHOLD = RLGC::Math::KPHToVel(80.f);

	constexpr static float RE_ARM_SECONDS = 0.5f;

	bool armed[2] = {true, true};
	float offTargetTime[2] = {RE_ARM_SECONDS, RE_ARM_SECONDS};
	std::vector<bool> paying;

	bool IsLiveShot(const GameState &state, Team shootingTeam) const {
		return state.ball.vel.Length() >= MIN_SHOT_THRESHOLD &&
			   onTarget(state, RS_OPPOSITE_TEAM(shootingTeam), false);
	}

	virtual void Reset(const GameState &initialState) override {
		for (int team = 0; team < 2; team++) {
			armed[team] = true;
			offTargetTime[team] = RE_ARM_SECONDS;
		}
		paying.assign(initialState.players.size(), false);
	}

	virtual void PreStep(const GameState &state) override {
		paying.assign(state.players.size(), false);

		for (int team = 0; team < 2; team++) {
			if (IsLiveShot(state, (Team)team)) {
				offTargetTime[team] = 0.f;
				continue;
			}

			offTargetTime[team] += state.deltaTime;
			if (offTargetTime[team] >= RE_ARM_SECONDS)
				armed[team] = true;
		}

		for (const Player &player : state.players)
			if (player.ballTouchedStep)
				armed[(int)RS_OPPOSITE_TEAM(player.team)] = true;

		for (const Player &player : state.players) {
			const int team = (int)player.team;
			if (!armed[team] || !player.ballTouchedStep ||
				!IsLiveShot(state, player.team))
				continue;

			armed[team] = false;
			if (player.index >= 0 &&
				static_cast<size_t>(player.index) < paying.size())
				paying[player.index] = true;
		}
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.index < 0 ||
			static_cast<size_t>(player.index) >= paying.size())
			return 0.f;

		return paying[player.index] ? 1.f : 0.f;
	}
};

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg);
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig &cfg);

} // namespace Dash
