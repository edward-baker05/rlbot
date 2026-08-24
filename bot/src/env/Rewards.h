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

struct RewardSpec {
	std::string name;
	float weight;
	std::function<RLGC::Reward *()> make;
};

class DirectionalTouchReward : public Reward {
  public:
	constexpr static float MAX_REWARDED_HIT_VEL = RLGC::Math::KPHToVel(110);

	constexpr static float DIR_OFFSET = 1.1f;

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
		// TODO Uncomment this at a later time
		// float power = RS_MIN(1.f, hitVel / MAX_REWARDED_HIT_VEL);
		float power = sqrt(RS_MIN(1.f, hitVel / MAX_REWARDED_HIT_VEL));

		float alignment = ballDirToGoal.Dot(deltaVel.Normalized());
		float dirFactor = (alignment + DIR_OFFSET) / (1.f + DIR_OFFSET);

		return power * dirFactor;
	}
};

class ImprovedAirTouchReward : public Reward {
  public:
	constexpr static float MIN_HEIGHT = 100.f;
	constexpr static float MAX_HEIGHT = 1800.f;
	constexpr static float HEIGHT_SPAN = MAX_HEIGHT - MIN_HEIGHT;

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

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.ballTouchedStep || player.isOnGround)
			return 0;

		// TODO Swap these back at some point
		// float height = (state.ball.pos.z - MIN_HEIGHT) / HEIGHT_SPAN;
		float height = sqrt((state.ball.pos.z - MIN_HEIGHT) / HEIGHT_SPAN);
		height = RS_CLAMP(height, 0.f, 1.f);
		if (height <= 0)
			return 0;

		float climb = (player.pos.z - launchZ[player.index]) / HEIGHT_SPAN;
		climb = RS_CLAMP(climb, 0.f, 1.f);

		return height * (0.3 + 0.7 * climb);
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
