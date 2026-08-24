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
	constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110);

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		if (!state.prev)
			return 0;

		if (player.ballTouchedStep) {
			float prevSpeedFrac = RS_MIN(1, state.prev->ball.vel.Length() /
												MAX_REWARDED_BALL_SPEED);
			float curSpeedFrac =
				RS_MIN(1, state.ball.vel.Length() / MAX_REWARDED_BALL_SPEED);

			if (curSpeedFrac > prevSpeedFrac) {
				bool targetOrangeGoal = player.team == Team::BLUE;

				Vec targetPos = targetOrangeGoal
									? CommonValues::ORANGE_GOAL_BACK
									: CommonValues::BLUE_GOAL_BACK;

				Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
				return ballDirToGoal.Dot(
					(state.ball.vel - state.prev->ball.vel) /
					MAX_REWARDED_BALL_SPEED);
			}
		}
		return 0;
	}
};

class ImprovedAirTouchReward : public Reward {
  public:
	constexpr static float MAX_TIME_IN_AIR = 2.f;

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		if (!state.prev)
			return 0;

		float air_time_frac =
			RS_MIN(player.airTime, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR;
		float height_frac = state.ball.pos[2] / RLGC::CommonValues::CEILING_Z;

		return RS_MIN(air_time_frac, height_frac);
	}
};

class PossessionReward : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) {
		float selfDot = player.pos.Dot(state.ball.pos);
		float oppDot;
		for (Player opponent : state.players) {
			if (opponent.team == player.team)
				continue;

			oppDot = RS_MIN(oppDot, opponent.pos.Dot(state.ball.pos));
		}

		return selfDot - oppDot;
	}
};

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg);

std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig &cfg);

} // namespace Dash
