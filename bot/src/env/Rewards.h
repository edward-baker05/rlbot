#pragma once

#include "../Config.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace Hive {

class TouchEdgeReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		return (player.prev && player.prev->ballTouchedStep) ? 0.f : 1.f;
	}
};

class SpeedToBallReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		const float closing = player.vel.Dot(toBall / len);
		return RS_MAX(0.f, closing) / RLGC::CommonValues::CAR_MAX_SPEED;
	}
};

class TouchGoalAccelReward : public RLGC::Reward {
public:
	explicit TouchGoalAccelReward(float exponent) : exponent(exponent) {}

	float exponent;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep || !state.prev)
			return 0.f;

		const Vec target = (player.team == Team::BLUE)
			? RLGC::CommonValues::ORANGE_GOAL_CENTER
			: RLGC::CommonValues::BLUE_GOAL_CENTER;

		const Vec toGoal = (target - state.ball.pos).Normalized();
		const float now = state.ball.vel.Dot(toGoal);
		const float before = state.prev->ball.vel.Dot(toGoal);

		const float x =
			RS_CLAMP((now - before) / RLGC::Math::KPHToVel(130), -1.f, 1.f);
		return std::copysign(std::pow(std::fabs(x), exponent), x);
	}
};

class AirTouchReward : public RLGC::Reward {
public:
	explicit AirTouchReward(float heightExponent) : heightExponent(heightExponent) {}

	float heightExponent;
	static constexpr float MAX_AIR_TIME = 1.75f;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		const float airTimeFrac = RS_MIN(player.airTime, MAX_AIR_TIME) / MAX_AIR_TIME;
		const float heightFrac =
			RS_MAX(0.f, state.ball.pos.z / RLGC::CommonValues::CEILING_Z);

		return RS_MIN(airTimeFrac, std::pow(heightFrac, heightExponent));
	}
};

class FlipSpeedReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev || !state.prev)
			return 0.f;

		const bool justFlipped = (player.isFlipping || player.hasFlipped) &&
		                         (!player.prev->isFlipping && !player.prev->hasFlipped);
		if (!justFlipped)
			return 0.f;

		if (player.prev->vel.Length() >= RLGC::CommonValues::SUPERSONIC_THRESHOLD)
			return 0.f;

		const Vec toBall = (state.ball.pos - player.pos).Normalized();
		const float closingNow = player.vel.Dot(toBall);
		const float closingPrev = player.prev->vel.Dot(toBall);
		const float deltaClosing = closingNow - closingPrev;

		if (deltaClosing <= 50.f)
			return 0.f;

		return RS_CLAMP(deltaClosing / 500.f, 0.f, 1.f);
	}
};

class TieredPickupBoostReward : public RLGC::Reward {
public:
	float smallPadBase = 0.25f;

	TieredPickupBoostReward(float smallPadBase = 0.25f) : smallPadBase(smallPadBase) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev)
			return 0.f;

		const float delta = player.boost - player.prev->boost;
		if (delta <= 0.f)
			return 0.f;

		const bool isBig = (delta > 25.f);

		if (isBig) {
			return 1.0f * (delta / 100.f);
		} else {
			const float potGain = std::sqrt(player.boost / 100.f) - std::sqrt(player.prev->boost / 100.f);
			return smallPadBase + potGain;
		}
	}
};

inline constexpr float THROTTLE_TOP_SPEED = 1410.f;
inline constexpr float HARSH_LOSS_THRESHOLD = 400.f;

struct RewardSpec {
	std::string name;
	float weight;
	std::function<RLGC::Reward*()> make;
};

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg);

std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

}  // namespace Hive
