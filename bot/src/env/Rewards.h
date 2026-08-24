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

// Rewards touches by how much velocity they add towards the opposing goal,
// teaching power (delta-v magnitude) and accuracy (alignment to the goal) at
// the same time. Power and direction are kept as separate factors so that the
// direction offset below can never pay out on a touch that added no speed.
class DirectionalTouchReward : public Reward {
  public:
	// Velocity added to the ball that counts as a full-power hit. Same
	// normalizer StrongTouchReward uses for the same job.
	constexpr static float MAX_REWARDED_HIT_VEL = RLGC::Math::KPHToVel(110);

	// Offset applied to the [-1, 1] alignment cosine before rescaling it to
	// [0, 1], so that a solid touch in the wrong direction still pays a little
	// instead of going negative.
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
		float power = RS_MIN(1.f, hitVel / MAX_REWARDED_HIT_VEL);

		// Normalized() returns a zero vector (not NaN) for a zero-length hit,
		// which lands on the midpoint of dirFactor -- harmless, since power is
		// zero in that case anyway.
		float alignment = ballDirToGoal.Dot(deltaVel.Normalized());
		float dirFactor = (alignment + DIR_OFFSET) / (1.f + DIR_OFFSET);

		return power * dirFactor;
	}
};

// Rewards touches high off the floor, but only for the height the car earned
// under its own power. We track each car's Z at its last ground contact --
// and driving up a wall counts as ground contact -- so a wall jump into a
// high ball scores near zero, while the same touch reached by aerialing off
// the floor scores full. Dropping off the ceiling scores zero.
class ImprovedAirTouchReward : public Reward {
  public:
	constexpr static float MIN_HEIGHT = 300.f;  // below this, no reward at all
	constexpr static float MAX_HEIGHT = 1800.f; // full height credit
	constexpr static float HEIGHT_SPAN = MAX_HEIGHT - MIN_HEIGHT;

	// Car Z at each player's last ground contact, by player index. Seeded from
	// the spawn position because RandomState drops half the cars in mid-air,
	// and they shouldn't be credited for height they were handed.
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

		float height = (state.ball.pos.z - MIN_HEIGHT) / HEIGHT_SPAN;
		height = RS_CLAMP(height, 0.f, 1.f);
		if (height <= 0)
			return 0;

		// How much of that height did the car actually climb to get here?
		float climb = (player.pos.z - launchZ[player.index]) / HEIGHT_SPAN;
		climb = RS_CLAMP(climb, 0.f, 1.f);

		return height * climb;
	}
};

// Punishes contact between the car's body and the arena whenever the car is
// not aligned to the surface it hit -- landing on its roof or side, or
// slamming a wall shoulder-first instead of landing on its wheels.
// worldContact is chassis-only (wheels are raycast, not rigid bodies), so a
// car rolling normally on its wheels never triggers this at all.
class AwkwardContactPenalty : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.worldContact.hasContact)
			return 0;

		// contactNormal points out of the surface and towards the car, so this
		// is +1 when the car's wheels face the surface, 0 when it is on its
		// side, and -1 when it is on its roof. Holds against the walls and
		// ceiling just as well as the floor.
		float upness = player.rotMat.up.Dot(player.worldContact.contactNormal);
		upness = RS_CLAMP(upness, -1.f, 1.f);

		return -(1.f - upness) / 2.f;
	}
};

class PossessionReward : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
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
