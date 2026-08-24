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

// Rewards holding possession of the ball, scored as a contest between the two
// teams' best-placed cars. Every car gets an estimated time to intercept the
// ball -- position enters through the distance, velocity through the closing
// speed -- and the team whose best car gets there sooner is in possession.
// The whole team is paid the team's score, so a second man is credited for a
// possession their teammate is holding.
class PossessionReward : public Reward {
  public:
	// Constant acceleration used to model a car chasing the ball, roughly real
	// throttle acceleration near zero speed. The model has no top-speed cap,
	// so intercept times past ~3000uu come out optimistic; near the ball,
	// which is where this signal has to be sharp, they are close to honest.
	constexpr static float CHASE_ACCEL = 1600.f;

	// Intercept times are clamped to this window before being compared. The
	// floor keeps the ratio below from whipping around when both cars are
	// sitting on the ball; the ceiling keeps a far-away opponent from pinning
	// the reward near its maximum for a whole play.
	constexpr static float MIN_TIME = 0.1f;
	constexpr static float MAX_TIME = 4.0f;

	// Estimated seconds for this car to reach the ball, assuming it holds a
	// straight line and accelerates at CHASE_ACCEL from whatever closing speed
	// it already has. Solves dist = v0*t + a*t^2/2 for the positive root.
	//
	// The discriminant can never go negative, so "won't intersect" needs no
	// special case: a car driving away from the ball simply gets a long time,
	// having to shed its speed and come back first.
	static float TimeToBall(const Player &player, const GameState &state) {
		Vec toBall = state.ball.pos - player.pos;

		float surfaceDist = toBall.Length() - CommonValues::BALL_RADIUS;
		float dist = RS_MAX(0.f, surfaceDist);

		// Closing speed is taken relative to the ball, which is what makes
		// this an intercept rather than a chase: running down a ball rolling
		// away scores worse than the same geometry with it rolling towards us.
		float v0 = (player.vel - state.ball.vel).Dot(toBall.Normalized());

		float t = (-v0 + sqrtf(v0 * v0 + 2 * CHASE_ACCEL * dist)) / CHASE_ACCEL;
		return RS_CLAMP(t, MIN_TIME, MAX_TIME);
	}

	virtual std::vector<float> GetAllRewards(const GameState &state,
											 bool isFinal) override {
		std::vector<float> rewards(state.players.size(), 0.f);

		// Best (shortest) intercept time per team. Demoed cars are skipped so
		// that a team getting demoed cedes possession rather than blanking the
		// term -- their team is left on MAX_TIME, not excluded.
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

		// Possession only means anything as a contest.
		if (!teamExists[0] || !teamExists[1])
			return rewards;

		for (int i = 0; i < state.players.size(); i++) {
			int team = (int)state.players[i].team;

			float selfTime = teamTime[team];
			float oppTime = teamTime[1 - team];

			// The denominator is at least 2*MIN_TIME, so this is always safe.
			// Scale-free, zero at symmetry, and bounded by
			// +-(MAX_TIME - MIN_TIME) / (MAX_TIME + MIN_TIME).
			rewards[i] = (oppTime - selfTime) / (oppTime + selfTime);
		}

		return rewards;
	}
};

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig &cfg);

std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig &cfg);

} // namespace Dash
