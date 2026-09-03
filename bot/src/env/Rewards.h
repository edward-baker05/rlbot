#pragma once

#include "../Config.h"

#include "CommunityRewards.h"
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

// Fires on the release of a ground carry, scoring the velocity the flick
// imparted against how much of it points at the net. A vertical pop scores near
// zero on direction; simply dropping the ball scores near zero on power.
class FlickReward : public Reward {
  public:
	constexpr static float MAX_GROUND_Z = 200.f;

	float minDistance, maxVelDiff, powerRef, freeFlickScale;
	int windowSteps, minCarrySteps, maxAirSteps;
	Community::CradleReward cradleReward;

	std::vector<float> pending;
	std::vector<Vec> carryVel;
	std::vector<int> windowLeft, carrySteps, airSteps;
	std::vector<float> bestScore;
	std::vector<bool> latched, wasChallenged;

	FlickReward(float minDistance = 300.f, float maxVelDiff = 300.f,
				float powerRef = RLGC::Math::KPHToVel(120.f),
				float freeFlickScale = 0.5f, int windowSteps = 3,
				int minCarrySteps = 4, int maxAirSteps = 5)
		: minDistance(minDistance), maxVelDiff(maxVelDiff), powerRef(powerRef),
		  freeFlickScale(freeFlickScale), windowSteps(windowSteps),
		  minCarrySteps(minCarrySteps), maxAirSteps(maxAirSteps),
		  cradleReward(0.f) {
		Clear(0);
	}

	void Clear(size_t numPlayers) {
		pending.assign(numPlayers, 0.f);
		carryVel.assign(64, Vec());
		windowLeft.assign(64, 0);
		carrySteps.assign(64, 0);
		airSteps.assign(64, 0);
		bestScore.assign(64, 0.f);
		latched.assign(64, false);
		wasChallenged.assign(64, false);
	}

	virtual void Reset(const GameState &initialState) override {
		Clear(initialState.players.size());
	}

	// Looser than CradleReward: no absolute height ceiling, so the latch
	// survives the jump at the start of a flick.
	bool Attached(const Player &player, const GameState &state) {
		return player.pos.z < state.ball.pos.z &&
			   player.pos.Dist2D(state.ball.pos) <= 170.f &&
			   (player.vel - state.ball.vel).Length() <= maxVelDiff;
	}

	// All of the latch advances here, so Train.cpp's second GetReward call for
	// the metrics row reads the same value instead of consuming the window.
	virtual void PreStep(const GameState &state) override {
		pending.assign(state.players.size(), 0.f);

		for (const Player &player : state.players) {
			const int idx = player.index;
			const int id = player.carId;
			if (idx < 0 || static_cast<size_t>(idx) >= pending.size())
				continue;

			const bool cradled =
				cradleReward.GetReward(player, state, false) > 0.f;

			if (cradled || (latched[id] && Attached(player, state))) {
				const bool grounded =
					player.isOnGround && player.pos.z < MAX_GROUND_Z;
				airSteps[id] =
					grounded ? 0 : (latched[id] ? airSteps[id] + 1 : 1);

				// A carry held this far off the ground is an air dribble, not a
				// ground dribble, so drop it instead of arming the window.
				if (airSteps[id] > maxAirSteps) {
					latched[id] = false;
					carrySteps[id] = 0;
					windowLeft[id] = 0;
					continue;
				}

				carrySteps[id] = latched[id] ? carrySteps[id] + 1 : 1;
				latched[id] = true;
				carryVel[id] = player.vel;
				wasChallenged[id] =
					Community::IsChallenged(player, state, minDistance);
				// A ball that merely bounced off the roof is not a flick, so
				// only a carry held this long arms the payout window.
				windowLeft[id] =
					(carrySteps[id] >= minCarrySteps) ? windowSteps : 0;
				bestScore[id] = 0.f;
				continue;
			}

			latched[id] = false;
			carrySteps[id] = 0;
			if (windowLeft[id] <= 0)
				continue;
			windowLeft[id]--;

			Vec targetGoal =
				(player.team == Team::BLUE) ? ORANGE_GOAL_BACK : BLUE_GOAL_BACK;
			Vec ballDirToGoal = (targetGoal - state.ball.pos).Normalized();

			float power = RS_MIN(
				(state.ball.vel - carryVel[id]).Length() / powerRef, 1.f);
			float dir = ballDirToGoal.Dot(state.ball.vel.Normalized());
			float score = power * RS_MAX(0.f, dir) *
						  (wasChallenged[id] ? 1.f : freeFlickScale);

			// Only the gain over the window's best, so the total paid out per
			// flick is its peak rather than a sum over the separation.
			pending[idx] = RS_MAX(0.f, score - bestScore[id]);
			bestScore[id] = RS_MAX(bestScore[id], score);
		}
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.index < 0 ||
			static_cast<size_t>(player.index) >= pending.size())
			return 0.f;

		return pending[player.index];
	}
};

class AerialReward : public Reward {
  public:
	constexpr static float MAX_GROUND_LAUNCH_Z = 200.f;

	constexpr static float MIN_CLIMB = 150.f;

	constexpr static float MIN_DODGE_CLIMB = 300.f;

	std::vector<float> launchZ;

	virtual void Reset(const GameState &initialState) override {
		// An airborne spawn has no real launch height, so it is left at the
		// floor; seeding it from the spawn z instead trips the
		// MAX_GROUND_LAUNCH_Z check and kills the reward for the whole episode.
		launchZ.assign(initialState.players.size(), 0.f);
		for (const Player &player : initialState.players)
			if (player.isOnGround)
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

class AerialDistanceReward : public Reward {
  public:
	constexpr static float RAMP_HEIGHT = 256.f;

	float touchHeightWeight = 1.f;
	float carDistanceWeight = 1.f;
	float ballDistanceWeight = 1.f;
	float distanceNormalisation = 1 / BACK_WALL_Y;
	std::vector<float> distances;

	std::vector<float> pending;

	int chainCarId = -1;

	AerialDistanceReward(float _touchHeightWeight = 1.f,
						 float _carDistanceWeight = 1.f,
						 float _ballDistanceWeight = 2.f)
		: touchHeightWeight(_touchHeightWeight),
		  carDistanceWeight(_carDistanceWeight),
		  ballDistanceWeight(_ballDistanceWeight) {}

	virtual void Reset(const GameState &initialState) override {
		distances.assign(initialState.players.size(), 0.f);
		pending.assign(initialState.players.size(), 0.f);
		chainCarId = -1;
	}

	virtual void PreStep(const GameState &state) override {
		distances.resize(state.players.size(), 0.f);
		pending.assign(state.players.size(), 0.f);

		for (const Player &player : state.players) {
			int idx = player.index;
			int id = (int)player.carId;

			if (idx < 0 || static_cast<size_t>(idx) >= pending.size())
				continue;

			if (chainCarId == id) {
				if (player.pos.z < RAMP_HEIGHT) {
					distances[idx] = 0;
					chainCarId = -1;
				} else if (state.prev && player.prev) {
					float distCar = player.pos.Dist(player.prev->pos);
					float distBall = state.ball.pos.Dist(state.prev->ball.pos);
					distances[idx] += (distCar * carDistanceWeight +
									   distBall * ballDistanceWeight);
				}
			}

			if (player.ballTouchedStep) {
				if (chainCarId == id) {
					pending[idx] += distances[idx] * distanceNormalisation;
				} else {
					float w1 = carDistanceWeight;
					float w2 = ballDistanceWeight;
					if (w1 == 0.f && w2 == 0.f)
						w1 = w2 = 1.f;

					float touchHeight =
						(w1 * player.pos.z + w2 * state.ball.pos.z) / (w1 + w2);
					touchHeight = RS_MAX(0.f, touchHeight - RAMP_HEIGHT);
					pending[idx] +=
						touchHeight * distanceNormalisation * touchHeightWeight;
					chainCarId = id;
				}
				distances[idx] = 0;
			}
		}
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.index < 0 ||
			static_cast<size_t>(player.index) >= pending.size())
			return 0.f;

		return pending[player.index];
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
	float minHeight;
	float maxHeight;
	float heightSpan;

	ImprovedAirTouchReward(float minHeight = 250.f, float maxHeight = 1800.f)
		: minHeight(minHeight), maxHeight(maxHeight),
		  heightSpan(maxHeight - minHeight) {}

	bool IsAirTouch(const Player &player, const GameState &state) const {
		return player.ballTouchedStep && IsAerialing(player) &&
			   state.ball.pos.z > minHeight;
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

	std::vector<float> pending;

	static float TimeToBall(const Player &player, const GameState &state) {
		Vec toBall = state.ball.pos - player.pos;

		float surfaceDist = toBall.Length() - CommonValues::BALL_RADIUS;
		float dist = RS_MAX(0.f, surfaceDist);

		float v0 = (player.vel - state.ball.vel).Dot(toBall.Normalized());

		float t = (-v0 + sqrtf(v0 * v0 + 2 * CHASE_ACCEL * dist)) / CHASE_ACCEL;
		return RS_CLAMP(t, MIN_TIME, MAX_TIME);
	}

	virtual void Reset(const GameState &initialState) override {
		pending.assign(initialState.players.size(), 0.f);
	}

	virtual void PreStep(const GameState &state) override {
		pending.assign(state.players.size(), 0.f);

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
			return;

		for (const Player &player : state.players) {
			if (player.index < 0 ||
				static_cast<size_t>(player.index) >= pending.size())
				continue;

			int team = (int)player.team;
			float selfTime = teamTime[team];
			float oppTime = teamTime[1 - team];

			pending[player.index] = (oppTime - selfTime) / (oppTime + selfTime);
		}
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.index < 0 ||
			static_cast<size_t>(player.index) >= pending.size())
			return 0.f;

		return pending[player.index];
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
