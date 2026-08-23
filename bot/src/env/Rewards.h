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

// The only place goalward delta-v is computed; every term and metric shares it.
inline float GoalwardDeltaV(const RLGC::GameState& state, Team team) {
	if (!state.prev)
		return 0.f;

	const Vec target = (team == Team::BLUE)
		? RLGC::CommonValues::ORANGE_GOAL_CENTER
		: RLGC::CommonValues::BLUE_GOAL_CENTER;

	const Vec toGoal = (target - state.ball.pos).Normalized();
	const float now = state.ball.vel.Dot(toGoal);
	const float before = state.prev->ball.vel.Dot(toGoal);

	return RS_CLAMP((now - before) / RLGC::Math::KPHToVel(130), -1.f, 1.f);
}

class TouchGoalAccelReward : public RLGC::Reward {
public:
	explicit TouchGoalAccelReward(float exponent) : exponent(exponent) {}

	float exponent;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep || !state.prev)
			return 0.f;

		const float x = GoalwardDeltaV(state, player.team);
		return std::copysign(std::pow(std::fabs(x), exponent), x);
	}
};

// The only place ball-goalward direction is computed, in the GROUND plane.
inline float BallGoalwardCos(const RLGC::GameState& state, Team team) {
	const Vec flatVel = {state.ball.vel.x, state.ball.vel.y, 0.f};
	const float flatSpeed = flatVel.Length();

	// A ball going straight up has no horizontal opinion; 0 maps to neutral below.
	if (flatSpeed < 1e-4f)
		return 0.f;

	const Vec target = (team == Team::BLUE)
		? RLGC::CommonValues::ORANGE_GOAL_CENTER
		: RLGC::CommonValues::BLUE_GOAL_CENTER;

	const Vec toGoal = {target.x - state.ball.pos.x, target.y - state.ball.pos.y, 0.f};
	if (toGoal.Length() < 1e-4f)
		return 1.f;

	return RS_CLAMP(flatVel.Dot(toGoal.Normalized()) / flatSpeed, -1.f, 1.f);
}

// Exponent 1 is the shipped curve; raising it only pulls the middle down.
inline float GoalwardFactor(float goalwardCos, float exponent) {
	const float dir = RS_CLAMP(0.5f + 0.5f * goalwardCos, 0.f, 1.f);
	return exponent == 1.f ? dir : std::pow(dir, exponent);
}

class AirTouchReward : public RLGC::Reward {
public:
	explicit AirTouchReward(float heightExponent, float directionExponent = 1.f)
		: heightExponent(heightExponent), directionExponent(directionExponent) {}

	float heightExponent;
	float directionExponent;
	static constexpr float MAX_AIR_TIME = 1.75f;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		const float airTimeFrac = RS_MIN(player.airTime, MAX_AIR_TIME) / MAX_AIR_TIME;
		const float heightFrac =
			RS_MAX(0.f, state.ball.pos.z / RLGC::CommonValues::CEILING_Z);

		const float base = RS_MIN(airTimeFrac, std::pow(heightFrac, heightExponent));
		if (base <= 0.f)
			return 0.f;

		// Smooth, not a gate: a marginally misaimed aerial must keep its gradient.
		return base * GoalwardFactor(BallGoalwardCos(state, player.team),
									 directionExponent);
	}
};

class FlipSpeedReward : public RLGC::Reward {
public:
	// Prices a flip used as TRAVEL only; contact flips are TouchGoalAccel's.
	static constexpr float MIN_BALL_DIST = 1500.f;

	// RLConst::FLIP_INITIAL_VEL_SCALE -- a dodge is worth +500 uu/s any direction.
	static constexpr float FULL_GAIN = 500.f;
	static constexpr float MIN_GAIN = 50.f;

	// Broadly ball-ward, not precisely; demanding better re-creates the v1 mistake.
	static constexpr float TOWARD_BALL_COS = 0.5f;

	void Reset(const RLGC::GameState& initialState) override { tracked.clear(); }

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev || player.index < 0)
			return 0.f;

		if ((int)tracked.size() <= player.index)
			tracked.resize(player.index + 1);

		Track& t = tracked[player.index];
		const float speed = FlatSpeed(player.vel);

		if (player.isFlipping) {
			if (player.prev->isFlipping) {
				if (t.active)
					t.peakSpeed = RS_MAX(t.peakSpeed, speed);
			} else {
				t.active =
					(state.ball.pos - player.pos).Length() >= MIN_BALL_DIST &&
					player.prev->vel.Length() < RLGC::CommonValues::SUPERSONIC_THRESHOLD;
				t.startSpeed = FlatSpeed(player.prev->vel);
				t.peakSpeed = speed;
			}
			return 0.f;
		}

		if (!player.prev->isFlipping || !t.active)
			return 0.f;

		t.active = false;

		// isFlipping also clears on landing, so the peak is what the flip produced.
		const float gain = RS_MAX(t.peakSpeed, speed) - t.startSpeed;
		if (gain <= MIN_GAIN)
			return 0.f;

		const Vec toBall = state.ball.pos - player.pos;
		const Vec flatToBall = {toBall.x, toBall.y, 0.f};
		const Vec flatVel = {player.vel.x, player.vel.y, 0.f};
		if (flatToBall.Length() < 1e-4f || flatVel.Length() < 1e-4f)
			return 0.f;

		if (flatVel.Normalized().Dot(flatToBall.Normalized()) < TOWARD_BALL_COS)
			return 0.f;

		return RS_CLAMP(gain / FULL_GAIN, 0.f, 1.f);
	}

private:
	struct Track {
		bool active = false;
		float startSpeed = 0.f;
		float peakSpeed = 0.f;
	};

	// The jump's own vertical impulse is not travel, so only the plane counts.
	static float FlatSpeed(const Vec& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

	std::vector<Track> tracked;
};

// The only place a shot is projected; the reward and Shot/* telemetry share it.
struct ShotProjection {
	bool valid = false;   // heading at the target goal, arriving within MAX_TIME
	float missDist = 0.f; // uu outside the mouth at the goal plane, 0 if on target
	float time = 0.f;
};

inline ShotProjection ProjectShot(const RLGC::GameState& state, Team team) {
	constexpr float MAX_TIME = 3.f;

	ShotProjection out = {};

	const float targetY = (team == Team::BLUE)
		? RLGC::CommonValues::BACK_WALL_Y
		: -RLGC::CommonValues::BACK_WALL_Y;

	const float dy = targetY - state.ball.pos.y;
	if (dy * state.ball.vel.y <= 0.f)
		return out;

	const float t = dy / state.ball.vel.y;
	if (t <= 0.f || t > MAX_TIME)
		return out;

	const float x = state.ball.pos.x + state.ball.vel.x * t;

	// The floor clamp is a correction, not an approximation: ground shots score.
	const float z = RS_MAX(
		RLGC::CommonValues::BALL_RADIUS,
		state.ball.pos.z + state.ball.vel.z * t +
			0.5f * RLGC::CommonValues::GRAVITY_Z * t * t);

	const float missX =
		RS_MAX(0.f, std::fabs(x) - RLGC::CommonValues::GOAL_WIDTH_FROM_CENTER);
	const float missZ = RS_MAX(0.f, z - RLGC::CommonValues::GOAL_HEIGHT);

	out.valid = true;
	out.time = t;
	out.missDist = std::sqrt(missX * missX + missZ * missZ);
	return out;
}

class ShotOnTargetReward : public RLGC::Reward {
public:
	// One goal half-width, and never zero, so a wide shot keeps a gradient inward.
	static constexpr float MISS_SCALE = RLGC::CommonValues::GOAL_WIDTH_FROM_CENTER;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		const ShotProjection shot = ProjectShot(state, player.team);
		if (!shot.valid)
			return 0.f;

		// Keyed on this touch's delta-v, not ball speed, so a carry pays ~0.
		const float strength = RS_MAX(0.f, GoalwardDeltaV(state, player.team));
		if (strength <= 0.f)
			return 0.f;

		// A plateau inside the mouth: the corners pay what the centre pays.
		return strength * std::exp(-shot.missDist / MISS_SCALE);
	}
};

class SaveReward : public RLGC::Reward {
public:
	static float ThreatAtOwnNet(const RLGC::GameState& state, Team team) {
		// ProjectShot aims at the net the given team ATTACKS, so pass the opponent.
		const Team attacker = (team == Team::BLUE) ? Team::ORANGE : Team::BLUE;

		const ShotProjection shot = ProjectShot(state, attacker);
		if (!shot.valid)
			return 0.f;

		return std::exp(-shot.missDist / ShotOnTargetReward::MISS_SCALE);
	}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep || !state.prev)
			return 0.f;

		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		return ThreatAtOwnNet(*state.prev, player.team) -
			   ThreatAtOwnNet(state, player.team);
	}
};

class WrongSurfaceReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		// isOnGround is >=3 wheels on ANY surface, so this is surface-relative for free.
		return (player.worldContact.hasContact && !player.isOnGround) ? -1.f : 0.f;
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
