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

// The ONLY place "how much did this touch move the ball at their net" is
// computed. TouchGoalAccelReward, ShotOnTargetReward and the Touch/Goal Accel
// telemetry all go through here.
//
// Signed and normalized to the same 130 kph (3611 uu/s) saturation everywhere,
// so the currency is unchanged whichever term consumes it.
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

// The ONLY place "is the ball heading for their half" is computed. AirTouchReward
// and the AirTouch/* telemetry both go through here, for the same reason
// ProjectShot exists: a metric that disagrees with the reward it audits is
// worse than no metric.
//
// Measured in the GROUND PLANE. The goal centre sits at z = 321 while this
// question is asked about balls near the ceiling, so a 3D direction would dock
// a legitimate high carry for not diving at the net.
inline float BallGoalwardCos(const RLGC::GameState& state, Team team) {
	const Vec flatVel = {state.ball.vel.x, state.ball.vel.y, 0.f};
	const float flatSpeed = flatVel.Length();

	// A ball going straight up has no horizontal opinion. Zero cos maps to a
	// neutral factor below, rather than to a penalty.
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

// Exponent 1 is the shipped curve. Raising it leaves both ends fixed --
// straight at the net is 1, straight back is 0 -- and only pulls the middle
// down, so a sideways carry can be made worth less without touching the aerial
// game at either extreme.
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

		// The direction factor, added 2026-08-20. Without it this term is
		// blind to which net the ball is heading for, and at 19.1% of reward
		// mass that made carrying the ball back into your own half pay exactly
		// what carrying it at the opponent's net pays.
		//
		// TouchGoalAccel cannot supply the missing signal on its own: it is
		// convex, so a 10x softer touch gives a 100x weaker direction signal,
		// and an air dribble is by definition a sequence of very soft touches.
		//
		// SMOOTH rather than a gate. A hard cutoff at cos = 0 would have no
		// gradient either side of it and would make a marginally misaimed
		// aerial worthless; this pays 1.0 straight at the net, 0.5 sideways
		// and 0 straight backwards, with gradient throughout.
		// Added 2026-08-20. Without this the term is blind to which net the
		// ball is heading for, and at 19.1% of reward mass that made carrying
		// the ball back into your own half pay exactly what carrying it at the
		// opponent's net pays. TouchGoalAccel cannot supply the signal on its
		// own: it is convex, so a 10x softer touch gives a 100x weaker
		// direction signal, and an air dribble is a sequence of soft touches.
		return base * GoalwardFactor(BallGoalwardCos(state, player.team),
									 directionExponent);
	}
};

class FlipSpeedReward : public RLGC::Reward {
public:
	// Contact flips are already paid by TouchGoalAccel; this term exists only
	// to price a flip used as TRAVEL, so it is blind inside this radius.
	static constexpr float MIN_BALL_DIST = 1500.f;

	// RLConst::FLIP_INITIAL_VEL_SCALE: a dodge is worth +500 uu/s whatever
	// direction it is aimed, so a full-value flip is one that keeps all of it.
	static constexpr float FULL_GAIN = 500.f;
	static constexpr float MIN_GAIN = 50.f;

	// Broadly ball-ward, not precisely: a travel flip is aimed at where play
	// is, and demanding better would re-create the v1 term's mistake of paying
	// only for flips already lined up on the ball.
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

		// isFlipping also clears on landing (Car.cpp:114) and the wheels scrub
		// speed on contact, so the peak over the dodge is what the flip
		// produced; the value after landing is not.
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

	// The jump itself adds JUMP_IMMEDIATE_FORCE straight up, which is not
	// travel, so only the ground plane counts.
	static float FlatSpeed(const Vec& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

	std::vector<Track> tracked;
};

// The ONLY place a shot is projected. Both ShotOnTargetReward and the
// `Shot/*` telemetry go through here, for the same reason MakeActionParser is
// the only construction site: a metric that disagrees with the reward it is
// meant to audit is worse than no metric.
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

	// Drag is left out: BALL_DRAG costs a few percent over a sub-3-second
	// flight and does not move a shot across the post. The floor clamp is not
	// an approximation but a correction -- a ballistic projection puts a
	// ground shot underground, and a ground shot scores.
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
	// One goal half-width, so a shot that misses by a full goal keeps e^-2 of
	// the payout. Never reaches zero: a wide shot must always have a gradient
	// pulling it back toward the mouth.
	static constexpr float MISS_SCALE = RLGC::CommonValues::GOAL_WIDTH_FROM_CENTER;

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		if (player.prev && player.prev->ballTouchedStep)
			return 0.f;

		const ShotProjection shot = ProjectShot(state, player.team);
		if (!shot.valid)
			return 0.f;

		// STRENGTH, added 2026-08-20. Without it this term was indifferent to
		// how hard the ball was hit, so a ball rolling goalward on the car's
		// hood paid exactly what a strike paid -- on every contact rising
		// edge. Walking the ball to the line was therefore strictly optimal
		// and slower was strictly better, which is what the bot learned.
		//
		// Keyed on the CHANGE this touch made, not the ball's speed: a carried
		// ball travels at the car's speed, so a speed factor would still have
		// paid a dribble handsomely, while its delta-v is ~0.
		//
		// LINEAR on purpose. TouchGoalAccel already pays for power convexly;
		// making this convex too would double-count it and re-create the
		// blast-it-goalward failure convexity invites.
		const float strength = RS_MAX(0.f, GoalwardDeltaV(state, player.team));
		if (strength <= 0.f)
			return 0.f;

		// PLACEMENT stays a plateau inside the mouth, so the corners pay
		// exactly what the centre pays. Corner shots are usually the right
		// shot; what is priced here is on-target against off-target, and Goal
		// is left to decide where within the mouth is best.
		return strength * std::exp(-shot.missDist / MISS_SCALE);
	}
};

class WrongSurfaceReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		// isOnGround is >=3 wheels on ANY surface, so this is surface-relative
		// for free: a car driving up a wall is on its wheels and pays nothing,
		// while a car on its roof or scraping its chassis pays every step.
		// Free flight makes no world contact, so it is not an air tax.
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
