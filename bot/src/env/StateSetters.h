#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>

#include <memory>

namespace Hive {

// Wraps any spawner and, on a fraction of episodes, gives both cars a full tank
// that never drains (RocketSim's `boostUsedPerSecond = 0`).
//
// WHY. p10touch's bot found air dribbling off the wall and could only sustain
// it when it happened to have boost, running at `Player/Boost` **7.3 out of
// 100** for the whole run. A policy cannot learn the value of a resource it
// never has: with a near-empty tank, every aerial it attempts fails for a
// reason that has nothing to do with the aerial. This gives it a supply of
// episodes where the boost constraint is simply absent, so the behaviour can be
// discovered first and the economy learned second.
//
// The observation carries `boost / 100`, so the policy can tell an infinite
// episode from a normal one within a step or two of boosting and does not have
// to average the two regimes into one behaviour.
//
// MUST RESTORE. The mutator config is arena state, not episode state, so an
// arena that goes infinite stays infinite for every subsequent episode unless
// the normal rate is written back. That failure would be invisible -- the run
// would simply look like a bot that solved its boost problem. Asserted in
// bot/tests/test_statesetters.cpp.
class InfiniteBoostState : public RLGC::StateSetter {
public:
	// Takes ownership of the inner spawner.
	InfiniteBoostState(RLGC::StateSetter* inner, float chance)
		: inner(inner), chance(chance) {}

	void ResetArena(Arena* arena) override;

	// Whether the episode just spawned has infinite boost, for metrics.
	bool LastWasInfinite() const { return lastWasInfinite; }

private:
	std::unique_ptr<RLGC::StateSetter> inner;
	float chance;
	bool lastWasInfinite = false;
};

// Each setter spawns the arena into the *start* of one situation, so the
// policy gets a dense supply of it instead of waiting for it to occur
// naturally (a flip reset happens roughly never under random play).
//
// Design rules followed throughout:
//   * Always call arena->ResetToRandomKickoff() first, then overwrite what we
//     care about -- it resets boost pad timers, the ball, and every car.
//   * Randomise generously; a fixed scenario teaches memorisation, not skill.
//   * Set state for EVERY car, or leftover cars pollute the episode.
//   * Respect team symmetry: mirror positions by team.

// The jump-flip strike: ball at jump height, car already rolling at it and
// already at pace, so the only open decisions are timing and steering trim.
class StrikeState : public RLGC::StateSetter {
public:
	// Jump-reachable band, above where the bot can reach on wheels.
	float minBallZ = 250.f, maxBallZ = 550.f;

	// Close enough that contact is likely, far enough that the jump has to be
	// timed rather than mashed on spawn.
	float minDist = 700.f, maxDist = 1400.f;

	float minSpeed = 900.f, maxSpeed = 1600.f;

	// Full boost on a fraction of spawns, to connect "full tank" with "can hit
	// harder".
	float fullBoostChance = 0.4f;
	float minBoost = 20.f, maxBoost = 100.f;

	void ResetArena(Arena* arena) override;
};

// Ball off the ground, cars on the ground with boost. Teaches driving-to-takeoff
// and airborne ball contact. Height and distance are one setting, not two,
// because the curriculum instantiates this twice at different heights and
// spawn distance has to track ball height (too far and the ball is back on
// the ground before the car arrives).
class AerialState : public RLGC::StateSetter {
public:
	float minBallZ = 700.f, maxBallZ = 1700.f;
	float minCarDist = 1200.f, maxCarDist = 2600.f;
	float minBoost = 40.f, maxBoost = 100.f;

	void ResetArena(Arena* arena) override;
};

// Car and ball both airborne and travelling together, car just under the ball.
// Teaches carrying the ball through the air once contact is established.
class AirDribbleState : public RLGC::StateSetter {
public:
	float minZ = 400.f, maxZ = 1400.f;

	void ResetArena(Arena* arena) override;
};

// Car airborne under a high ball with its flip already used. Reaching the ball
// underside restores the flip -- that is the reset. Teaches the approach, not
// the follow-up.
class FlipResetState : public RLGC::StateSetter {
public:
	float minBallZ = 1100.f, maxBallZ = 1750.f;
	float carBelowBall = 300.f;

	void ResetArena(Arena* arena) override;
};

// Ball resting on the car roof, both moving forward together on the ground.
class GroundDribbleState : public RLGC::StateSetter {
public:
	float minSpeed = 700.f, maxSpeed = 1600.f;

	void ResetArena(Arena* arena) override;
};

// Two cars converging at speed with boost. Teaches bumps and demos, and
// teaches the receiving car to avoid them.
class DemoState : public RLGC::StateSetter {
public:
	float minSpeed = 1200.f, maxSpeed = 2200.f;
	float separation = 2500.f;

	void ResetArena(Arena* arena) override;
};

// Ball moving towards one team's goal with a defender positioned behind it.
// Teaches saves and shadow defence.
class DefendState : public RLGC::StateSetter {
public:
	float minBallSpeed = 800.f, maxBallSpeed = 2600.f;

	void ResetArena(Arena* arena) override;
};

// Cars tumbling in the air away from the ball. Teaches wave dashes, recoveries
// and landing on wheels -- unglamorous but a large share of real playtime.
class RecoverState : public RLGC::StateSetter {
public:
	float minZ = 300.f, maxZ = 1500.f;

	void ResetArena(Arena* arena) override;
};

// Ball and cars placed in plausible mid-play positions on the ground. This is
// the "everything else" setter and should carry most of the curriculum weight.
class NeutralPlayState : public RLGC::StateSetter {
public:
	void ResetArena(Arena* arena) override;
};

// One car spawned right next to the ball, on the ground, already pointed at
// it. Unlike other setters (which assume the bot can already drive), contact
// here is available within a second or two, so the touch reward fires often
// enough to reinforce "hit the ball" before the bot has learned to navigate
// to it. Not a permanent fixture -- weight should come down in favour of
// NeutralPlayState once touch ratio is healthy.
class BallContactState : public RLGC::StateSetter {
public:
	// Near enough that contact is close to unavoidable, far enough that the
	// car still has to steer and commit.
	float minDist = 250.f, maxDist = 700.f;

	// Fraction of spawns where the ball is already rolling rather than still.
	float movingBallChance = 0.6f;

	void ResetArena(Arena* arena) override;
};

} // namespace Hive
