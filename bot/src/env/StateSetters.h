#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>

namespace Hive {

// ============================================================================
// Situation state setters
// ============================================================================
// Each setter spawns the arena into the *start* of one situation, so the policy
// gets a dense supply of that situation instead of waiting for it to occur
// naturally. This is the single highest-leverage tool you have: a flip reset
// happens roughly never under random play, so without a setter the policy will
// never see enough of them to learn anything.
//
// Design rules followed throughout:
//   * Always call arena->ResetToRandomKickoff() first. It resets boost pads,
//     the ball, and every car to a known-good state; we then overwrite what we
//     care about. Skipping it leaves stale boost pad timers.
//   * Randomise generously. A setter that always spawns the identical scenario
//     teaches the policy to memorise one trajectory rather than learn a skill.
//   * Set state for EVERY car, not just one, or leftover cars sit at kickoff
//     positions and pollute the episode.
//   * Respect team symmetry: mirror positions by team so both sides see the
//     same distribution.
// ============================================================================

// Ball high in the air, cars on the ground with boost. Teaches driving-to-aerial
// takeoff and mid-air ball contact.
class AerialState : public RLGC::StateSetter {
public:
	float minBallZ = 700.f, maxBallZ = 1700.f;
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

// One car spawned right next to the ball, on the ground, already pointed at it.
//
// This exists to break a specific deadlock. The touch-quality reward is the
// main outcome signal of phase 1, but it only pays when the ball is actually
// struck -- and a fresh policy touches the ball roughly once every twenty
// seconds. A reward that almost never fires cannot shape behaviour, so the bot
// has no gradient towards the very skill the phase is meant to teach.
//
// Every other setter spawns cars 1200-2600 uu away, which assumes the bot can
// already drive. This one does not: contact is available within a second or
// two, so the touch reward fires often enough to reinforce, and the policy can
// learn "when the ball is in front of you, hit it" before it has learned to
// navigate to it at all.
//
// Deliberately not a permanent fixture. Once touch ratio is healthy this is
// training on a situation the bot gets for free, and its weight should come
// down in favour of NeutralPlayState. See docs/rewards.md.
class BallContactState : public RLGC::StateSetter {
public:
	// Spawn distance from the ball. Near enough that contact is close to
	// unavoidable, far enough that the car still has to steer and commit
	// rather than starting already inside the ball.
	float minDist = 250.f, maxDist = 700.f;

	// Fraction of spawns where the ball is already rolling rather than sitting
	// still. A stationary ball is the easier lesson; a moving one is what the
	// game actually presents.
	float movingBallChance = 0.6f;

	void ResetArena(Arena* arena) override;
};

} // namespace Hive
