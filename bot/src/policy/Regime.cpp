#include "Regime.h"

#include <RLGymCPP/CommonValues.h>

using namespace RLGC;

namespace Hive {

const char* RegimeName(Regime r) {
	switch (r) {
	case Regime::Kickoff: return "Kickoff";
	case Regime::General: return "General";
	default:              return "Unknown";
	}
}

const char* PlayPhaseName(PlayPhase p) {
	switch (p) {
	case PlayPhase::Aerial:        return "Aerial";
	case PlayPhase::AirDribble:    return "AirDribble";
	case PlayPhase::GroundDribble: return "GroundDribble";
	case PlayPhase::Defend:        return "Defend";
	case PlayPhase::Recover:       return "Recover";
	case PlayPhase::Neutral:       return "Neutral";
	default:                       return "Unknown";
	}
}

// ----------------------------------------------------------------------------
// Kickoff tracking
// ----------------------------------------------------------------------------

bool KickoffTracker::LooksLikeKickoffSpawn(const GameState& state) const {
	const Vec& p = state.ball.pos;

	// At rest. This is the discriminating test: the ball crosses the centre
	// spot constantly during play, but essentially never at zero velocity.
	if (state.ball.vel.Length() > t.ballSpeed)
		return false;

	// On the ground at the centre spot.
	if (p.z > t.ballHeight)
		return false;

	return Vec(p.x, p.y, 0).Length() < t.ballRadius;
}

Regime KickoffTracker::Update(const GameState& state, float deltaTime) {
	if (inKickoff) {
		elapsed += deltaTime;

		// The ball only moves once somebody hits it, so motion is the most
		// portable "first touch" signal -- it works identically in RocketSim
		// and against a live RLBot packet, with no touch bookkeeping needed.
		const bool ballMoved = state.ball.vel.Length() > t.ballSpeed;

		// Use explicit touch information as well when it is available, since it
		// fires a step earlier than the ball visibly accelerating.
		bool touched = (state.lastTouchCarID != -1);
		if (!touched) {
			for (const Player& pl : state.players) {
				if (pl.ballTouchedStep) {
					touched = true;
					break;
				}
			}
		}

		if (ballMoved || touched || elapsed > t.timeoutSeconds)
			inKickoff = false;

	} else if (LooksLikeKickoffSpawn(state)) {
		// Only a genuine reset can put the ball at rest at the centre spot.
		inKickoff = true;
		elapsed = 0.f;
	}

	return Current();
}

// ----------------------------------------------------------------------------
// Play phase classification (metrics only)
// ----------------------------------------------------------------------------

// Flip Y so negative always means "towards our own goal", whichever team we are.
static inline float OwnGoalRelY(float y, Team team) {
	return (team == Team::BLUE) ? y : -y;
}

PlayPhase ClassifyPhase(const Player& player, const GameState& state, const PhaseThresholds& t) {
	const Vec& ballPos = state.ball.pos;
	const float ballDist = (ballPos - player.pos).Length();
	const bool airborne = !player.isOnGround && player.pos.z > t.airborneZ;

	// Ordered most specific first.
	if (airborne && ballDist < t.ballNearDist && ballPos.z > t.airDribbleBallZ)
		return PlayPhase::AirDribble;

	if (airborne && ballPos.z > t.aerialBallZ)
		return PlayPhase::Aerial;

	if (player.isOnGround && ballDist < t.dribbleDist &&
	    ballPos.z < t.dribbleBallZMax && ballPos.z > player.pos.z)
		return PlayPhase::GroundDribble;

	// Checked before Defend so a tumbling car is not counted as shadowing.
	if (!player.isOnGround && ballDist > t.ballNearDist * 2.f)
		return PlayPhase::Recover;

	if (OwnGoalRelY(ballPos.y, player.team) < t.defendThirdY)
		return PlayPhase::Defend;

	return PlayPhase::Neutral;
}

} // namespace Hive
