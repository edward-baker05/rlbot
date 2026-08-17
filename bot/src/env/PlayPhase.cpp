#include "PlayPhase.h"

using namespace RLGC;

namespace Hive {

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
