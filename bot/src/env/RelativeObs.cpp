#include "RelativeObs.h"

#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/Math.h>

#include <algorithm>

using namespace RLGC;

namespace {

constexpr Vec POS_COEF = Vec(1 / CommonValues::SIDE_WALL_X,
                             1 / CommonValues::BACK_WALL_Y,
                             1 / CommonValues::CEILING_Z);
constexpr float VEL_COEF = 1 / CommonValues::CAR_MAX_SPEED;
constexpr float ANG_VEL_COEF = 1 / CommonValues::CAR_MAX_ANG_VEL;

// Express a world-frame vector in the car's own basis. RotMat's rows are the
// forward/right/up unit vectors, so this is the change of basis the network
// would otherwise have to learn from the raw orientation vectors.
inline Vec ToCarFrame(const Vec& v, const RotMat& rot) {
	return Vec(v.Dot(rot.forward), v.Dot(rot.right), v.Dot(rot.up));
}

// The block whose absence defined p8ref: where the target is and how it is
// moving, from the car's point of view.
//
// Ten floats: a unit direction (scale-free, and literally the argument of both
// dense reward terms), a separate distance, the full relative offset, and the
// relative velocity. Direction and offset are redundant with each other by
// construction; the unit vector is kept because it stays well-conditioned at
// every distance, while the offset carries the magnitude the unit vector drops.
void AddRelativeToObs(FList& obs, const PhysState& self, const PhysState& target) {
	const Vec offset = target.pos - self.pos;
	const float dist = offset.Length();

	const Vec dir = dist > 1e-4f ? offset / dist : Vec(0, 0, 0);
	obs += ToCarFrame(dir, self.rotMat);
	obs += dist / Hive::RelativeObs::RELATIVE_POS_SCALE;

	obs += ToCarFrame(offset, self.rotMat) / Hive::RelativeObs::RELATIVE_POS_SCALE;
	obs += ToCarFrame(target.vel - self.vel, self.rotMat) * VEL_COEF;
}

// Absolute block, byte-for-byte what DefaultObs::AddPlayerToObs emits, so the
// only difference between this builder and the old one is what is ADDED.
void AddPlayerToObs(FList& obs, const Player& player, bool inv) {
	const PhysState phys = InvertPhys(player, inv);

	obs += phys.pos * POS_COEF;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * VEL_COEF;
	obs += phys.angVel * ANG_VEL_COEF;

	obs += player.boost / 100;
	obs += player.isOnGround;
	obs += player.HasFlipOrJump();
	obs += player.isDemoed;
}

} // namespace

namespace Hive {

FList RelativeObs::BuildObs(const Player& player, const GameState& state) {
	FList result = {};

	const bool inv = player.team == Team::ORANGE;

	const PhysState ball = InvertPhys(state.ball, inv);
	const PhysState self = InvertPhys(player, inv);
	const auto& pads = state.GetBoostPads(inv);

	// --- Global, team frame (unchanged from DefaultObsPadded) ---------------
	result += ball.pos * POS_COEF;
	result += ball.vel * VEL_COEF;
	result += ball.angVel * ANG_VEL_COEF;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		result += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		result += (float)pads[i];

	// --- Self ---------------------------------------------------------------
	FList selfObs = {};
	AddPlayerToObs(selfObs, player, inv);
	result += selfObs;
	const int playerObsSize = static_cast<int>(selfObs.size());

	// --- Ball, in the car's frame -------------------------------------------
	AddRelativeToObs(result, self, ball);

	// --- Other cars: absolute block then relative block ----------------------
	// Both are emitted per slot so that padding zeroes an entire car rather
	// than leaving a relative block that reads as "a car exactly on top of me".
	std::vector<FList> teammates = {}, opponents = {};

	for (const auto& other : state.players) {
		if (other.carId == player.carId)
			continue;

		FList otherObs = {};
		AddPlayerToObs(otherObs, other, inv);
		AddRelativeToObs(otherObs, self, InvertPhys(other, inv));

		((other.team == player.team) ? teammates : opponents).push_back(otherObs);
	}

	if ((int)teammates.size() > maxPlayers - 1)
		RG_ERR_CLOSE("RelativeObs: too many teammates, maximum is " << (maxPlayers - 1));
	if ((int)opponents.size() > maxPlayers)
		RG_ERR_CLOSE("RelativeObs: too many opponents, maximum is " << maxPlayers);

	const int slotSize = playerObsSize + RELATIVE_BLOCK;
	for (int i = 0; i < 2; i++) {
		auto& list = i ? teammates : opponents;
		const int target = i ? maxPlayers - 1 : maxPlayers;
		while ((int)list.size() < target)
			list.push_back(FList(slotSize));
	}

	// Shuffled for the same reason DefaultObsPadded shuffles: a fixed slot
	// order teaches slot identity rather than what is in the slot.
	std::shuffle(teammates.begin(), teammates.end(), RocketSim::Math::GetRandEngine());
	std::shuffle(opponents.begin(), opponents.end(), RocketSim::Math::GetRandEngine());

	for (const auto& t : teammates)
		result += t;
	for (const auto& o : opponents)
		result += o;

	return result;
}

} // namespace Hive
