#include "RelativeObs.h"

#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/Math.h>

#include <algorithm>
#include <cmath>

using namespace RLGC;

namespace {
constexpr Vec POS_COEF = Vec(1 / CommonValues::SIDE_WALL_X,
                             1 / CommonValues::BACK_WALL_Y,
                             1 / CommonValues::CEILING_Z);
constexpr float VEL_COEF = 1 / CommonValues::CAR_MAX_SPEED;
constexpr float ANG_VEL_COEF = 1 / CommonValues::CAR_MAX_ANG_VEL;

inline Vec ToCarFrame(const Vec& v, const RotMat& rot) {
	return Vec(v.Dot(rot.forward), v.Dot(rot.right), v.Dot(rot.up));
}

void AddRelativeToObs(FList& obs, const PhysState& self, const PhysState& target) {
	const Vec offset = target.pos - self.pos;
	const float dist = offset.Length();
	const Vec dir = dist > 1e-4f ? offset / dist : Vec(0, 0, 0);
	obs += ToCarFrame(dir, self.rotMat);
	obs += dist / Hive::RelativeObs::RELATIVE_POS_SCALE;

	obs += ToCarFrame(offset, self.rotMat) / Hive::RelativeObs::RELATIVE_POS_SCALE;
	obs += ToCarFrame(target.vel - self.vel, self.rotMat) * VEL_COEF;
}

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

}  // namespace

namespace Hive {

FList RelativeObs::BuildObs(const Player& player, const GameState& state) {
	FList result = {};
	const bool inv = player.team == Team::ORANGE;
	const PhysState ball = InvertPhys(state.ball, inv);
	const PhysState self = InvertPhys(player, inv);
	const auto& pads = state.GetBoostPads(inv);

	result += ball.pos * POS_COEF;
	result += ball.vel * VEL_COEF;
	result += ball.angVel * ANG_VEL_COEF;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		result += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		result += (float)pads[i];

	FList selfObs = {};
	AddPlayerToObs(selfObs, player, inv);
	result += selfObs;
	const int playerObsSize = static_cast<int>(selfObs.size());

	AddRelativeToObs(result, self, ball);

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

	std::shuffle(teammates.begin(), teammates.end(), RocketSim::Math::GetRandEngine());
	std::shuffle(opponents.begin(), opponents.end(), RocketSim::Math::GetRandEngine());

	for (const auto& t : teammates)
		result += t;
	for (const auto& o : opponents)
		result += o;

	uint64_t nonFinite = 0;
	for (float& v : result) {
		if (!std::isfinite(v)) {
			v = 0.f;
			nonFinite++;
		}
	}
	NoteObsHealth(result.size(), nonFinite);

	return result;
}

}  // namespace Hive
