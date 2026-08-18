#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <rlbot/Bot.h>

#include <unordered_map>
#include <vector>

namespace Hive {

// Translates an RLBot v5 GamePacket into the RLGymCPP GameState the policy was
// trained on. Errors here are silent -- the bot just plays worse than in
// training -- so two traps get explicit handling instead of relying on
// upstream agreement: boost pad order (RLGymCPP's hardcoded table vs RLBot's
// y-then-x order, reconciled via an index map built from FieldInfo) and flip
// availability (RocketSim derives it from internal fields, which we set to
// match RLBot's dodge_timeout ground truth).

class PacketConverter {
public:
	// Build the boost pad index map from FieldInfo. Call once, from
	// Bot::initialize(). Safe to call with nullptr (falls back to identity
	// mapping and logs a warning).
	void Initialize(const rlbot::flat::FieldInfo* fieldInfo);

	// Convert a packet into a GameState.
	RLGC::GameState Convert(const rlbot::flat::GamePacket* packet);

	void Reset() { lastTouchTimes.clear(); }

private:
	// rlgymIndex[i] = index into RLBot's boost_pads array for RLGymCPP pad i.
	std::vector<int> rlgymToRLBotPad;

	// player_id -> game_seconds of that player's most recent registered touch,
	// used to turn RLBot's "latest touch" into a per-step "touched this step".
	std::unordered_map<int, float> lastTouchTimes;
};

} // namespace Hive
