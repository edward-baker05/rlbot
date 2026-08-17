#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <rlbot/Bot.h>

#include <unordered_map>
#include <vector>

namespace Hive {

// ============================================================================
// PacketConverter
// ============================================================================
// Translates an RLBot v5 GamePacket into the RLGymCPP GameState the policy was
// trained on.
//
// This is the highest-risk file in the deployment path, because every error in
// it is silent. The bot loads, connects, drives around, and simply plays worse
// than it did in training -- there is no crash and no warning. The two classic
// ways to get it wrong:
//
//   1. BOOST PAD ORDER. The observation includes 34 boost pad states. RLGymCPP
//      indexes them by its own hardcoded location table; RLBot orders them by
//      y then x. These orders happen to agree today, but relying on that is a
//      silent-corruption bug waiting to happen. We build an explicit index map
//      from FieldInfo locations instead, once, at connect time.
//
//   2. FLIP AVAILABILITY. The observation includes HasFlipOrJump(), which
//      RocketSim derives from several internal fields. RLBot reports the same
//      fact directly via dodge_timeout. We set the internal fields to whatever
//      makes RocketSim's derivation agree with RLBot's ground truth, rather
//      than trying to reconstruct the internal state.
// ============================================================================

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
